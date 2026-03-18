# inspiration:
# 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py  # noqa
# 2. https://github.com/karpathy/minGPT
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Optional, Tuple

import traceback
import time
import json
import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import wandb
import utils.functions as func
import utils.dt_functions as dt_func

from torch.nn import functional as F
from tqdm.auto import trange  # noqa
from dataclasses import dataclass
from utils.logger import init_logger, Logger
from utils.attack import Evaluation_Attacker
from datetime import datetime

MODEL_PATH = {
    "IQL": os.path.join(os.path.dirname(os.path.dirname(__file__)), "IQL_model"),
}


@dataclass
class TrainConfig:
    # Experiment
    eval_every: int = 10
    n_episodes: int = 10  # How many episodes run during evaluation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 100
    eval_final: int =100
    num_updates_on_epoch: int = 1000
    # model params
    embedding_dim: int = 128 # 768
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.0
    residual_dropout: float = 0.1
    embedding_dropout: float = None
    # training params
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 100_000
    reward_scale: float = 0.001
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    recalculate_return: bool = False
    # evaluation params
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    eval_id: str = "attack"
    eval_only: bool = False
    eval_attack: bool = False
    checkpoint_dir: str = None
    test_time: str = None
    # Wandb logging
    use_wandb: int = 1
    group: str = os.path.basename(__file__).rstrip(".py")
    env: str = ""
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    # dataset
    down_sample: bool = True
    sample_ratio: float = 1.0
    # others
    debug: bool = False
    alg_type: str = os.path.basename(__file__).rstrip(".py")
    logdir: str = "results"
    dataset_path: str = "/home/user/.d4rl/datasets"
    save_model: bool = False
    debug_eval: bool = False
    # corruption
    corruption_agent: str = "IQL"
    corruption_seed: int = 0
    corruption_mode: str = ""  # random, adversarial
    corruption_tag: str = "" # obs, act, rew
    corruption_next_obs: float = 0.0  # 0 or 1
    corruption_rate: float = 0.3
    use_original: int = 0  # 0 or 1
    same_index: int = 0
    froce_attack: int = 0

    def __post_init__(self):
        # train
        if not self.eval_only:
            if self.corruption_tag == "obs":
                self.corruption_obs = 1.0
                self.corruption_act = 0.0
                self.corruption_rew = 0.0
            if self.corruption_tag == "act":
                self.corruption_obs = 0.0
                self.corruption_act = 1.0
                self.corruption_rew = 0.0
            if self.corruption_tag == "rew":
                self.corruption_obs = 0.0
                self.corruption_act = 0.0
                self.corruption_rew = 1.0  
            # target_returns and reward_scale
            if self.env.startswith("antmaze"):
                self.target_returns = [1.0, 0.5]
                self.reward_scale = 1.0
            if self.env.startswith("hopper"):
                self.target_returns = [3600, 1800]
                self.reward_scale = 0.001
            if self.env.startswith("halfcheetah"):
                self.target_returns = [12000, 6000]
                self.reward_scale = 0.001
            if self.env.startswith("walker"):
                self.target_returns = [5000, 2500]
                self.reward_scale = 0.001
            if self.env.startswith("kitchen"):
                self.target_returns = [400, 500] # 500
                self.reward_scale = 1.0
            if self.env.startswith("door"):
                self.target_returns = [2900, 1450]
                self.reward_scale = 1.0
            if self.env.startswith("pen"):
                self.target_returns = [3100, 1550]
                self.reward_scale = 1.0
            if self.env.startswith("hammer"):
                self.target_returns = [12800, 6400]
                self.reward_scale = 1.0
            if self.env.startswith("relocate"):
                self.target_returns = [4300, 2150]
                self.reward_scale = 1.0
            # sample ratio
            if self.down_sample:
                if self.env.startswith("kitchen"):
                    self.sample_ratio = 1.0
                if "medium-expert" in self.env:
                    self.sample_ratio = 0.01
                if "medium-replay" in self.env:
                    self.sample_ratio = 0.1
                if "medium-v2" in self.env:
                    self.sample_ratio = 0.02
                key = self.env.split("-")[0]
                if key in ["door", "pen", "hammer", "relocate"]:
                    self.sample_ratio = 0.01
            # others
            if self.env.startswith("antmaze"):
                self.num_epochs = 100
                self.n_episodes = 100
            if self.corruption_mode == "random" and self.corruption_rew > 0.0:
                self.corruption_rew *= 30
            # auto set
            self.update_steps = int(self.num_epochs * self.num_updates_on_epoch)
            self.warmup_steps = int(0.1 * self.update_steps)
            self.decay_steps = int(0.1 * self.update_steps)
        # evaluation
        if self.eval_only:
            assert self.checkpoint_dir is not None, "Please provide checkpoint_dir for evaluation."
            self.checkpoint_dir = os.path.join(self.logdir, self.group, self.env, self.checkpoint_dir)
            with open(os.path.join(self.checkpoint_dir, "params.json"), "r") as f:
                config = json.load(f)
            unoverwritten_keys = ["eval_id", "test_time", "group", "checkpoint_dir", "eval_only", "eval_attack", "eval_attack_mode", "eval_attack_eps", "eval_corruption_rate"]
            for key, value in config.items():
                if key not in unoverwritten_keys:
                    try:
                        value = eval(value)
                    except:
                        pass
                    self.__dict__[key] = value
                    # print(f"Set {key} to {value}")
        self.eval_attack_mode = self.corruption_mode # random, adversarial
        self.eval_attack_eps = 1
        self.eval_corruption_rate = 0.3
        if self.eval_attack_mode == "random" and self.corruption_tag == "rew":
            self.eval_attack_eps *= 30


def set_model(config: TrainConfig):
    model = dt_func.DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
    ).to(config.device)
    return model


def train(config: TrainConfig, logger: Logger):
    # Set seeds
    func.set_seed(config.seed)

    if config.use_wandb:
        func.wandb_init(config)

    env = gym.make(config.env)
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]
    config.max_action = float(env.action_space.high[0])
    config.action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
    ]

    # data & dataloader setup
    dataset = dt_func.SequenceDataset(config, logger)
    logger.info(f"Dataset: {len(dataset.dataset)} trajectories")
    # logger.info(f"State mean: {dataset.state_mean}, std: {dataset.state_std}")

    env = func.wrap_env(
        env,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    env.seed(config.seed)

    # model
    model = set_model(config)
    # logger.info(f"Network: \n{str(model)}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # optimizer
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )

    if config.eval_attack:
                state_std, act_std, rew_std, rew_min = func.get_state_std(config)
                eval_attacker = Evaluation_Attacker(
                    config, config.env, config.corruption_agent, config.eval_attack_eps,
                    config.state_dim, config.action_dim, state_std, act_std, rew_std, rew_min, config.eval_attack_mode,
                    MODEL_PATH[config.corruption_agent],
                )
                print("eval_attack: True")
    else:
        eval_attacker = None
        print("eval_attack: False")
    
    if config.debug_eval:
        model.eval()
        eval_log = dt_func.eval_fn(config, env, model, eval_attacker)
        model.train()
        logger.record("epoch", 0)
        for k, v in eval_log.items():
            logger.record(k, v)
        logger.dump(0)
    
    # if config.use_wandb:
    #     wandb.log({"epoch": 0, **eval_log})

    total_updates = 0
    best_score = -np.inf
    best_score_50 = -np.inf
    # trainloader_iter = iter(trainloader)
    for epoch in trange(1, config.num_epochs + 1, desc="Training"):
        time_start = time.time()
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            log_dict = {}
            # batch = next(trainloader_iter)
            batch = dataset.get_batch(config.batch_size)
            states, actions, returns, _, time_steps, mask, _, _ = [b.to(config.device) for b in batch]
            # True value indicates that the corresponding key value will be ignored
            padding_mask = ~mask.to(torch.bool)

            predicted = model(
                states=states,
                actions=actions,
                returns_to_go=returns,
                time_steps=time_steps,
                padding_mask=padding_mask,
            )
            predicted_actions = predicted[0]

            optim.zero_grad()
            loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
            # [batch_size, seq_len, d] * [batch_size, seq_len, 1]
            loss = (loss * mask.unsqueeze(-1)).mean()
            loss.backward()
            if config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optim.step()
            log_dict.update({"loss_action": loss.item()})

            log_dict.update({"learning_rate": scheduler.get_last_lr()[0]})
            scheduler.step()
            total_updates += 1

        time_end = time.time()
        epoch_time = time_end - time_start

        # validation in the env for the actual online performance
        if epoch % config.eval_every == 0 and epoch > (config.num_epochs - config.eval_final):
            model.eval()
            eval_log = dt_func.eval_fn(config, env, model)
            model.train()
            logger.record("epoch", epoch)
            logger.record("epoch_time", epoch_time)
            for k, v in eval_log.items():
                logger.record(k, v)
            for k, v in log_dict.items():
                logger.record(f"update/{k}", v)
            logger.record("update/gradient_step", total_updates)
            logger.dump(epoch)
            if config.save_model:
                torch.save(
                    model.state_dict(),
                    os.path.join(logger.get_dir(), f"policy_{epoch}.pth"),
                )

            if config.use_wandb:
                update_log = {f"update/{k}": v for k, v in log_dict.items()}
                wandb.log({"epoch": epoch, **update_log})
                wandb.log({"epoch": epoch, **eval_log})

            now_score = max(eval_log[f"eval/{config.target_returns[0]}_normalized_score_mean"], eval_log[f"eval/{config.target_returns[1]}_normalized_score_mean"])
            with open(os.path.join(logger.get_dir(), "eval_scores.txt"), "a") as f:
                    f.write(f"{now_score:.4f}_{epoch}\n")
            if config.save_model and now_score > best_score:
                best_score = now_score
                with open(os.path.join(logger.get_dir(), "best_score.txt"), "w") as f:
                        f.write(f"{best_score:.4f}_{epoch}")
                if config.save_model:
                    torch.save(
                        model.state_dict(),
                        os.path.join(logger.get_dir(), f"best_policy.pth"),
                    )
            if epoch > config.num_epochs - 50:
                if now_score > best_score_50:
                    best_score_50 = now_score
                    with open(os.path.join(logger.get_dir(), "best_score_50.txt"), "w") as f:
                            f.write(f"{best_score_50:.4f}_{epoch}")
                    if config.save_model:
                        torch.save(
                            model.state_dict(),
                            os.path.join(logger.get_dir(), f"best_policy_50.pth"),
                        )
            if epoch == config.num_epochs:
                with open(os.path.join(logger.get_dir(), "final_score.txt"), "w") as f:
                        f.write(f"{now_score:.4f}_{epoch}")
                if config.save_model:
                    torch.save(
                        model.state_dict(),
                        os.path.join(logger.get_dir(), f"final_policy.pth"),
                    )


def test(config: TrainConfig, logger: Logger):
    # Set seeds
    func.set_seed(config.seed)

    env = gym.make(config.env)
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]
    config.max_action = float(env.action_space.high[0])
    config.action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
    ]

    # data & dataloader setup
    dataset = dt_func.SequenceDataset(config, logger)
    logger.info(f"Dataset: {len(dataset.dataset)} trajectories")
    # logger.info(f"State mean: {dataset.state_mean}, std: {dataset.state_std}")

    env = func.wrap_env(
        env,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    env.seed(config.seed)
    
    if config.eval_attack:
        state_std, act_std, rew_std, rew_min = func.get_state_std(config)
        eval_attacker = Evaluation_Attacker(
            config, config.env, config.corruption_agent, config.eval_attack_eps,
            config.state_dim, config.action_dim, state_std, act_std, rew_std, rew_min, config.eval_attack_mode,
            MODEL_PATH[config.corruption_agent],
        )
        print("eval_attack: True")
    else:
        eval_attacker = None
        print("eval_attack: False")
            
    all_files = os.listdir(config.checkpoint_dir)
    model_epoches = [
        f for f in all_files 
        if f.startswith("policy") and f.endswith(".pth")
    ]
    model_epoches.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))

    best_score = -np.inf
    best_score_50 = -np.inf
    for i, model_epoch in enumerate(model_epoches):
        epoch = int(model_epoch.split(".")[0].split("_")[1])
        print(f"eval epoch: {epoch}")
        
        # model
        model = set_model(config)
        model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, model_epoch)), strict=False)
        model.eval()
        # logger.info(f"Network: \n{str(model)}")
        # logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
        eval_log = dt_func.eval_fn(config, env, model, eval_attacker)
        for k, v in eval_log.items():
            logger.record(k, v)
        logger.dump(0)
    
        now_score = max(eval_log[f"eval/{config.target_returns[0]}_normalized_score_mean"], eval_log[f"eval/{config.target_returns[1]}_normalized_score_mean"])
        if i == 0:
            with open(os.path.join(logger.get_dir(), "eval_scores.txt"), "w") as f:
                f.write(f"{now_score:.4f}_{epoch}\n")
        if i > 0:
            with open(os.path.join(logger.get_dir(), "eval_scores.txt"), "a") as f:
                f.write(f"{now_score:.4f}_{epoch}\n")
        if now_score > best_score:
            best_score = now_score
            with open(os.path.join(logger.get_dir(), "best_score.txt"), "w") as f:
                f.write(f"{best_score:.4f}_{epoch}")
        if epoch > config.num_epochs - 50:
            if now_score > best_score_50:
                best_score_50 = now_score
                with open(os.path.join(logger.get_dir(), "best_score_50.txt"), "w") as f:
                    f.write(f"{best_score_50:.4f}_{epoch}")  
        if epoch == config.num_epochs:
            with open(os.path.join(logger.get_dir(), "final_score.txt"), "w") as f:
                f.write(f"{now_score:.4f}_{epoch}")  

@pyrallis.wrap()
def main(config: TrainConfig):
    logger = init_logger(config)
    # train(config, logger)
    try:
        if config.eval_only:
            test(config, logger)
        else:
            train(config, logger)
    except Exception:
        error_info = traceback.format_exc()
        logger.error(f"\n{error_info}")


if __name__ == "__main__":
    main()