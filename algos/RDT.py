import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Optional, Tuple

import os
import time
import json
import traceback
import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import wandb
import utils.functions as func
import utils.dt_functions as dt_func

from tqdm.auto import trange  # noqa
from torch.nn import functional as F
from dataclasses import dataclass
from utils.logger import init_logger, Logger
from utils.attack import Evaluation_Attacker
from utils.run_mean_std import RunningMeanStd
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
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.0
    residual_dropout: float = 0.1
    embedding_dropout: float = None
    mlp_embedding: bool = False
    mlp_head: bool = False
    mlp_reward: bool = True
    embed_order: str = "rsa"  # rsa, sar
    # training params
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    reward_scale: float = 0.001
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    loss_fn: str = "wmse" # mse, wmse
    wmse_coef: float = (0.0, 0.0) # (act, rew)
    reward_coef: float = 1.0
    recalculate_return: bool = False
    correct_freq: int = 1
    correct_start: int = 50
    correct_thershold: Tuple[float] = None # (act, rew)
    # evaluation params
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    eval_id: str = "attack"
    eval_only: bool = False
    eval_attack: bool = False
    eval_checkpoint: str = "final"
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
    corruption_seed: int = 0 # 2023
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
            # RDT params
            if "medium-replay" in self.env and self.sample_ratio == 0.1:
                self.reward_coef = 1.0
                self.wmse_coef = (1.0, 1.0)
                self.embedding_dropout = 0.2
                if self.corruption_obs > 0.0:
                    self.correct_thershold = None
                if self.corruption_act > 0.0:
                    self.correct_thershold = (6.0, 0.0)
                if self.corruption_rew > 0.0:
                    self.correct_thershold = (0.0, 6.0)
                if self.corruption_rew > 0.0 and self.corruption_obs > 0.0 and self.corruption_act > 0.0:
                    self.wmse_coef = (0.1, 0.1)
                    self.correct_thershold = (6.0, 6.0)
                    self.correct_freq = 50
            if self.env.startswith("kitchen"):
                self.reward_coef = 0.1
                self.wmse_coef = (30.0, 30.0)
                self.embedding_dropout = 0.1
                self.correct_thershold = (5.0, 0.0)
                if self.corruption_obs > 0.0:
                    self.correct_thershold = None
                if self.corruption_act > 0.0:
                    self.correct_thershold = (5.0, 0.0)
                if self.corruption_rew > 0.0:
                    self.correct_thershold = (0.0, 6.0)
                if self.corruption_rew > 0.0 and self.corruption_obs > 0.0 and self.corruption_act > 0.0:
                    self.embedding_dropout = 0.2
                    self.correct_thershold = (6.0, 6.0)
                    self.correct_freq = 10
            if self.env.split("-")[0] in ["door", "pen", "hammer", "relocate"]:
                self.reward_coef = 1.0
                self.embedding_dropout = 0.1
                if self.corruption_obs > 0.0:
                    self.wmse_coef = (0.1, 0.0)
                    self.correct_thershold = None
                if self.corruption_act > 0.0:
                    self.wmse_coef = (10.0, 0.0)
                    self.correct_thershold = (6.0, 0.0)
                if self.corruption_rew > 0.0:
                    self.wmse_coef = (0.1, 0.1)
                    self.correct_thershold = (0.0, 6.0)
                if self.corruption_rew > 0.0 and self.corruption_obs > 0.0 and self.corruption_act > 0.0:
                    self.wmse_coef = (0.1, 0.1)
                    self.correct_thershold = (6.0, 6.0)
                    self.correct_freq = 10
            # auto set
            self.update_steps = int(self.num_epochs * self.num_updates_on_epoch)
            self.warmup_steps = int(0.1 * self.update_steps)
            self.decay_steps = int(0.1 * self.update_steps)
        # evaluation
        if self.eval_only:
            assert self.checkpoint_dir is not None, "Please provide checkpoint_dir for evaluation."
            self.checkpoint_dir = os.path.join(self.logdir, self.group, self.env, self.checkpoint_dir)
            with open(os.path.join(self.checkpoint_dir, "params.json"), "r") as f:
                train_config = json.load(f)
            unoverwritten_keys = ["eval_id", "test_time", "group", "checkpoint_dir", "eval_only", "eval_attack", "eval_attack_mode", "eval_attack_eps", "eval_corruption_rate"]
            for key, value in train_config.items():
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
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        mlp_embedding=config.mlp_embedding,
        mlp_head=config.mlp_head,
        mlp_reward=config.mlp_reward,
        embed_order=config.embed_order,
        predict_reward=True,
    ).to(config.device)
    return model


def loss_fn(config, predicted, target, mask, coef=None):
    if config.loss_fn == "mse":
        loss = F.mse_loss(predicted, target.detach(), reduction="none")
    elif config.loss_fn == "wmse": # New
        with torch.no_grad():
            diff = torch.square(predicted.detach() - target.detach()).mean(-1, keepdim=True)
            weight = torch.exp(-coef * diff)
        loss = F.mse_loss(predicted, target.detach(), reduction="none")
        loss = (loss * weight)
    loss = (loss * mask.unsqueeze(-1)).mean()
    return loss


def correct_outliers(config, data_info, data_dist, correct=False):  # New
    mask = data_info.pop("mask")
    time_steps = data_info.pop("time_steps")
    traj_indexs = data_info.pop("traj_indexs")

    attack_mask = data_info.pop("attack_mask")
    attack_num = len(torch.where(attack_mask == 1)[0])

    correct_info, correct_dict = {}, {}
    for name, data, dist in zip(data_info.keys(), data_info.values(), data_dist):
        if dist is not None:
            predicted, target = data
            with torch.no_grad():
                diff = torch.square(predicted.detach() - target.detach()).sum(-1)
            diff = diff.cpu().numpy()
            if correct:
                z_scores = (diff - dist.mean) / np.sqrt(dist.var)
                outlier_indices = np.where(np.abs(z_scores) > dist.thershold)
                correct_traj_indexs = traj_indexs.index_select(0, torch.tensor(outlier_indices[0], device=config.device))
                correct_time_steps = time_steps[outlier_indices]
                correct_data = predicted[outlier_indices]
                correct_info[name] = {
                    "traj_indexs": correct_traj_indexs.tolist(),
                    "time_steps": correct_time_steps.tolist(),
                    "correct_data": correct_data.detach().cpu().numpy(),
                }

                outlier_num = torch.where(mask[outlier_indices] == 1)[0].shape[0]
                correct_num = torch.where(attack_mask[outlier_indices] == 1)[0].shape[0]
                correct_ratio = correct_num / (outlier_num + 1e-6) # recall
                attack_ratio = correct_num / (attack_num + 1e-6) # precision
                correct_dict.update({
                    f"{name}/mean": dist.mean, f"{name}/var": dist.var,
                    f"{name}/outlier_num": outlier_num, f"{name}/correct_num": correct_num,
                    f"{name}/correct_recall": correct_ratio, f"{name}/correct_precision": attack_ratio
                })
            diff = diff[np.where(mask.cpu().numpy() == 1)]
            dist.update(diff)
    return correct_info, correct_dict


def compute_loss(config, model, batch):
    log_dict, debug_dict = {}, {}
    states, actions, returns, rewards, time_steps, mask, attack_mask, traj_indexs = [b.to(config.device) for b in batch]
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
    predicted_rewards = predicted[1]

    if config.debug:
        with torch.no_grad():
            diff = torch.square(predicted_actions.detach() - actions.detach()).sum(-1, keepdim=True)
            att_diff = diff[attack_mask == 1].mean()
            ori_diff = diff[attack_mask == -1].mean()
            debug_dict.update({"att_diff": att_diff.item(), "ori_diff": ori_diff.item()})

    loss = loss_fn(config, predicted_actions, actions, mask, config.wmse_coef[0])
    log_dict.update({"loss_action": loss.item()})
    loss_reward = loss_fn(config, predicted_rewards, rewards, mask, config.wmse_coef[1])
    loss += config.reward_coef * loss_reward
    log_dict.update({"loss_reward": loss_reward.item()})

    log_dict.update({"policy_loss": loss.item()})

    data_info = None
    if config.correct_thershold is not None:
        data_info = {
            "actions": [predicted_actions, actions], "rewards": [predicted_rewards, rewards],
            "mask": mask,  "attack_mask": attack_mask,
            "traj_indexs": traj_indexs, "time_steps": time_steps,
        }
    return loss, log_dict, debug_dict, data_info


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

    data_dist = []
    if config.correct_thershold is not None:
        for thershold in config.correct_thershold:
            data_dist.append(RunningMeanStd(thershold=thershold) if thershold > 0.0 else None)

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
        for step in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            # batch = next(trainloader_iter)
            batch = dataset.get_batch(config.batch_size, config.recalculate_return)
            config.recalculate_return = False

            optim.zero_grad()
            loss, log_dict, debug_dict, data_info = compute_loss(config, model, batch)
            loss.backward()
            if config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optim.step()
            log_dict.update({"learning_rate": scheduler.get_last_lr()[0]})
            scheduler.step()
            total_updates += 1
            correct_dict = {}
            if config.correct_thershold is not None:
                correct = epoch > config.correct_start and step % config.correct_freq == 0
                correct_info, correct_dict = correct_outliers(config, data_info, data_dist, correct=correct)
                if correct:
                    for name, info in correct_info.items():
                        dataset.correct(info["traj_indexs"], info["time_steps"], info["correct_data"], name)
                    if config.correct_thershold is not None and config.correct_thershold[1] > 0.0:
                        config.recalculate_return = True

        time_end = time.time()
        epoch_time = time_end - time_start

        # validation in the env for the actual online performance
        if epoch % config.eval_every == 0 and epoch > (config.num_epochs - config.eval_final):
            model.eval()
            eval_log = dt_func.eval_fn(config, env, model, eval_attacker)
            model.train()
            logger.record("epoch", epoch)
            logger.record("epoch_time", epoch_time)
            for k, v in eval_log.items():
                logger.record(k, v)
            for k, v in log_dict.items():
                logger.record(f"update/{k}", v)
            logger.record("update/gradient_step", total_updates)
            for k, v in debug_dict.items():
                logger.record(f"debug/{k}", v)
            for k, v in correct_dict.items():
                logger.record(f"correct/{k}", v)
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
            if now_score > best_score:
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