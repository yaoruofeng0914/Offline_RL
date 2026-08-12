# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import os, sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Any, Dict, List, Optional

import time
import json
import wandb
import traceback
import pyrallis
import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
import utils.functions as func

from torch.distributions import MultivariateNormal
from dataclasses import dataclass
from pathlib import Path
from tqdm import trange
from utils.logger import init_logger, Logger
from utils.attack import attack_dataset
from utils.replay_buffer import ReplayBuffer
from utils.networks import MLP
from utils.attack import Evaluation_Attacker
from datetime import datetime
from utils.dt_functions import NSAOPObsAttacker, NSAOPActAttacker, NSAOPRewAttacker
MODEL_PATH = {
    "IQL": os.path.join(os.path.dirname(os.path.dirname(__file__)), "IQL_model"),
}

TensorBatch = List[torch.Tensor]

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Experiment
    eval_every: int = 10
    eval_episodes: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 1000
    eval_final: int = 100
    num_updates_on_epoch: int = 1000
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # model params
    n_hidden: int = 2
    hidden_dim: int = 256
    # IQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    deterministic: bool = True  # Use deterministic actor
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # Wandb logging
    use_wandb: int = 1
    group: str = os.path.basename(__file__).rstrip(".py")
    env: str = ""
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    # evaluation params
    eval_id: str = "attack"
    eval_only: bool = False
    eval_attack: bool = False
    checkpoint_dir: str = None
    test_time: str = None
    # others
    alg_type: str = os.path.basename(__file__).rstrip(".py")
    logdir: str = "results"
    dataset_path: str = os.path.expanduser("~/Offline_RL/")
    sample_ratio: float = 1.0
    save_model: bool = False
    debug_eval: bool = False
    # corruption
    corruption_agent: str = "IQL"
    corruption_seed: int = 0  # 2023
    corruption_mode: str = ""  # random, adversarial
    corruption_tag: str = ""  # obs, act, rew
    corruption_next_obs: float = 0.0  # 0 or 1
    corruption_range: float = 1.0
    corruption_rate: float = 0.3
    use_original: int = 0  # 0 or 1
    same_index: int = 0
    froce_attack: int = 0
    # NSAOP 测试模式
    test_attack_mode: str = ""          # 设为 "nsaop" 启用
    nsaop_eps_coeff: float = 1.0
    reward_scale: float = 1.0    # 用于计算奖励攻击缩放

    def __post_init__(self):
        # train
        # if not self.eval_only:
        if True:
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
            if self.env == "halfcheetah-medium-v2":
                if self.corruption_obs:
                    self.threshold = 0.6
                if self.corruption_act:
                    self.threshold = 1.6
            elif self.env == "walker2d-medium-replay-v2":
                if self.corruption_obs:
                    self.threshold = 4.5
                if self.corruption_act:
                    self.threshold = 2.5
            if self.env.startswith("antmaze"):
                self.num_epochs = 1000
                self.buffer_size = 1000000
                self.eval_episodes = 100
            # sample ratio
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
            if self.corruption_mode == "random" and self.corruption_rew > 0.0:
                self.corruption_rew *= 30
            # auto set
            self.update_steps = int(self.num_epochs * self.num_updates_on_epoch)
            self.warmup_steps = int(0.1 * self.update_steps)
            self.decay_steps = int(0.1 * self.update_steps)
        # evaluation
        # if self.eval_only:
        #     assert self.checkpoint_dir is not None, "Please provide checkpoint_dir for evaluation."
        #     self.checkpoint_dir = os.path.join(self.logdir, self.group, self.env, self.checkpoint_dir)
        #     with open(os.path.join(self.checkpoint_dir, "params.json"), "r") as f:
        #         train_config = json.load(f)
        #     unoverwritten_keys = ["eval_id", "test_time", "group", "checkpoint_dir", "eval_only", "eval_attack",
        #                           "eval_attack_mode", "eval_attack_eps", "eval_corruption_rate"]
        #     for key, value in train_config.items():
        #         if key not in unoverwritten_keys:
        #             try:
        #                 value = eval(value)
        #             except:
        #                 pass
        #             self.__dict__[key] = value
        #             # print(f"Set {key} to {value}")
        self.eval_attack_mode = self.corruption_mode  # random, adversarial
        self.eval_attack_eps = 1
        self.eval_corruption_rate = 0.3
        if self.eval_attack_mode == "random" and self.corruption_tag == "rew":
            self.eval_attack_eps *= 30
        if self.env.startswith("antmaze") or self.env.startswith("kitchen") or \
                self.env.split("-")[0] in ["door", "pen", "hammer", "relocate"]:
            self.reward_scale = 1.0
        elif self.env.startswith("hopper") or self.env.startswith("halfcheetah") or self.env.startswith("walker"):
            self.reward_scale = 0.001


class GaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            max_action: float,
            hidden_dim: int = 256,
            n_hidden: int = 2,
    ):
        super().__init__()
        self.net = MLP([state_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> MultivariateNormal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(
            self.max_action * action, -self.max_action, self.max_action
        )
        return action.cpu().data.numpy().flatten()

    @torch.no_grad()
    def batch_act(self, state: np.ndarray, device: str = "cpu"):
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(
            self.max_action * action, -self.max_action, self.max_action
        )
        return action


class DeterministicPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            max_action: float,
            hidden_dim: int = 256,
            n_hidden: int = 2,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(
                self(state) * self.max_action, -self.max_action, self.max_action
            )
            .cpu()
            .data.numpy()
            .flatten()
        )


class BCLearning:
    def __init__(
            self,
            max_action: float,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            device: str = "cpu",
    ):
        self.max_action = max_action
        self.actor = actor
        self.actor_optimizer = actor_optimizer

        self.total_it = 0
        self.device = device

    def _update_policy(self, observations, actions, log_dict):
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        observations, actions = batch[0], batch[1]
        log_dict = {}
        self._update_policy(observations, actions, log_dict)
        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]

def eval_actor_nsaop(config, env, actor):
    device = config.device
    n_episodes = config.eval_episodes
    action_range = [
        float(env.action_space.low.min()) + 1e-6,
        float(env.action_space.high.max()) - 1e-6,
    ]

    actor.eval()
    episode_rewards_att = []
    episode_rewards_raw = []

    for _ in trange(n_episodes):

        # 每个 episode 都重新初始化 attacker
        nsaop_obs = None
        nsaop_act = None
        nsaop_rew = None

        if config.corruption_tag == "obs":
            nsaop_obs = NSAOPObsAttacker(
                state_dim=env.observation_space.shape[0],
                attack_state_std=config.attack_state_std,
                norm_state_mean=config.norm_state_mean,
                norm_state_std=config.norm_state_std,
                eps_coeff=config.nsaop_eps_coeff,
                device=device,
            )

        elif config.corruption_tag == "act":
            nsaop_act = NSAOPActAttacker(
                action_dim=env.action_space.shape[0],
                action_std=config.act_std,
                action_low=env.action_space.low,
                action_high=env.action_space.high,
                eps_coeff=config.nsaop_eps_coeff,
                device=device,
            )

        elif config.corruption_tag == "rew":
            nsaop_rew = NSAOPRewAttacker(
                rew_std=config.rew_std,
                reward_scale=1.0,
                eps_coeff=config.nsaop_eps_coeff,
                device=device,
            )

        state, done = env.reset(), False

        if nsaop_obs is not None:
            state = nsaop_obs.attack_obs(state)

        episode_reward_att = 0.0
        episode_reward_raw = 0.0

        while not done:
            action = actor.act(state, device)
            if nsaop_act:
                action = nsaop_act.attack_act(action)
            action = np.clip(action, *action_range)

            next_state, reward_raw, done, _ = env.step(action)  # 保存原始奖励

            # 观测攻击（next_state 篡改，模型看到篡改后的 state）
            if nsaop_obs:
                next_state = nsaop_obs.attack_obs(next_state)

            # 奖励攻击：模型使用攻击后的奖励，但原始奖励保留
            if nsaop_rew:
                reward_att = nsaop_rew.attack_rew(reward_raw)
            else:
                reward_att = reward_raw

            # 累加两种回报
            episode_reward_att += reward_att
            episode_reward_raw += reward_raw

            state = next_state

        episode_rewards_att.append(episode_reward_att)
        episode_rewards_raw.append(episode_reward_raw)

    actor.train()
    eval_returns_att = np.array(episode_rewards_att)
    eval_returns_raw = np.array(episode_rewards_raw)

    # 计算归一化分数
    normalized_score_att = env.get_normalized_score(eval_returns_att) * 100.0
    normalized_score_raw = env.get_normalized_score(eval_returns_raw) * 100.0

    eval_log = {
        "eval/reward_mean_att": np.mean(eval_returns_att),
        "eval/reward_std_att": np.std(eval_returns_att),
        "eval/normalized_score_mean_att": np.mean(normalized_score_att),
        "eval/normalized_score_std_att": np.std(normalized_score_att),
        "eval/reward_mean_raw": np.mean(eval_returns_raw),
        "eval/reward_std_raw": np.std(eval_returns_raw),
        "eval/normalized_score_mean_raw": np.mean(normalized_score_raw),
        "eval/normalized_score_std_raw": np.std(normalized_score_raw),
    }
    return eval_log
def train(config: TrainConfig, logger: Logger):
    # Set seeds
    best_score_att = -np.inf
    best_score_raw = 0.0
    func.set_seed(config.seed)

    if config.use_wandb:
        func.wandb_init(config)

    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if config.sample_ratio < 1.0:
        dataset_path = os.path.join(config.dataset_path, "datasets", f"{config.env}_ratio_{config.sample_ratio}.pt")
        dataset = torch.load(dataset_path)
    else:
        h5path = os.path.join(config.dataset_path, "original", f"{config.env}.hdf5")
        dataset = env.get_dataset(h5path=h5path)

    ##### corrupt
    if config.corruption_mode != "none":
        dataset, attack_indexs = attack_dataset(config, dataset, logger)

    dataset = d4rl.qlearning_dataset(env, dataset, terminate_on_end=True)
    dataset, state_mean, state_std = func.normalize_dataset(config, dataset)

    # BC 模型实际使用的 observation preprocessing
    config.norm_state_mean = state_mean
    config.norm_state_std = state_std

    # Drift-Attack 的 observation 物理尺度必须来自 clean dataset
    clean_state_std, _, _, _ = func.get_state_std(config)
    config.attack_state_std = clean_state_std

    # Action / Reward 暂时保持原 benchmark 逻辑
    config.act_std = np.std(dataset["actions"], axis=0) + 1e-6
    config.rew_std = np.std(dataset["rewards"]) + 1e-6
    env = func.wrap_env(env, state_mean=state_mean, state_std=state_std)
    env.seed(config.seed)

    buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    buffer.load_d4rl_dataset(dataset)

    if config.checkpoints_path is not None:
        logger.info(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, config.hidden_dim, config.n_hidden
        )
        if config.deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, config.hidden_dim, config.n_hidden
        )
    ).to(config.device)
    # logger.info(f"Actor Network: \n{str(actor)}")

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "max_action": max_action,
        "device": config.device,
    }

    # logger.info("---------------------------------------")
    # logger.info(f"Training BC, Env: {config.env}, Seed: {config.seed}")
    # logger.info("---------------------------------------")

    # Initialize actor
    trainer = BCLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    if config.eval_attack:
        state_std, act_std, rew_std, rew_min = func.get_state_std(config)
        eval_attacker = Evaluation_Attacker(
            config, config.env, config.corruption_agent, config.eval_attack_eps,
            state_dim, action_dim, state_std, act_std, rew_std, rew_min, config.eval_attack_mode,
            MODEL_PATH[config.corruption_agent],
        )
        print("eval_attack: True")
    else:
        eval_attacker = None
        print("eval_attack: False")

    if config.debug_eval:
        eval_log = func.eval(config, env, actor, eval_attacker)
        logger.record("epoch", 0)
        for k, v in eval_log.items():
            logger.record(k, v)
        logger.dump(0)

    # if config.use_wandb:
    #     wandb.log({"epoch": 0, **eval_log})
    #
    # best_score = -np.inf
    # best_score_50 = -np.inf
    best_score_raw_at_best = 0.0
    best_score_50_att = -np.inf
    best_score_50_raw = 0.0
    total_updates = 0.0
    for epoch in trange(1, config.num_epochs + 1, desc="Training"):
        time_start = time.time()
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            log_dict = trainer.train(batch)
            total_updates += 1
        time_end = time.time()
        epoch_time = time_end - time_start

        # Evaluate episode
        if epoch % config.eval_every == 0 and epoch > (config.num_epochs - config.eval_final):
            if config.test_attack_mode == "nsaop":
                eval_log = eval_actor_nsaop(config, env, actor)
            else:
                eval_log = func.eval(config, env, actor)
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
                    trainer.state_dict(),
                    os.path.join(logger.get_dir(), f"policy_{epoch}.pth"),
                )

            if config.use_wandb:
                update_log = {f"update/{k}": v for k, v in log_dict.items()}
                wandb.log({"epoch": epoch, **update_log})
                wandb.log({"epoch": epoch, **eval_log})

            score_att = eval_log["eval/normalized_score_mean_att"]
            score_raw = eval_log["eval/normalized_score_mean_raw"]
            with open(os.path.join(logger.get_dir(), "eval_scores.txt"), "a") as f:
                f.write(f"att:{score_att:.4f}_raw:{score_raw:.4f}_epoch{epoch}\n")
            if score_att > best_score_att:
                best_score_att = score_att
                best_score_raw_at_best = score_raw
                best_epoch = epoch
                with open(os.path.join(logger.get_dir(), "best_score.txt"), "w") as f:
                    f.write(f"{best_score_att:.4f}_{best_score_raw_at_best:.4f}_{best_epoch}")
                if config.save_model:
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(logger.get_dir(), f"best_policy.pth"),
                    )
            if epoch > config.num_epochs - 50:
                if score_att > best_score_50_att:
                    best_score_50_att = score_att
                    best_score_50_raw = score_raw
                    with open(os.path.join(logger.get_dir(), "best_score_50.txt"), "w") as f:
                        f.write(f"{best_score_50_att:.4f}_{best_score_50_raw:.4f}_{epoch}")
                    if config.save_model:
                        torch.save(
                            trainer.state_dict(),
                            os.path.join(logger.get_dir(), f"best_policy_50.pth"),
                        )
            if epoch == config.num_epochs:
                with open(os.path.join(logger.get_dir(), "final_score.txt"), "w") as f:
                    f.write(f"att:{score_att:.4f}_raw:{score_raw:.4f}_{epoch}")
                if config.save_model:
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(logger.get_dir(), f"final_policy.pth"),
                    )

    if config.use_wandb:
        wandb.finish()


def test(config: TrainConfig, logger: Logger):
    # Set seeds
    func.set_seed(config.seed)
    best_score_att = -np.inf
    best_score_raw_at_best = 0.0
    best_epoch = 0
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if config.sample_ratio < 1.0:
        dataset_path = os.path.join(config.dataset_path, "datasets", f"{config.env}_ratio_{config.sample_ratio}.pt")
        dataset = torch.load(dataset_path)
    else:
        h5path = os.path.join(config.dataset_path, "original", f"{config.env}.hdf5")
        dataset = env.get_dataset(h5path=h5path)

    ##### corrupt
    if config.corruption_mode != "none":
        dataset, attack_indexs = attack_dataset(config, dataset, logger)

    dataset = d4rl.qlearning_dataset(env, dataset, terminate_on_end=True)
    dataset, state_mean, state_std = func.normalize_dataset(config, dataset)

    # BC 模型实际使用的 observation preprocessing
    config.norm_state_mean = state_mean
    config.norm_state_std = state_std

    # Drift-Attack 的 observation 物理尺度必须来自 clean dataset
    clean_state_std, _, _, _ = func.get_state_std(config)
    config.attack_state_std = clean_state_std

    # Action / Reward 暂时保持原 benchmark 逻辑
    config.act_std = np.std(dataset["actions"], axis=0) + 1e-6
    config.rew_std = np.std(dataset["rewards"]) + 1e-6
    env = func.wrap_env(env, state_mean=state_mean, state_std=state_std)
    env.seed(config.seed)

    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, config.hidden_dim, config.n_hidden
        )
        if config.deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, config.hidden_dim, config.n_hidden
        )
    ).to(config.device)
    all_files = os.listdir(config.checkpoint_dir)
    model_epoches = [
        f for f in all_files
        if f.startswith("policy") and f.endswith(".pth")
    ]
    model_epoches.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))

    best_epoch = 0
    open(os.path.join(logger.get_dir(), "eval_scores.txt"), "w").close()
    for i, model_epoch in enumerate(model_epoches):
        epoch = int(model_epoch.split(".")[0].split("_")[1])
        print(f"eval epoch: {epoch}")
        actor.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, model_epoch))["actor"])
        actor.eval()

        if config.test_attack_mode == "nsaop":
            eval_log = eval_actor_nsaop(config, env, actor)
        else:
            if config.eval_attack:
                state_std, act_std, rew_std, rew_min = func.get_state_std(config)
                eval_attacker = Evaluation_Attacker(
                    config, config.env, config.corruption_agent, config.eval_attack_eps,
                    state_dim, action_dim, state_std, act_std, rew_std, rew_min, config.eval_attack_mode,
                    MODEL_PATH[config.corruption_agent],
                )
                print("eval_attack: True")
            else:
                eval_attacker = None
                print("eval_attack: False")
            eval_log = func.eval(config, env, actor, eval_attacker)
        for k, v in eval_log.items():
            logger.record(k, v)
        logger.dump(0)

        score_att = eval_log["eval/normalized_score_mean_att"]
        score_raw = eval_log["eval/normalized_score_mean_raw"]
        with open(os.path.join(logger.get_dir(), "eval_scores.txt"), "a") as f:
            f.write(f"att:{score_att:.4f}_raw:{score_raw:.4f}_epoch{epoch}\n")
        if score_att > best_score_att:
            best_score_att = score_att
            best_score_raw_at_best = score_raw
            best_epoch = epoch
        eval_atta_tag = "attack" if config.eval_attack else "clean"
        # train_time = config.checkpoint_dir.split("_")[-2]
        log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(logger.get_dir()))),
                            f"test_{config.group}_{config.env}_{config.corruption_mode}_{eval_atta_tag}_{model_epoch}_{config.test_time}.txt")
        title = f"{config.group}_{config.env}_{config.corruption_mode}_{config.corruption_tag}_{eval_atta_tag}_{config.seed}"
        with open(log_path, "a") as f:
            f.write(f"{title}: att={score_att:.4f} raw={score_raw:.4f}\n")
    if best_score_att > -np.inf:
        with open(os.path.join(logger.get_dir(), "best_score.txt"), "w") as f:
            f.write(f"{best_score_att:.4f}_{best_score_raw_at_best:.4f}_{best_epoch}")

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
