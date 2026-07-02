import os, sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from tqdm.auto import trange  # noqa

from networks import MLPBlock, ResidualBlock, TransformerBlock, DiagGaussianActor
from attack import attack_dataset, Evaluation_Attacker
from logger import Logger


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
        arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_d4rl_trajectories(
        config, env_name: str, gamma: float = 1.0, logger: Logger = None
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset_path = os.path.join(config.dataset_path, "datasets", f"{env_name}_ratio_{config.sample_ratio}.pt")
    dataset = torch.load(dataset_path)
    attack_mask = np.ones_like(dataset["rewards"]) * -1
    if config.corruption_mode != "none":  # New
        dataset, attack_indexs = attack_dataset(config, dataset, logger)
        attack_mask[attack_indexs] = 1
    dataset["attack_mask"] = attack_mask

    state_mean, state_std = 0.0, 1.0
    if config.normalize:
        state_mean = dataset["observations"].mean(0, keepdims=True)
        state_std = dataset["observations"].std(0, keepdims=True) + 1e-6

    traj, traj_len = [], []
    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])
        data_["attack_mask"].append(dataset["attack_mask"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": state_mean,
        "obs_std": state_std,
        "traj_lens": np.array(traj_len),
        "rew_std": dataset["rewards"].std() + 1e-6,
        "act_std": dataset["actions"].std(0, keepdims=True) + 1e-6,
    }
    return traj, info


class SequenceDataset:
    def __init__(self, config, logger: Logger = None):
        self.dataset, info = load_d4rl_trajectories(config, config.env, gamma=1.0, logger=logger)
        self.seq_len = config.seq_len
        self.max_ep_len = config.episode_len
        self.reward_scale = config.reward_scale
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.action_range = config.action_range

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        self.rew_std = info["rew_std"]
        self.act_std = info["act_std"]
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()
        self.float_dtype = np.float32

    def correct(self, traj_indexs, time_steps, correct_data, correct_type):
        for i, (tarj_i, step_j) in enumerate(zip(traj_indexs, time_steps)):
            if step_j < self.dataset[tarj_i][correct_type].shape[0]:
                self.dataset[tarj_i][correct_type][step_j] = correct_data[i]

    def __prepare_sample(self, traj_idx, start_idx, recalculate_return):
        traj = self.dataset[traj_idx]
        states = traj["observations"][start_idx: start_idx + self.seq_len]
        actions = traj["actions"][start_idx: start_idx + self.seq_len]
        if recalculate_return:
            returns = discounted_cumsum(traj["rewards"][start_idx:], gamma=1.0).reshape(-1, 1)[: self.seq_len]
        else:
            returns = traj["returns"][start_idx: start_idx + self.seq_len].reshape(-1, 1)
        rewards = traj["rewards"][start_idx: start_idx + self.seq_len].reshape(-1, 1)
        attack_mask = traj["attack_mask"][start_idx: start_idx + self.seq_len].reshape(-1, 1)
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        rewards = rewards * self.reward_scale
        # pad up to seq_len if needed
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)
            rewards = pad_along_axis(rewards, pad_to=self.seq_len)
            attack_mask = pad_along_axis(attack_mask, pad_to=self.seq_len)

        return states.astype(self.float_dtype), actions.astype(self.float_dtype), returns.astype(self.float_dtype), \
            rewards.astype(self.float_dtype), time_steps, mask, attack_mask, traj_idx

    def get_batch(self, batch_size: int, recalculate_return: bool = False) -> Tuple[torch.Tensor, ...]:
        traj_ids = np.random.choice(np.arange(len(self.dataset)), size=batch_size, p=self.sample_prob, replace=True)

        states, actions, returns, rewards, time_steps, masks, attack_mask, traj_index = [], [], [], [], [], [], [], []
        for traj_id in traj_ids:
            start_idx = np.random.randint(0, self.dataset[traj_id]["rewards"].shape[0])
            state, action, ret, reward, time_step, mask, att_mask, traj_i = self.__prepare_sample(traj_id, start_idx,
                                                                                                  recalculate_return)
            states.append(state)
            actions.append(action)
            returns.append(ret)
            rewards.append(reward)
            time_steps.append(time_step)
            masks.append(mask)
            attack_mask.append(att_mask)
            traj_index.append(traj_i)
        return [
            torch.tensor(np.array(states)),
            torch.tensor(np.array(actions)),
            torch.tensor(np.array(returns)),
            torch.tensor(np.array(rewards)),
            torch.tensor(np.array(time_steps)),
            torch.tensor(np.array(masks)),
            torch.tensor(np.array(attack_mask)),
            torch.tensor(np.array(traj_index))
        ]

    def save(self, path: str):
        torch.save(self.dataset, path)


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
        model: nn.Module,
        env: gym.Env,
        target_return: float,
        float_dtype: torch.dtype = torch.float32,
        eval_attacker: Evaluation_Attacker = None,
        eval_corruption_rate: float = 0.0,
        eval_attack_tag: str = "obs",
        device: str = "cpu",
        use_stochastic: bool = False,
        config=None
) -> Tuple[float, float]:
    action_range = [
        float(env.action_space.low.min()) + 1e-6,
        float(env.action_space.high.max()) - 1e-6,
    ]
    states = torch.zeros(
        1, model.episode_len + 1, model.state_dim, dtype=float_dtype, device=device
    )
    actions = torch.zeros(
        1, model.episode_len, model.action_dim, dtype=float_dtype, device=device
    )
    returns = torch.zeros(1, model.episode_len + 1, 1, dtype=float_dtype, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    nsaop_obs_attacker = None
    nsaop_act_attacker = None
    nsaop_rew_attacker = None

    global_eps_coeff = getattr(config, 'nsaop_eps_coeff', 1.0) if config else 1.0

    if hasattr(config, 'test_attack_mode') and config.test_attack_mode == "nsaop":
        if eval_attack_tag == "obs":
            env_state_std = getattr(config, 'state_std', None)
            nsaop_obs_attacker = NSAOPObsAttacker(
                state_dim=model.state_dim,
                state_std=env_state_std,
                eps_coeff=global_eps_coeff,
                device=device
            )
        elif eval_attack_tag == "act":
            # 动态获取环境边界与动作标准差
            action_low = env.action_space.low
            action_high = env.action_space.high
            env_act_std = getattr(config, 'act_std', None)
            nsaop_act_attacker = NSAOPActAttacker(
                action_dim=model.action_dim,
                action_std=env_act_std,
                action_low=action_low,
                action_high=action_high,
                eps_coeff=global_eps_coeff,
                device=device
            )
        elif eval_attack_tag == "rew":
            env_rew_std = getattr(config, 'rew_std', 1.0) if config else 1.0
            nsaop_rew_attacker = NSAOPRewAttacker(
                rew_std=env_rew_std,
                reward_scale=config.reward_scale if config else 1.0,
                eps_coeff=global_eps_coeff,
                device=device
            )

    obs = env.reset()

    if nsaop_obs_attacker is not None:
        obs = nsaop_obs_attacker.attack_obs(obs)
    elif eval_attacker is not None and eval_attack_tag == "obs":
        attack_flag = np.random.rand()
        if attack_flag < eval_corruption_rate:
            obs = eval_attacker.attack_obs(obs)
    states[:, 0] = torch.as_tensor(obs, device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    episode_return_att, episode_return_raw = 0.0, 0.0
    episode_len = 0

    for step in range(model.episode_len):
        predicted = model(
            states[:, : step + 1][:, -model.seq_len:],
            actions[:, : step + 1][:, -model.seq_len:],
            returns[:, : step + 1][:, -model.seq_len:],
            time_steps[:, : step + 1][:, -model.seq_len:],
        )
        predicted_actions = predicted[0]
        if use_stochastic:
            predicted_actions = predicted_actions.mean
        predicted_action = predicted_actions[0, -1].cpu().numpy()

        # [彻底移除平滑逻辑] 保持 RDT 纯净测试

        if nsaop_act_attacker is not None:
            predicted_action = nsaop_act_attacker.attack_act(predicted_action)
        elif eval_attacker is not None and eval_attack_tag == "act":
            attack_flag = np.random.rand()
            if attack_flag < eval_corruption_rate:
                predicted_action = eval_attacker.attack_act(predicted_action)

        predicted_action = np.clip(predicted_action, *action_range)
        next_state, reward_raw, done, info = env.step(predicted_action)
        episode_len += 1
        if nsaop_obs_attacker is not None:
            next_state = nsaop_obs_attacker.attack_obs(next_state)
        elif eval_attacker is not None and eval_attack_tag == "obs":
            attack_flag = np.random.rand()
            if attack_flag < eval_corruption_rate:
                next_state = eval_attacker.attack_obs(next_state)

        if nsaop_rew_attacker is not None:
            reward_att = nsaop_rew_attacker.attack_rew(reward_raw)
        elif eval_attacker is not None and eval_attack_tag == "rew":
            attack_flag = np.random.rand()
            if attack_flag < eval_corruption_rate:
                reward_att = eval_attacker.attack_rew(reward_raw)
            else:
                reward_att = reward_raw
        else:
            reward_att = reward_raw

        episode_return_att += reward_att
        episode_return_raw += reward_raw

        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward_att)
        if done:
            break
    return episode_return_att, episode_return_raw, episode_len


def eval_fn(config, env, model, eval_attacker=None):
    eval_log = {}
    eval_attack_tag = config.corruption_tag
    use_stochastic = config.use_stochastic if hasattr(config, 'use_stochastic') else False
    for target_return in config.target_returns:
        eval_returns_att = []
        eval_returns_raw = []
        for _ in trange(config.n_episodes, desc="Evaluation", leave=False):
            eval_return_att, eval_return_raw, eval_len = eval_rollout(
                model=model,
                env=env,
                target_return=target_return * config.reward_scale,
                eval_attacker=eval_attacker,
                eval_corruption_rate=config.eval_corruption_rate,
                eval_attack_tag=eval_attack_tag,
                device=config.device,
                use_stochastic=use_stochastic,
                config=config,
            )
            eval_returns_att.append(eval_return_att / config.reward_scale)
            eval_returns_raw.append(eval_return_raw / config.reward_scale)

        eval_returns_att = np.array(eval_returns_att)
        eval_returns_raw = np.array(eval_returns_raw)
        normalized_score_att = env.get_normalized_score(eval_returns_att) * 100
        normalized_score_raw = env.get_normalized_score(eval_returns_raw) * 100
        eval_log.update({
            f"eval/{target_return}_reward_mean_att": np.mean(eval_returns_att),
            f"eval/{target_return}_reward_std_att": np.std(eval_returns_att),
            f"eval/{target_return}_normalized_score_mean_att": np.mean(normalized_score_att),
            f"eval/{target_return}_normalized_score_std_att": np.std(normalized_score_att),
            f"eval/{target_return}_reward_mean_raw": np.mean(eval_returns_raw),
            f"eval/{target_return}_reward_std_raw": np.std(eval_returns_raw),
            f"eval/{target_return}_normalized_score_mean_raw": np.mean(normalized_score_raw),
            f"eval/{target_return}_normalized_score_std_raw": np.std(normalized_score_raw),
        })
    return eval_log


class DecisionTransformer(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            seq_len: int = 10,
            episode_len: int = 1000,
            embedding_dim: int = 128,
            num_layers: int = 4,
            num_heads: int = 8,
            attention_dropout: float = 0.0,
            residual_dropout: float = 0.0,
            embedding_dropout: float = 0.0,
            predict_dropout: float = 0.1,
            mlp_embedding: bool = False,
            mlp_head: bool = False,
            mlp_reward: bool = False,
            predict_reward: bool = False,
            embed_order: str = "rsa",
            use_stochastic: bool = False,
            init_temperature: float = 0.1,
            corruption_tag: str = "none",
    ):
        super().__init__()
        self.corruption_tag = corruption_tag
        if embedding_dropout is not None:
            self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)

        self.state_emb = nn.Linear(state_dim, embedding_dim) if not mlp_embedding else ResidualBlock(state_dim,
                                                                                                     embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim) if not mlp_embedding else ResidualBlock(action_dim,
                                                                                                       embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim) if not mlp_embedding else ResidualBlock(1, embedding_dim)

        effective_seq_len = 3 * seq_len

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=effective_seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.predict_dropout = nn.Dropout(predict_dropout)
        self.use_stochastic = use_stochastic
        if self.use_stochastic:
            self.action_head = DiagGaussianActor(embedding_dim, action_dim)
            self.log_temperature = torch.tensor(np.log(init_temperature), requires_grad=True)
            self.target_entropy = -action_dim
        else:
            num_layer = 2 if mlp_head else 1
            self.action_head = MLPBlock(embedding_dim, action_dim, num_layer, True)
        if predict_reward:
            num_layer = 2 if mlp_reward else 1
            self.reward_head = MLPBlock(embedding_dim, 1, num_layer)

        self.apply(self._init_weights)
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.embed_order = embed_order
        self.predict_reward = predict_reward

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            returns_to_go: torch.Tensor,
            time_steps: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len = states.shape[0], states.shape[1]
        time_emb = self.timestep_emb(time_steps)

        state_emb = self.state_emb(states)
        act_emb = self.action_emb(actions)
        returns_emb = self.return_emb(returns_to_go)

        if self.embed_order == "rsa":
            sequence = torch.stack([returns_emb, state_emb, act_emb], dim=1)
        elif self.embed_order == "sar":
            sequence = torch.stack([state_emb, act_emb, returns_emb], dim=1)
        else:
            raise ValueError(f"Invalid embedding order {self.embed_order}.")
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.embedding_dim)
        sequence = sequence + time_emb.repeat_interleave(3, dim=1)

        if padding_mask is not None:
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )

        out = self.emb_norm(sequence)
        if hasattr(self, "emb_drop"):
            out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        out = self.predict_dropout(out)

        if self.embed_order == "rsa":
            out_r_emb, out_s_emb, out_a_emb = out[:, 0::3], out[:, 1::3], out[:, 2::3]
        elif self.embed_order == "sar":
            out_s_emb, out_a_emb, out_r_emb = out[:, 0::3], out[:, 1::3], out[:, 2::3]
        else:
            raise ValueError(f"Invalid embedding order {self.embed_order}.")

        action_out = self.action_head(out_s_emb)
        if self.predict_reward:
            reward_out = self.reward_head(out_a_emb)
        else:
            reward_out = None
        return action_out, reward_out

class NSAOPObsAttacker:
    """
    非平稳观测攻击器：基于状态标准差的无量纲统一缩放方案
    """

    def __init__(
            self,
            state_dim: int,
            state_std: Optional[np.ndarray] = None,
            burst_prob: float = 0.1,
            recover_prob: float = 0.3,
            momentum: float = 0.85,
            eps_coeff: float = 1.0,
            device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.burst_prob = burst_prob
        self.recover_prob = recover_prob
        self.momentum = momentum
        self.eps_coeff = eps_coeff
        self.device = device

        self.m_state = 0
        self.accumulated_drift = torch.zeros(state_dim, device=device)

        if state_std is None:
            state_std = np.ones(state_dim)
        self.state_std = torch.tensor(state_std, dtype=torch.float32, device=device).view(-1)

    def step(self):
        if self.m_state == 0:
            if np.random.rand() < self.burst_prob: self.m_state = 1
        else:
            if np.random.rand() < self.recover_prob: self.m_state = 0
        return self.m_state

    def attack_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.step() == 0:
            self.accumulated_drift.zero_()
            return obs

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        direction = torch.sign(obs_tensor + 1e-8)

        # 无量纲缩放：全局系数 * 状态标准差 * 方向
        scaled_eps = self.eps_coeff * self.state_std * direction

        self.accumulated_drift = (
                self.momentum * self.accumulated_drift
                + (1 - self.momentum) * scaled_eps
        )

        # 物理饱和截断限制：防止长序列发散，限制在 3 倍标准差以内
        self.accumulated_drift = torch.clamp(self.accumulated_drift, -3.0 * self.state_std, 3.0 * self.state_std)

        return (obs_tensor + self.accumulated_drift).cpu().numpy()


class NSAOPActAttacker:
    """
    非平稳动作攻击器：基于动作自然标准差的无量纲统一缩放方案 + 动态环境边界
    """

    def __init__(
            self,
            action_dim: int,
            action_std: Optional[np.ndarray] = None,
            action_low: Optional[np.ndarray] = None,
            action_high: Optional[np.ndarray] = None,
            burst_prob: float = 0.1,
            recover_prob: float = 0.3,
            momentum: float = 0.85,
            eps_coeff: float = 1.0,
            device: str = "cpu"
    ):
        self.action_dim = action_dim
        self.burst_prob = burst_prob
        self.recover_prob = recover_prob
        self.momentum = momentum
        self.eps_coeff = eps_coeff
        self.device = device
        self.m_state = 0
        self.accumulated_drift = torch.zeros(action_dim, device=device)

        # 1. 统一信噪比基准：若未提供，预估为 0.5
        if action_std is None:
            action_std = np.ones(action_dim) * 0.5
        self.action_std = torch.tensor(action_std, dtype=torch.float32, device=device).view(-1)

        # 2. 真实环境边界
        if action_low is None:
            action_low = -np.ones(action_dim)
        if action_high is None:
            action_high = np.ones(action_dim)
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)

    def step(self):
        if self.m_state == 0:
            if np.random.rand() < self.burst_prob: self.m_state = 1
        else:
            if np.random.rand() < self.recover_prob: self.m_state = 0
        return self.m_state

    def attack_act(self, action: np.ndarray) -> np.ndarray:
        if self.step() == 0:
            self.accumulated_drift.zero_()
            return action

        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        direction = torch.sign(action_tensor + 1e-8)

        # 无量纲缩放：全局系数 * 动作标准差 * 方向
        scaled_eps = self.eps_coeff * self.action_std * direction

        self.accumulated_drift = (
                self.momentum * self.accumulated_drift
                + (1 - self.momentum) * scaled_eps
        )

        # 物理饱和截断限制：防止漂移无限发散 (3 倍标准差)
        self.accumulated_drift = torch.clamp(self.accumulated_drift, -3.0 * self.action_std, 3.0 * self.action_std)

        perturbed = action_tensor + self.accumulated_drift

        # 输出给环境前执行动态硬边界截断
        return np.clip(perturbed.cpu().numpy(),
                       self.action_low.cpu().numpy(),
                       self.action_high.cpu().numpy())


class NSAOPRewAttacker:
    """
    非平稳奖励攻击器：基于奖励标准差的无量纲统一缩放方案
    """

    def __init__(
            self,
            rew_std: float = 1.0,
            reward_scale: float = 1.0,
            burst_prob: float = 0.1,
            recover_prob: float = 0.3,
            momentum: float = 0.85,
            eps_coeff: float = 1.0,
            device: str = "cpu"
    ):
        self.burst_prob = burst_prob
        self.recover_prob = recover_prob
        self.momentum = momentum
        self.eps_coeff = eps_coeff
        self.device = device
        self.m_state = 0
        self.accumulated_drift = 0.0

        # 对齐统一量纲缩放
        self.base_eps = self.eps_coeff * rew_std * reward_scale
        self.rew_std_scaled = rew_std * reward_scale

    def step(self):
        if self.m_state == 0:
            if np.random.rand() < self.burst_prob: self.m_state = 1
        else:
            if np.random.rand() < self.recover_prob: self.m_state = 0
        return self.m_state

    def attack_rew(self, reward: float) -> float:
        if self.step() == 0:
            self.accumulated_drift = 0.0
            return reward

        direction = -np.sign(reward) if reward != 0 else -1.0
        scaled_eps = self.base_eps * direction

        self.accumulated_drift = (
                self.momentum * self.accumulated_drift
                + (1 - self.momentum) * scaled_eps
        )

        # 物理饱和截断限制：防止奖励信号完全崩溃
        self.accumulated_drift = np.clip(self.accumulated_drift, -3.0 * self.rew_std_scaled, 3.0 * self.rew_std_scaled)

        return reward + self.accumulated_drift