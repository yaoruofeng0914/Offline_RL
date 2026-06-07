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

    # dataset, state_mean, state_std = func.normalize_dataset(config, dataset)
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
        "obs_mean": state_mean,  # dataset["observations"].mean(0, keepdims=True),
        "obs_std": state_std,  # dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    # return ori_traj, traj, info
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
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()
        self.float_dtype = np.float32

    def correct(self, traj_indexs, time_steps, correct_data, correct_type):
        for i, (tarj_i, step_j) in enumerate(zip(traj_indexs, time_steps)):
            if step_j < self.dataset[tarj_i][correct_type].shape[0]:
                self.dataset[tarj_i][correct_type][step_j] = correct_data[i]

    def __prepare_sample(self, traj_idx, start_idx, recalculate_return):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
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
class BayesianEmbedding(nn.Module):
    """输出高斯分布参数 (mean, log_var) 的贝叶斯嵌入层"""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.mean = nn.Linear(input_dim, embed_dim)
        self.logvar = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        mu = self.mean(x)
        logvar = self.logvar(x)
        logvar = torch.clamp(logvar, -10, 5)   # 防止数值爆炸
        return mu, logvar
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
    if hasattr(config, 'test_attack_mode') and config.test_attack_mode == "nsaop":
        if eval_attack_tag == "obs":
            nsaop_obs_attacker = NSAOPObsAttacker(state_dim=model.state_dim, device=device)
        elif eval_attack_tag == "act":
            nsaop_act_attacker = NSAOPActAttacker(action_dim=model.action_dim, device=device)
        elif eval_attack_tag == "rew":
            nsaop_rew_attacker = NSAOPRewAttacker(device=device)
    obs = env.reset()

    # NSAOP: 替换初始观测攻击
    if nsaop_obs_attacker is not None:
        obs = nsaop_obs_attacker.attack_obs(obs)
    elif eval_attacker is not None and eval_attack_tag == "obs":
        attack_flag = np.random.rand()
        if attack_flag < eval_corruption_rate:
            obs = eval_attacker.attack_obs(obs)
    states[:, 0] = torch.as_tensor(obs, device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)
        predicted = model(  # fix this noqa!!!
            states[:, : step + 1][:, -model.seq_len:],
            actions[:, : step + 1][:, -model.seq_len:],
            returns[:, : step + 1][:, -model.seq_len:],
            time_steps[:, : step + 1][:, -model.seq_len:],
        )
        predicted_actions = predicted[0]
        if use_stochastic:
            predicted_actions = predicted_actions.mean
        predicted_action = predicted_actions[0, -1].cpu().numpy()

        # NSAOP-act: 替换动作攻击
        if nsaop_act_attacker is not None:
            predicted_action = nsaop_act_attacker.attack_act(predicted_action)
        elif eval_attacker is not None and eval_attack_tag == "act":
            attack_flag = np.random.rand()
            if attack_flag < eval_corruption_rate:
                predicted_action = eval_attacker.attack_act(predicted_action)
        predicted_action = np.clip(predicted_action, *action_range)
        next_state, reward, done, info = env.step(predicted_action)
        episode_return += reward
        episode_len += 1

        if nsaop_obs_attacker is not None:
            next_state = nsaop_obs_attacker.attack_obs(next_state)
        elif eval_attacker is not None and eval_attack_tag == "obs":
            attack_flag = np.random.rand()
            if attack_flag < eval_corruption_rate:
                next_state = eval_attacker.attack_obs(next_state)
        # NSAOP-rew: 替换奖励攻击
        if nsaop_rew_attacker is not None:
            reward = nsaop_rew_attacker.attack_rew(reward)
        elif eval_attacker is not None and eval_attack_tag == "rew":
            attack_flag = np.random.rand()
            if attack_flag < eval_corruption_rate:
                reward = eval_attacker.attack_rew(reward)
        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        if done:
            break
    return episode_return, episode_len


def eval_fn(config, env, model, eval_attacker=None):
    eval_log = {}
    eval_attack_tag = config.corruption_tag
    use_stochastic = config.use_stochastic if hasattr(config, 'use_stochastic') else False
    for target_return in config.target_returns:
        eval_returns = []
        for _ in trange(config.n_episodes, desc="Evaluation", leave=False):
            eval_return, eval_len = eval_rollout(
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
            # unscale for logging & correct normalized score computation
            eval_returns.append(eval_return / config.reward_scale)

        eval_returns = np.array(eval_returns)
        normalized_score = env.get_normalized_score(eval_returns) * 100
        eval_log.update({
            f"eval/{target_return}_reward_mean": np.mean(eval_returns),
            f"eval/{target_return}_reward_std": np.std(eval_returns),
            f"eval/{target_return}_normalized_score_mean": np.mean(normalized_score),
            f"eval/{target_return}_normalized_score_std": np.std(normalized_score),
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
        self.skip_gating = False
        self.corruption_tag = corruption_tag
        if embedding_dropout is not None:
            self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)

        if mlp_embedding:
            self.state_emb = ResidualBlock(state_dim, embedding_dim)
            self.action_emb = ResidualBlock(action_dim, embedding_dim)
            self.return_emb = ResidualBlock(1, embedding_dim)
        else:
            self.state_emb = BayesianEmbedding(state_dim, embedding_dim)
            self.action_emb = BayesianEmbedding(action_dim, embedding_dim)
            self.return_emb = BayesianEmbedding(1, embedding_dim)
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

        self.log_lambda = nn.Parameter(torch.tensor(0.0))   # 初始化为 0，即 λ=1
        self.log_lambda.requires_grad = False
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


    def toggle_temperature_grad(self, mode: bool):
        """
        mode: True (解冻，允许更新), False (冻结，禁止更新)
        """
        # 遍历所有的 Transformer Block
        for block in self.blocks:
        # 检查是否有 attention 属性且 attention 有 log_temperature 参数
            if hasattr(block, 'attention') and hasattr(block.attention, 'log_temperature'):
            # 设置 requires_grad
                block.attention.log_temperature.requires_grad = mode

            # 如果是 False，最好手动清空一下现有的梯度，防止残留
                if not mode and block.attention.log_temperature.grad is not None:
                    block.attention.log_temperature.grad = None


    def w_stochastic_action_loss(self, config, predicted_actions, actions, mask):
        with torch.no_grad():
            diff = torch.square(predicted_actions.mean.detach() - actions.detach()).mean(-1, keepdim=True)
            weight = torch.exp(-config.wmse_coef[0] * diff)

        mask = mask.unsqueeze(-1)
        log_likelihood = predicted_actions.log_prob(actions.detach())
        log_likelihood = (log_likelihood * weight * mask).mean()
        entropy = (predicted_actions.entropy() * mask).mean()
        if config.use_entropy:
            entropy_reg = self.log_temperature.exp().detach()
        else:
            entropy_reg = 0
        act_loss = -(log_likelihood + entropy_reg * entropy)
        return act_loss, entropy


    def forward(
            self,
            states: torch.Tensor,  # [batch_size, seq_len, state_dim]
            actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
            returns_to_go: torch.Tensor,  # [batch_size, seq_len]
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ):
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        # act_emb = self.action_emb(actions)
        # returns_emb = self.return_emb(returns_to_go)

        state_mu, state_logvar = self.state_emb(states)  # [B, T, E]
        act_mu, act_logvar = self.action_emb(actions)
        ret_mu, ret_logvar = self.return_emb(returns_to_go)

        # 计算每个 token 的标量方差（取嵌入维度的均值）
        state_var = state_logvar.exp().mean(dim=-1)  # [B, T]
        act_var = act_logvar.exp().mean(dim=-1)
        ret_var = ret_logvar.exp().mean(dim=-1)

        # 按 embed_order 构造方差序列
        if self.embed_order == "rsa":
            var_seq = torch.stack([ret_var, state_var, act_var], dim=1)  # [B, 3, T]
        elif self.embed_order == "sar":
            var_seq = torch.stack([state_var, act_var, ret_var], dim=1)
        else:
            raise ValueError(f"Invalid embed_order {self.embed_order}")
        var_seq = var_seq.permute(0, 2, 1).reshape(batch_size, 3 * seq_len)  # [B, 3T]

        # 构造均值嵌入序列
        if self.embed_order == "rsa":
            sequence = torch.stack([ret_mu, state_mu, act_mu], dim=1)
        elif self.embed_order == "sar":
            sequence = torch.stack([state_mu, act_mu, ret_mu], dim=1)
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.embedding_dim)
        sequence = sequence + time_emb.repeat_interleave(3, dim=1)

        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )

    # LayerNorm and Dropout (!!!) as in original implementation,
    # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        if hasattr(self, "emb_drop"):
            out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask,
                        var_seq=var_seq if not self.skip_gating else None,
                        log_lambda=self.log_lambda if not self.skip_gating else None)

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
        # 在 for block in self.blocks: 循环之后添加
        # 保存最后一个 batch 的诊断信息（仅用于调试）
        if hasattr(self, 'log_lambda') and var_seq is not None:
            self._debug_lambda = torch.exp(self.log_lambda).item()
            self._debug_var_mean = var_seq.mean().item()
            self._debug_var_max = var_seq.max().item()
            self._debug_var_min = var_seq.min().item()
        return action_out, reward_out, (state_mu, state_logvar, act_mu, act_logvar, ret_mu, ret_logvar)

class NSAOPObsAttacker:
    """
    非平稳对抗性观测摄动 (NSAOP) 攻击器
    模拟传感器间歇性故障和渐进式机械磨损

    物理对应:
    - 马尔可夫突发: 传感器间歇故障、电磁干扰
    - 动量累积漂移: 机械磨损、渐进式攻击（定向持久漂移）
    """

    def __init__(
            self,
            state_dim: int,
            burst_prob: float = 0.3,      # 更高故障突发概率
            recover_prob: float = 0.1,     # 更难恢复，故障持续时间更长
            momentum: float = 0.98,         # 极强惯性，漂移几乎不衰减
            eps: float = 5.0,               # 大幅扰动强度
            device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.burst_prob = burst_prob
        self.recover_prob = recover_prob
        self.momentum = momentum
        self.eps = eps
        self.device = device

        # 马尔可夫链状态: 0 = normal, 1 = burst
        self.m_state = 0
        # 累积漂移动量
        self.accumulated_drift = torch.zeros(state_dim, device=device)

        # 固定基础漂移方向（模拟传感器磨损的持续偏向）
        self.base_direction = torch.randn(state_dim, device=device)
        self.base_direction = self.base_direction / (torch.norm(self.base_direction) + 1e-8)

    def step(self):
        """更新马尔可夫链状态"""
        if self.m_state == 0:
            if np.random.rand() < self.burst_prob:
                self.m_state = 1
        else:
            if np.random.rand() < self.recover_prob:
                self.m_state = 0
        return self.m_state

    def generate_drift(self) -> torch.Tensor:
        """生成动量累积的定向漂移（方向大致固定，带微小随机扰动）"""
        # 在固定方向基础上加入微小噪声，模拟磨损方向缓慢变化
        perturbation = 0.02 * torch.randn(self.state_dim, device=self.device)
        current_dir = self.base_direction + perturbation
        current_dir = current_dir / (torch.norm(current_dir) + 1e-8)

        # 动量累积: Δ_t = ρ·Δ_{t-1} + (1-ρ)·ε·direction
        self.accumulated_drift = (
                self.momentum * self.accumulated_drift
                + (1 - self.momentum) * self.eps * current_dir
        )
        return self.accumulated_drift.clone()

    def attack_obs(self, obs: np.ndarray) -> np.ndarray:
        m_state = self.step()

        if not hasattr(self, '_debug_step_counter'):
            self._debug_step_counter = 0
        self._debug_step_counter += 1

        if m_state == 0:
            # 正常状态，不施加攻击
            return obs

        # 突发状态：施加动量累积漂移
        drift = self.generate_drift()
        return obs + drift.cpu().numpy()
# ==========================================
# NSAOP-act: 非平稳动作攻击器（间歇性执行器故障）
# ==========================================

class NSAOPActAttacker:
    """
    非平稳动作攻击器：模拟间歇性执行器故障

    物理对应:
    - 马尔可夫故障门控: 电机间歇性卡滞、液压泄漏
    - 状态依赖定向偏差: 特定方向出力不足（如关节极限位置）
    - 动量累积: 温度导致的扭矩逐渐衰减
    """
    def __init__(
        self,
        action_dim: int,
        burst_prob: float = 0.1,        # 故障突发概率（比观测故障更低频）
        recover_prob: float = 0.3,       # 恢复概率（更难恢复）
        bias_scale: float = 1,         # 基础偏差强度
        momentum: float = 0.85,           # 偏差动量的累积因子
        device: str = "cpu"
    ):
        self.action_dim = action_dim
        self.burst_prob = burst_prob
        self.recover_prob = recover_prob
        self.bias_scale = bias_scale
        self.momentum = momentum
        self.device = device

        self.m_state = 0  # 0: normal, 1: faulty
        self.accumulated_bias = torch.zeros(action_dim, device=device)

    def step(self):
        """更新马尔可夫故障状态"""
        if self.m_state == 0:
            if np.random.rand() < self.burst_prob:
                self.m_state = 1
        else:
            if np.random.rand() < self.recover_prob:
                self.m_state = 0
        return self.m_state

    def attack_act(self, action: np.ndarray) -> np.ndarray:
        """对动作施加 NSAOP 攻击"""
        m_state = self.step()

        if m_state == 0:
            self.accumulated_bias.zero_()
            return action

        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        # 偏差方向与当前动作方向相关（模拟特定方向出力不足）
        direction = torch.sign(action_tensor + 1e-8)
        new_bias = direction * self.bias_scale

        # 动量累积: 模拟温度效应导致的扭矩逐渐衰减
        self.accumulated_bias = (
            self.momentum * self.accumulated_bias
            + (1 - self.momentum) * new_bias
        )

        perturbed = action_tensor + self.accumulated_bias
        return np.clip(perturbed.cpu().numpy(), -1.0, 1.0)


# ==========================================
# NSAOP-rew: 非平稳奖励攻击器（间歇性反馈信号污染）
# ==========================================

class NSAOPRewAttacker:
    """
    非平稳奖励攻击器：模拟间歇性反馈信号污染 (NSAOP‑rew)

    物理对应:
    - 马尔可夫突发误导: 通信丢包、欺骗性奖励注入
    - 虚假高奖励注入: 攻击者向控制器注入虚假的高回报信号，
      迫使 RTG 异常快速衰减，让模型误以为目标已达成并停止努力
    - 时序聚集性: 攻击倾向于在关键时间步集中发生
    """
    def __init__(
        self,
        burst_prob: float = 0.3,          # 突发概率
        recover_prob: float = 0.1,        # 恢复概率 (更低 = 故障持续更久)
        attack_mode: str = "fake_high",    # "fake_high": 注入虚假高奖励
        fake_reward_value: float = 1.0,   # 虚假奖励的具体数值
        device: str = "cpu"
    ):
        self.burst_prob = burst_prob
        self.recover_prob = recover_prob
        self.attack_mode = attack_mode
        self.fake_reward_value = fake_reward_value
        self.device = device

        self.m_state = 0  # 0: normal, 1: misleading

    def step(self):
        """更新马尔可夫误导状态"""
        if self.m_state == 0:
            if np.random.rand() < self.burst_prob:
                self.m_state = 1
        else:
            if np.random.rand() < self.recover_prob:
                self.m_state = 0
        return self.m_state

    def attack_rew(self, reward: float) -> float:
        """对奖励施加 NSAOP 攻击"""
        m_state = self.step()

        if m_state == 0:
            # 正常状态，返回真实奖励
            return reward

        # 误导状态：根据模式返回攻击值
        if self.attack_mode == "fake_high":
            # 注入虚假高奖励，让 RTG 快速衰减
            return self.fake_reward_value
        elif self.attack_mode == "flip":
            # 保留原始反转模式（可选）
            return -abs(reward)
        elif self.attack_mode == "zero":
            return 0.0
        # 默认回退
        return reward
