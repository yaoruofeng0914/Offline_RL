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
import gc

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
# @torch.no_grad()
# @torch.no_grad()
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

    obs = env.reset()
    if eval_attacker is not None and eval_attack_tag == "obs":
        attack_flag = np.random.rand()
        if attack_flag < eval_corruption_rate:
            obs = eval_attacker.attack_obs(obs)
    states[:, 0] = torch.as_tensor(obs, device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = 0.0, 0.0
    env_is_poisoned = False
    ema_kpe_ratio = 1.0
    for step in range(model.episode_len):
        hist_kpe = 0.5
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)
        if model.use_asts and not model.training:
            with torch.no_grad():
                clean_preds = model(
                    states[:, : step + 1][:, -model.seq_len:],
                    actions[:, : step + 1][:, -model.seq_len:],
                    returns[:, : step + 1][:, -model.seq_len:],
                    time_steps[:, : step + 1][:, -model.seq_len:],
                )
                clean_action_preds = clean_preds[0]
                clean_curr_act = clean_action_preds.mean[0, -1] if use_stochastic else clean_action_preds[0, -1]

            # =================================================================
            # 🌟 1. 触发器重构：多模态融合验证
            # =================================================================
            trigger_asts = env_is_poisoned

            if not trigger_asts and model.use_koopman and step > 0:
                # 1. 过去状态的真实流形 (去掉多余维度，保证矩阵乘法安全: [1, hidden_dim])
                g_prev = model.koopman.g(states[:, step - 1]).squeeze(0)
                # 2. 当前状态的真实流形
                g_curr = model.koopman.g(states[:, step]).squeeze(0)

                # 3. Koopman 基于物理法则外推的当前流形
                action_prev = actions[:, step - 1].squeeze(0)
                g_curr_pred = torch.matmul(g_prev, model.koopman.K) + model.koopman.B(action_prev)

                # 4. 计算 Koopman 流形预测误差 (KPE)
                kpe_error = torch.norm(g_curr - g_curr_pred).item()

                # 5. 动态环境基准：不要用写死的 0.5。计算过去几步的平滑误差作为基线。

                if step > 2:
                    g_hist_prev = model.koopman.g(states[:, max(0, step - 5):step - 1])  # [1, T, hidden]
                    g_hist_curr = model.koopman.g(states[:, max(1, step - 4):step])
                    a_hist_prev = actions[:, max(0, step - 5):step - 1]

                    g_hist_pred = torch.matmul(g_hist_prev, model.koopman.K) + model.koopman.B(a_hist_prev)
                    hist_kpe = torch.norm(g_hist_curr - g_hist_pred, dim=-1).mean().item()

                    # 如果当前瞬间误差，超过历史平滑误差的 2.5 倍（且有绝对下限），说明流形被黑客撕裂了！
                    base_kpe = max(hist_kpe * 2.5, 0.2)
                else:
                    base_kpe = 0.5

                if kpe_error > base_kpe:
                    trigger_asts = True

            # =================================================================
            # 🌟 2. 带有 L2 信任域和自我锚点的优化器 (TR-ASTS)
            # =================================================================
            # =================================================================
            # 🌟 2. 带有 L2 信任域和自我锚点的优化器 (TR-ASTS)
            # =================================================================
            if trigger_asts:
                model.steering_vector.data.zero_()

                # 🌟 将 max_norm 的计算提出来！保证每个物理 step 只更新一次阻尼状态！
                if model.use_koopman and step > 2:
                    # 1. 提取当前瞬态 KPE 与 历史平滑 KPE
                    instant_kpe = kpe_error if 'kpe_error' in locals() else 0.5
                    smoothed_kpe = hist_kpe

                    # 2. 计算无量纲比值 R_t (Scale Invariance)
                    raw_ratio = instant_kpe / (smoothed_kpe + 1e-6)

                    # 3. 一阶低通阻尼滤波 (Absorbing Spikes)
                    lam = 0.2
                    ema_kpe_ratio = (1 - lam) * ema_kpe_ratio + lam * raw_ratio

                    # 4. 指数弹性边界计算
                    alpha = 0.4  # SAM 护城河带来的底气：平坦流形下的极大宽容度
                    beta = 1.5   # 对恶性撕裂攻击的敏感衰减因子

                    penalty_factor = max(0.0, ema_kpe_ratio - 1.0)
                    dynamic_upper = alpha * np.exp(-beta * penalty_factor)

                    # 5. 守住绝对防线
                    max_norm = max(0.05, dynamic_upper)
                else:
                    # 开局或无 Koopman 时的后备保守策略
                    max_norm = 0.1

                with torch.enable_grad():
                    model.steering_vector.requires_grad = True
                    optimizer = torch.optim.Adam([model.steering_vector], lr=0.005)

                    for i_step in range(3):
                        optimizer.zero_grad()
                        preds = model(
                            states[:, : step + 1][:, -model.seq_len:],
                            actions[:, : step + 1][:, -model.seq_len:],
                            returns[:, : step + 1][:, -model.seq_len:],
                            time_steps[:, : step + 1][:, -model.seq_len:],
                        )

                        steer_loss = 0.0
                        action_preds = preds[0]
                        curr_act = action_preds.mean[0, -1] if use_stochastic else action_preds[0, -1]

                        if model.predict_reward and preds[1] is not None:
                            steer_loss -= preds[1][0, -1].mean() * 0.1

                        steer_loss += F.mse_loss(curr_act, clean_curr_act.detach()) * 0.2

                        if model.use_koopman:
                            shift_steered = model.koopman.B(curr_act.unsqueeze(0))
                            shift_clean = model.koopman.B(clean_curr_act.detach().unsqueeze(0))
                            steer_loss += torch.norm(shift_steered - shift_clean) * 0.5

                        steer_loss.backward(retain_graph=(i_step < 2))
                        optimizer.step()

                        # 🌟 投影截断保留在循环内，每次梯度走完都检查是否越界
                        vec_norm_val = torch.norm(model.steering_vector.data).item()
                        if vec_norm_val > max_norm:
                            model.steering_vector.data.mul_(max_norm / (vec_norm_val + 1e-8))

                    del steer_loss, action_preds, preds, curr_act

                # ============================================================
                # 🌟 3. MPC 潜空间线搜索 (修复维度映射冲突)
                # ============================================================
                raw_steering_vec = model.steering_vector.data[0, 0, :].clone()
                best_scale = 0.0

                if model.predict_reward:
                    with torch.no_grad():
                        model.steering_vector.data.zero_()
                        hypo_actions_clean = actions[:, : step + 1][:, -model.seq_len:].clone()
                        hypo_actions_clean[0, -1] = clean_curr_act

                        preds_eval_clean = model(
                            states[:, : step + 1][:, -model.seq_len:],
                            hypo_actions_clean,
                            returns[:, : step + 1][:, -model.seq_len:],
                            time_steps[:, : step + 1][:, -model.seq_len:],
                        )
                        reward_clean = preds_eval_clean[1][0, -1].item() if preds_eval_clean[1] is not None else 0.0

                        search_scales = [1.0, 0.5, 0.25, 0.1]
                        for scale in search_scales:
                            model.steering_vector.data[0, 0, :] = raw_steering_vec * scale

                            steered_preds = model(
                                states[:, : step + 1][:, -model.seq_len:],
                                actions[:, : step + 1][:, -model.seq_len:],
                                returns[:, : step + 1][:, -model.seq_len:],
                                time_steps[:, : step + 1][:, -model.seq_len:],
                            )
                            test_act = steered_preds[0].mean[0, -1] if use_stochastic else steered_preds[0][0, -1]

                            if torch.norm(test_act - clean_curr_act).item() > 0.4:
                                continue

                            hypo_actions_steered = actions[:, : step + 1][:, -model.seq_len:].clone()
                            hypo_actions_steered[0, -1] = test_act

                            eval_steered = model(
                                states[:, : step + 1][:, -model.seq_len:],
                                hypo_actions_steered,
                                returns[:, : step + 1][:, -model.seq_len:],
                                time_steps[:, : step + 1][:, -model.seq_len:],
                            )
                            reward_steered = eval_steered[1][0, -1].item() if eval_steered[1] is not None else 0.0

                            if reward_steered >= reward_clean - 0.02:
                                best_scale = scale
                                break
                else:
                    best_scale = 1.0 if torch.norm(raw_steering_vec).item() <= 0.04 else 0.0

                model.steering_vector.data.zero_()
                model.steering_vector.data[0, 0, :] = raw_steering_vec * best_scale
            else:
                model.steering_vector.data.zero_()
        with torch.no_grad():
            predicted = model(  # fix this noqa!!!
                states[:, : step + 1][:, -model.seq_len:],
                actions[:, : step + 1][:, -model.seq_len:],
                returns[:, : step + 1][:, -model.seq_len:],
                time_steps[:, : step + 1][:, -model.seq_len:],
            )
            predicted_actions = predicted[0]
            if use_stochastic:
                predicted_actions = predicted_actions.mean
            if torch.isnan(predicted_actions[0, -1]).any() or torch.isinf(predicted_actions[0, -1]).any():
                break
            predicted_action = predicted_actions[0, -1].cpu().numpy()
            if eval_attacker is not None and eval_attack_tag == "act":
                attack_flag = np.random.rand()
                if attack_flag < eval_corruption_rate:
                    predicted_action = eval_attacker.attack_act(predicted_action)
            predicted_action = np.clip(predicted_action, *action_range)
            next_state, reward, done, info = env.step(predicted_action)
            episode_return += reward
            episode_len += 1
            if eval_attacker is not None and eval_attack_tag == "obs":
                attack_flag = np.random.rand()
                if attack_flag < eval_corruption_rate:
                    next_state = eval_attacker.attack_obs(next_state)
            if eval_attacker is not None and eval_attack_tag == "rew":
                attack_flag = np.random.rand()
                if attack_flag < eval_corruption_rate:
                    reward = eval_attacker.attack_rew(reward)
            pred_reward_val = 0.0
            if model.predict_reward and predicted[1] is not None:
                pred_reward_val = predicted[1][0, -1].cpu().item()

            divergence_threshold = abs(pred_reward_val) * 0.5 + 0.05
            if abs(pred_reward_val - reward) > divergence_threshold and model.use_asts and not model.training:
                env_is_poisoned = True  # 触发测谎仪
                trusted_reward = pred_reward_val  # 截断毒源
            else:
                env_is_poisoned = False
                trusted_reward = reward
            # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
            actions[:, step] = torch.as_tensor(predicted_action)
            states[:, step + 1] = torch.as_tensor(next_state)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - trusted_reward)

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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return eval_log
# ==========================================
# 🌟 极简且标准的 SAM 优化器实现
# ==========================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # 爬上恶劣的锐度山峰
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # 退回原位，但在原位使用山峰上的梯度更新
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


# ==========================================
# 🌟 创新点 1 (重构版): 基于 SWT 的双通道频域交叉注意力融合 (WFDiffuser 范式)
# 完全可微分，不丢失高频的物理特征，利用因果门控进行自主防守
# ==========================================
class DifferentiableSWT(nn.Module):
    """
    可微分的平稳小波变换 (Stationary Wavelet Transform, SWT)
    基于 Haar 小波，使用不可训练的 1D 卷积实现，支持 GPU 加速和 autograd
    保证输出的时间分辨率与输入完全一致 (无下采样)
    """

    def __init__(self, channels):
        super().__init__()
        # Haar 小波滤波器系数
        h0 = np.array([1, 1]) / np.sqrt(2)  # 低通
        h1 = np.array([-1, 1]) / np.sqrt(2)  # 高通

        # 构造卷积核 [channels, 1, 2] 用于 Depthwise Conv
        weight_low = torch.tensor(h0, dtype=torch.float32).view(1, 1, 2).repeat(channels, 1, 1)
        weight_high = torch.tensor(h1, dtype=torch.float32).view(1, 1, 2).repeat(channels, 1, 1)

        self.register_buffer('weight_low', weight_low)
        self.register_buffer('weight_high', weight_high)
        self.channels = channels

    def forward(self, x):
        # x: [Batch, Length, Channels]
        x_t = x.transpose(1, 2)  # [B, C, L]
        seq_len = x_t.size(-1)

        # 动态获取卷积核长度
        kernel_size = self.weight_low.size(-1)

        # 极其鲁棒的因果补零：在左侧补 (kernel_size - 1) 个零
        x_pad = F.pad(x_t, (kernel_size - 1, 0), mode='constant', value=0.0)

        # Depthwise 1D 卷积分别提取低频 (趋势) 和高频 (细节)
        x_low = F.conv1d(x_pad, self.weight_low, groups=self.channels)
        x_high = F.conv1d(x_pad, self.weight_high, groups=self.channels)

        # 终极保险：强制截取前 seq_len 个时间步，确保维度绝对对齐
        x_low = x_low[..., :seq_len]
        x_high = x_high[..., :seq_len]

        return x_low.transpose(1, 2), x_high.transpose(1, 2)  # [B, L, C], [B, L, C]


class GatedCrossFrequencyInjection(nn.Module):
    """
    门控频域交叉融合机制：让主干低频 (宏观动力学) 去 Query 高频细节 (专家微操)。
    合法的物理突发将被引入，恶意的对抗噪声将被门控阻断。
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.swt = DifferentiableSWT(embed_dim)

        # 交叉注意力：Low freq (Query), High freq (Key, Value)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        # KPE 条件映射：接收 [绝对KPE, 对数比值] 两维输入
        self.kpe_condition = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
        nn.init.zeros_(self.kpe_condition[0].weight)
        nn.init.zeros_(self.kpe_condition[0].bias)

        # 门控网络：偏置初始化为 2.0，sigmoid 输出 ≈ 0.88
        self.gating_network = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        nn.init.normal_(self.gating_network[-2].weight, mean=0.0, std=0.02)
        nn.init.constant_(self.gating_network[-2].bias, 1.0)

        # FFN 融合后处理
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, kpe=None):
        # 1. 频域分解
        x_low, x_high = self.swt(x)

        # 2. 交叉注意力（因果掩码）
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        attn_output, _ = self.cross_attn(
            query=x_low, key=x_high, value=x_high,
            need_weights=False, is_causal=True, attn_mask=causal_mask
        )

        # 3. 处理 KPE 条件特征
        if kpe is not None:
            kpe_feat = self.kpe_condition(kpe)  # [B, L, embed_dim]
        else:
            kpe_feat = torch.zeros_like(x_low)

        # 4. 动态门控（拼接三个信息源）
        gate = self.gating_network(torch.cat([x_low, attn_output, kpe_feat], dim=-1))
        injected_features = x_low + gate * attn_output
        injected_features = self.layer_norm1(injected_features)

        # 5. FFN + 残差
        out = injected_features + self.ffn(injected_features)
        return self.layer_norm2(out)
# --- 创新点 2: 对比 Koopman 流形模块 ---
class ContrastiveKoopman(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # 编码器 g(s) 将状态映射到线性流形
        self.g = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 线性算子 K (Koopman Matrix)
        self.K = nn.Parameter(torch.eye(hidden_dim) + torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.B = nn.Linear(action_dim, hidden_dim, bias=False)

    def forward(self, s_t, a_t, s_next):
        g_t = self.g(s_t)  # [B, T, hidden]
        g_next_true = self.g(s_next)
        # 谱范数有界化
        k_norm = torch.norm(self.K, p='fro')
        safe_K = self.K / torch.clamp(k_norm, min=1.0)
        g_next_pred = g_t @ safe_K + self.B(a_t)
        return g_next_true, g_next_pred, g_t

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
            use_mwpa: bool = True,  # 开关1：小波注意力
            use_koopman: bool = True,  # 开关2：对比学习约束
            use_asts: bool = False,  # 开关3：在线引导
    ):
        super().__init__()
        self.use_mwpa = use_mwpa
        self.use_koopman = use_koopman
        self.use_asts = use_asts
        self.corruption_tag = corruption_tag
        if embedding_dropout is not None:
            self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)

        self.state_emb = nn.Linear(state_dim, embedding_dim) if not mlp_embedding else ResidualBlock(state_dim,embedding_dim)

        self.action_emb = nn.Linear(action_dim, embedding_dim) if not mlp_embedding else ResidualBlock(action_dim,embedding_dim)

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
        # 🌟 核心替换：使用频域交叉融合机制替代原有的硬截断滤波
        if self.use_mwpa:
            self.freq_fusion_S = GatedCrossFrequencyInjection(embedding_dim, num_heads=4)
            self.freq_fusion_A = GatedCrossFrequencyInjection(embedding_dim, num_heads=4)
            self.freq_fusion_R = GatedCrossFrequencyInjection(embedding_dim, num_heads=4)
        if self.use_koopman:
            self.koopman = ContrastiveKoopman(state_dim, action_dim, embedding_dim)

            # --- 初始化 ASTS 向量 ---
        self.register_buffer('steering_vector', torch.zeros(1, 1, embedding_dim))

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
        kpe_sequence = None
        if self.use_koopman:
            with torch.no_grad():
                s_t = states[:, :-1, :]
                a_t = actions[:, :-1, :]
                s_next = states[:, 1:, :]
                _, g_next_pred, _ = self.koopman(s_t, a_t, s_next)
                g_next_true = self.koopman.g(s_next)
                kpe_errors = torch.norm(g_next_pred - g_next_true, dim=-1)
                kpe_sequence = F.pad(kpe_errors, (1, 0), value=0.0)

                window = 3
                kpe_padded = F.pad(kpe_sequence.unsqueeze(1), (window - 1, 0), mode='replicate')
                smooth_kpe = F.avg_pool1d(kpe_padded, kernel_size=window, stride=1).squeeze(1)
                smooth_kpe_safe = torch.clamp(smooth_kpe, min=1e-4)
                log_ratio = torch.log1p(kpe_sequence) - torch.log1p(smooth_kpe_safe)
                kpe_sequence = torch.clamp(kpe_sequence, max=100.0)
                log_ratio = torch.clamp(log_ratio, min=-5.0, max=5.0)
                kpe_sequence = torch.stack([kpe_sequence, log_ratio], dim=-1)

        if self.use_koopman:
            state_emb = self.koopman.g(states)
        else:
            state_emb = self.state_emb(states)
        act_emb = self.action_emb(actions)
        returns_emb = self.return_emb(returns_to_go)
        if self.use_mwpa:
            state_emb = self.freq_fusion_S(state_emb, kpe=kpe_sequence)
            act_emb = self.freq_fusion_A(act_emb, kpe=None)  # 去掉 KPE
            returns_emb = self.freq_fusion_R(returns_emb, kpe=None)
            # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        if self.embed_order == "rsa":
            sequence = torch.stack([returns_emb, state_emb, act_emb], dim=1)
        elif self.embed_order == "sar":
            sequence = torch.stack([state_emb, act_emb, returns_emb], dim=1)
        else:
            raise ValueError(f"Invalid embedding order {self.embed_order}.")
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
            out = block(out, padding_mask=padding_mask)
        if not self.training and self.use_asts:
            out = out + self.steering_vector

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