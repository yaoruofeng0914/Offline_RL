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
import utils.dt_functions_UDT as dt_func

from torch.nn import functional as F
from tqdm.auto import trange  # noqa
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
    embedding_dim: int = 128 # 768
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
    embed_order: str = "rsa"
    # training params
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 100_000
    reward_scale: float = 0.001
    normalize: bool = True # Normalize states
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
    dataset_path: str = os.path.expanduser("~/Offline_RL/")
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

    test_attack_mode: str = ""   # 留空表示跟随 corruption_mode，设为 "nsaop" 启用新基准

    beta: float = 0.00001   # KL散度正则化系数
    use_udt: bool = True

    # ========== 新增：RDT 训练策略参数 ==========
    loss_fn: str = "wmse"                   # 使用 WMSE 损失
    wmse_coef: Tuple[float, float] = (0.0, 0.0)  # (act, rew) 的 WMSE 系数
    reward_coef: float = 1.0                # 奖励预测损失的权重
    recalculate_return: bool = False        # 是否重新计算回报
    correct_freq: int = 1                   # 异常值校正频率
    correct_start: int = 50                 # 从第几个 epoch 开始校正
    correct_thershold: Tuple[float] = None  # 异常值校正阈值 (act, rew)
    # ===========================================
    # ========== 新增：UDT 熵权参数 ==========
    loss_weight_type: str = "entropy"  # 可选: "wmse", "entropy"
    entropy_k: float = 0.1            # sigmoid 陡峭度 (调小让过渡平滑)
    entropy_theta: float = 182.0
    lambda_ent : float = 3e-6 # 中心阈值 (略高于当前熵均值 181)
    entropy_baseline: float = 184.0
    # ===========================================

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
        # if self.eval_only:
            # assert self.checkpoint_dir is not None, "Please provide checkpoint_dir for evaluation."
            #self.checkpoint_dir = os.path.join(self.logdir, self.group, self.env, self.checkpoint_dir)
            #with open(os.path.join(self.checkpoint_dir, "params.json"), "r") as f:
                #config = json.load(f)
            #unoverwritten_keys = ["eval_id", "test_time", "group", "checkpoint_dir", "eval_only", "eval_attack", "eval_attack_mode", "eval_attack_eps", "eval_corruption_rate"]
            #for key, value in config.items():
                #if key not in unoverwritten_keys:
                    #try:
                        #value = eval(value)
                    #except:
                        #pass
                    #self.__dict__[key] = value
                    # print(f"Set {key} to {value}")
        self.eval_attack_mode = self.corruption_mode # random, adversarial
        self.eval_attack_eps = 1
        self.eval_corruption_rate = 0.3
        if self.eval_attack_mode == "random" and self.corruption_tag == "rew":
            self.eval_attack_eps *= 30
        # 如果指定了 test_attack_mode，则使用指定的模式；否则跟随训练模式
        # if self.test_attack_mode:
        #     self.eval_attack_mode = self.test_attack_mode
        # else:
        #     self.eval_attack_mode = self.corruption_mode


def set_model(config: TrainConfig):
    model = dt_func.DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        predict_reward=True,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        mlp_embedding=config.mlp_embedding,
        mlp_head=config.mlp_head,
        mlp_reward=config.mlp_reward,
        use_udt=config.use_udt,
        embed_order=config.embed_order,
    ).to(config.device)
    return model


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
def loss_fn(config, predicted, target, mask, coef=None, log_var=None):
    """
    统一的损失函数，支持 WMSE 和 Entropy-based 权重，并返回详细日志。
    """
    log_info = {}

    if config.loss_weight_type == "wmse":
        # 原始 WMSE 逻辑
        with torch.no_grad():
            diff = torch.square(predicted.detach() - target.detach()).mean(-1, keepdim=True)
            weight = torch.exp(-coef * diff)

        log_info["weight_mean"] = weight[mask.to(torch.bool)].mean().item()

    elif config.loss_weight_type == "entropy":
        # 基于微分熵的信任度加权
        if log_var is None:
            raise ValueError("Using 'entropy' weight requires log_var from BayesianEmbedding.")

        # 阻断梯度
        log_var_detached = log_var.detach()

        # 计算微分熵 H = 0.5 * sum(logvar + 1 + ln(2π))  按嵌入维度求和
        entropy = 0.5 * torch.sum(log_var_detached + 1.0 + np.log(2 * np.pi), dim=-1, keepdim=True)

        # 信任度系数 τ = 1 / (1 + exp(k * (H - θ)))
        weight = 1.0 / (1.0 + torch.exp(config.entropy_k * (entropy - config.entropy_theta)))

        # 日志记录
        valid_mask = mask.to(torch.bool).unsqueeze(-1)
        valid_entropy = entropy[valid_mask]
        valid_weight = weight[valid_mask]

        log_info["entropy_mean"] = valid_entropy.mean().item()
        log_info["entropy_std"]  = valid_entropy.std().item() if valid_entropy.numel() > 1 else 0.0
        log_info["entropy_max"]  = valid_entropy.max().item() if valid_entropy.numel() > 0 else 0.0
        log_info["entropy_min"]  = valid_entropy.min().item() if valid_entropy.numel() > 0 else 0.0
        log_info["weight_mean"]  = valid_weight.mean().item()
    else:
        raise ValueError(f"Unknown loss_weight_type: {config.loss_weight_type}")

    # 计算加权 MSE 损失
    loss = F.mse_loss(predicted, target.detach(), reduction="none")
    loss = (loss * weight * mask.unsqueeze(-1)).mean()

    return loss, log_info

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
    config.state_std = dataset.state_std
    config.rew_std = dataset.rew_std
    config.act_std = dataset.act_std
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
        if config.test_attack_mode == "nsaop":
            eval_attacker = None  # 由 eval_rollout 内部创建 NSAOP 攻击器
            print("eval_attack: NSAOP enabled during training eval")
        else:
            state_std, act_std, rew_std, rew_min = func.get_state_std(config)
            print(config.eval_attack_mode)
            eval_attacker = Evaluation_Attacker(
                config, config.env, config.corruption_agent, config.eval_attack_eps,
                config.state_dim, config.action_dim, state_std, act_std, rew_std, rew_min,
                config.eval_attack_mode,
                MODEL_PATH[config.corruption_agent],
            )
            eval_attacker.attack_mode = config.eval_attack_mode
            print("eval_attack: True (original attacker)")
    else:
        eval_attacker = None
        print("eval_attack: False")

    if config.debug_eval:
        print(eval_attacker.attack_mode)
        model.eval()
        # model.skip_gating = True  # 关闭门控
        eval_log = dt_func.eval_fn(config, env, model, eval_attacker)
        # model.skip_gating = False  # 恢复门控
        model.train()
        logger.record("epoch", 0)
        for k, v in eval_log.items():
            logger.record(k, v)
        logger.dump(0)

    # if config.use_wandb:
    #     wandb.log({"epoch": 0, **eval_log})
    # 初始化异常值校正所需的统计量
    data_dist = []
    if config.correct_thershold is not None:
        for thershold in config.correct_thershold:
            data_dist.append(RunningMeanStd(thershold=thershold) if thershold > 0.0 else None)

    total_updates = 0
    best_score = -np.inf
    best_score_50 = -np.inf

    for epoch in trange(1, config.num_epochs + 1, desc="Training"):
        time_start = time.time()  # 添加这一行
        # log_lambda 默认 requires_grad=True，无需手动设置
        for step in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            log_dict = {}
            # batch = next(trainloader_iter)
            batch = dataset.get_batch(config.batch_size, config.recalculate_return)
            states, actions, returns, rewards, time_steps, mask, attack_mask, traj_indexs = [b.to(config.device) for b
                                                                                             in batch]
            padding_mask = ~mask.to(torch.bool)

            predicted = model(
                states=states,
                actions=actions,
                returns_to_go=returns,
                time_steps=time_steps,
                padding_mask=padding_mask,
            )
            predicted_actions = predicted[0]
            predicted_rewards = predicted[1]  # 奖励预测，WMSE 和异常值校正需要
            # UDT 特有的嵌入参数
            udt_info = predicted[2]

            # ========== RDT 风格的加权均方误差 (WMSE) ==========
            # ========== 解包 udt_info，准备 logvar ==========
            if config.use_udt and udt_info is not None:
                state_mu, state_logvar, act_mu, act_logvar, ret_mu, ret_logvar = udt_info
                state_logvar_for_loss = state_logvar
                act_logvar_for_loss = act_logvar
                ret_logvar_for_loss = ret_logvar
            else:
                act_logvar_for_loss = None
                ret_logvar_for_loss = None

            # ========== 动作和奖励损失（根据 loss_weight_type 自动选择 wmse 或 entropy） ==========
            loss_action, action_log_info = loss_fn(
                config, predicted_actions, actions, mask,
                coef=config.wmse_coef[0], log_var=state_logvar_for_loss
            )
            loss_reward, reward_log_info = loss_fn(
                config, predicted_rewards, rewards, mask,
                coef=config.wmse_coef[1], log_var=act_logvar_for_loss
            )

            # ========== KL 散度约束与退火 + 方案 B 熵提升 ==========
            kl_total = torch.tensor(0.0, device=config.device)
            current_beta = 0.0
            ent_boost_loss = torch.tensor(0.0, device=config.device)

            if config.use_udt and udt_info is not None:
                state_mu, state_logvar, act_mu, act_logvar, ret_mu, ret_logvar = udt_info

                # --- KL 散度 ---
                kl_state = -0.5 * torch.sum(1 + state_logvar - state_mu.pow(2) - state_logvar.exp(), dim=-1).mean()
                kl_act = -0.5 * torch.sum(1 + act_logvar - act_mu.pow(2) - act_logvar.exp(), dim=-1).mean()
                kl_ret = -0.5 * torch.sum(1 + ret_logvar - ret_mu.pow(2) - ret_logvar.exp(), dim=-1).mean()
                kl_total = kl_state + kl_act + kl_ret

                kl_warmup = getattr(config, 'kl_warmup_steps', 20000)
                beta_target = getattr(config, 'beta', 0.00001)
                if kl_warmup > 0:
                    current_beta = beta_target * min(1.0, total_updates / kl_warmup)
                else:
                    current_beta = beta_target

                if config.lambda_ent > 0 and attack_mask is not None:
                    attack_mask_bool = (attack_mask.squeeze(-1) > 0).to(torch.bool)
                    valid_mask = mask.to(torch.bool)
                    attack_token_mask = valid_mask & attack_mask_bool

                    if attack_token_mask.sum() > 0:
                        state_entropy = 0.5 * torch.sum(
                            state_logvar + 1.0 + np.log(2 * np.pi), dim=-1
                        )
                        act_entropy = 0.5 * torch.sum(
                            act_logvar + 1.0 + np.log(2 * np.pi), dim=-1
                        )

                        baseline = config.entropy_baseline
                        ent_boost = (
                                            (state_entropy[attack_token_mask] - baseline).clamp(min=0).mean() +
                                            (act_entropy[attack_token_mask] - baseline).clamp(min=0).mean()
                                    ) * 0.5
                        ent_boost_loss = - config.lambda_ent * ent_boost

            # 合并总损失（KL 和方案 B 已经在上面计算完毕）
            loss_total = loss_action + config.reward_coef * loss_reward + current_beta * kl_total + ent_boost_loss
            optim.zero_grad()
            loss_total.backward()

            if config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optim.step()

            # ========== 熵监控（观察 KL 加入后的变化） ==========
            if config.use_udt and udt_info is not None:
                # 重新解包 udt_info，得到 logvar（忽略 mu）
                _, state_logvar, _, act_logvar, _, ret_logvar = udt_info

                def calc_entropy(logvar, mask, attack_mask, name, log_dict):
                    with torch.no_grad():
                        entropy = 0.5 * torch.sum(logvar + 1.0 + np.log(2 * np.pi), dim=-1)  # (batch, seq_len)
                        valid_mask = mask.to(torch.bool)
                        attack_mask_bool = (attack_mask.squeeze(-1) > 0).to(torch.bool)

                        all_ent = entropy[valid_mask]
                        attack_ent = entropy[valid_mask & attack_mask_bool]

                        log_dict[f"entropy/{name}_mean"] = all_ent.mean().item()
                        log_dict[f"entropy/{name}_std"] = all_ent.std().item() if all_ent.numel() > 1 else 0.0
                        log_dict[f"entropy/{name}_max"] = all_ent.max().item() if all_ent.numel() > 0 else 0.0
                        log_dict[f"entropy/{name}_min"] = all_ent.min().item() if all_ent.numel() > 0 else 0.0

                        if attack_ent.numel() > 0:
                            log_dict[f"entropy/attack_{name}_mean"] = attack_ent.mean().item()
                            log_dict[
                                f"entropy/attack_{name}_std"] = attack_ent.std().item() if attack_ent.numel() > 1 else 0.0
                        else:
                            log_dict[f"entropy/attack_{name}_mean"] = 0.0
                            log_dict[f"entropy/attack_{name}_std"] = 0.0

                calc_entropy(state_logvar, mask, attack_mask, "state", log_dict)
                calc_entropy(act_logvar, mask, attack_mask, "act", log_dict)
                calc_entropy(ret_logvar, mask, attack_mask, "ret", log_dict)
            # =====================================================

            # 记录日志（只需一次）
            log_dict.update({
                "loss_action": loss_action.item(),
                "loss_reward": loss_reward.item(),
                "kl_loss": kl_total.item(),
                "current_beta": current_beta,
                "ent_boost_loss": ent_boost_loss.item(),
                "policy_loss": loss_total.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            })
            for k, v in action_log_info.items():
                log_dict[f"act_{k}"] = v
            for k, v in reward_log_info.items():
                log_dict[f"rew_{k}"] = v
            scheduler.step()
            total_updates += 1

            # ========== 新增：异常值校正 ==========
            correct_dict = {}
            if config.correct_thershold is not None:
                # 每隔 correct_freq 步且在 correct_start 之后才执行
                correct = epoch > config.correct_start and step % config.correct_freq == 0
                data_info = {
                    "actions": [predicted_actions, actions],
                    "rewards": [predicted_rewards, rewards],
                    "mask": mask,
                    "attack_mask": attack_mask,
                    "traj_indexs": traj_indexs,
                    "time_steps": time_steps,
                }
                correct_info, correct_dict = correct_outliers(config, data_info, data_dist, correct=correct)
                if correct:
                    for name, info in correct_info.items():
                        dataset.correct(info["traj_indexs"], info["time_steps"], info["correct_data"], name)
                    if config.correct_thershold is not None and config.correct_thershold[1] > 0.0:
                        config.recalculate_return = True
            # ====================================
        time_end = time.time()
        epoch_time = time_end - time_start

        # validation in the env for the actual online performance
        if epoch % config.eval_every == 0 and epoch > (config.num_epochs - config.eval_final):
            model.eval()
            # model.skip_gating = True      # 关闭门控
            eval_log = dt_func.eval_fn(config, env, model, eval_attacker)
            # model.skip_gating = False
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
    config.state_std = dataset.state_std
    config.rew_std = dataset.rew_std
    config.act_std = dataset.act_std
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
        if config.test_attack_mode == "nsaop":
            eval_attacker = None
            print("eval_attack: NSAOP (原有攻击器已禁用)")
        else:
            state_std, act_std, rew_std, rew_min = func.get_state_std(config)
            print(config.eval_attack_mode)
            eval_attacker = Evaluation_Attacker(
                config, config.env, config.corruption_agent, config.eval_attack_eps,
                config.state_dim, config.action_dim, state_std, act_std, rew_std, rew_min,
                config.eval_attack_mode,
                MODEL_PATH[config.corruption_agent],
            )
            eval_attacker.attack_mode = config.eval_attack_mode
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
        model.skip_gating = False
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