"""Configuration dataclasses and CLI parsing for trajectory-centric skill discovery."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Literal

RewardType = Literal["diayn", "surrogate", "info_gain", "var_infonce"]


@dataclass
class EnvConfig:
    env_name: str = "dm_control/hopper-hop-v0"
    max_episode_len: int = 1000
    max_n_episodes: int = 5000


@dataclass
class SkillConfig:
    n_skills: int = 50
    skill_embed_dim: int = 64
    traj_len: int = 8  # k: trajectory window length
    reward_type: RewardType = "surrogate"
    n_negatives: int = 16  # K for variational InfoNCE, not used for now


@dataclass
class SACConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    discriminator_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    alpha: float = 0.1
    tau: float = 0.005
    n_hiddens: int = 1024
    reward_scale: float = 1.0
    mem_size: int = 1_000_000
    target_update_interval: int = 2
    seed: int = 11042004
    learning_alpha: bool = True



@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    log_interval: int = 20
    use_wandb: bool = False
    wandb_project: str = "Temp"
    run_name: str = ""


def get_train_config() -> TrainConfig:
    """Parse CLI and return merged TrainConfig."""
    parser = argparse.ArgumentParser(description="Trajectory-Centric Skill Discovery")
    parser.add_argument("--env_name", type=str, default="dm_control/hopper-hop-v0")
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--max_n_episodes", type=int, default=5000)
    parser.add_argument("--n_skills", type=int, default=50)
    parser.add_argument("--traj_len", type=int, default=16)
    parser.add_argument(
        "--reward_type",
        type=str,
        default="surrogate",
        choices=["diayn", "surrogate", "info_gain", "var_infonce"],
    )
    parser.add_argument("--n_negatives", type=int, default=16)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--discriminator-lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--learning_alpha", action="store_true")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--n_hiddens", type=int, default=300)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--mem_size", type=int, default=1_000_000)
    parser.add_argument("--target_update_interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Temp")
    parser.add_argument("--run_name", type=str, default="")

    args = parser.parse_args()

    return TrainConfig(
        env=EnvConfig(
            env_name=args.env_name,
            max_episode_len=args.max_episode_len,
            max_n_episodes=args.max_n_episodes,
        ),
        skill=SkillConfig(
            n_skills=args.n_skills,
            traj_len=args.traj_len,
            reward_type=args.reward_type,  # type: ignore[arg-type]
            n_negatives=args.n_negatives,
        ),
        sac=SACConfig(
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            alpha_lr=args.alpha_lr,
            discriminator_lr=args.discriminator_lr,
            batch_size=args.batch_size,
            gamma=args.gamma,
            alpha=args.alpha,
            tau=args.tau,
            n_hiddens=args.n_hiddens,
            reward_scale=args.reward_scale,
            mem_size=args.mem_size,
            seed=args.seed,
        ),
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )


def train_config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Flatten config for logging / checkpoints."""
    return {
        "env_name": cfg.env.env_name,
        "max_episode_len": cfg.env.max_episode_len,
        "max_n_episodes": cfg.env.max_n_episodes,
        "n_skills": cfg.skill.n_skills,
        "traj_len": cfg.skill.traj_len,
        "reward_type": cfg.skill.reward_type,
        "n_negatives": cfg.skill.n_negatives,
        "actor_lr": cfg.sac.actor_lr,
        "critic_lr": cfg.sac.critic_lr,
        "alpha_lr": cfg.sac.alpha_lr,
        "discriminator_lr": cfg.sac.discriminator_lr,
        "batch_size": cfg.sac.batch_size,
        "gamma": cfg.sac.gamma,
        "alpha": cfg.sac.alpha,
        "tau": cfg.sac.tau,
        "n_hiddens": cfg.sac.n_hiddens,
        "reward_scale": cfg.sac.reward_scale,
        "mem_size": cfg.sac.mem_size,
        "seed": cfg.sac.seed,
        "log_interval": cfg.log_interval,
        "use_wandb": cfg.use_wandb,
        "wandb_project": cfg.wandb_project,
        "run_name": cfg.run_name,
    }
