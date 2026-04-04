from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
import shimmy  # noqa: F401
from agent import Agent
from buffer import ReplayBuffer
from config import TrainConfig, get_train_config
from logger import RunLogger
from wrapper import DMControlHopperWrapper, SkillWrapper


class SkillTracker:
    """Running per-skill statistics (return, length, std)."""

    def __init__(self, n_skills: int):
        self.n_skills = n_skills
        self.counts = np.zeros(n_skills, dtype=np.int64)
        self.return_sums = np.zeros(n_skills, dtype=np.float64)
        self.return_sq_sums = np.zeros(n_skills, dtype=np.float64)
        self.length_sums = np.zeros(n_skills, dtype=np.int64)

    def update(self, skill: int, ep_return: float, ep_length: int):
        self.counts[skill] += 1
        self.return_sums[skill] += ep_return
        self.return_sq_sums[skill] += ep_return**2
        self.length_sums[skill] += ep_length

    def mean_return(self, skill: int) -> float:
        if self.counts[skill] == 0:
            return 0.0
        return float(self.return_sums[skill] / self.counts[skill])

    def std_return(self, skill: int) -> float:
        if self.counts[skill] < 2:
            return 0.0
        mean = self.mean_return(skill)
        var = self.return_sq_sums[skill] / self.counts[skill] - mean**2
        return float(np.sqrt(max(var, 0.0)))

    def mean_length(self, skill: int) -> float:
        if self.counts[skill] == 0:
            return 0.0
        return float(self.length_sums[skill] / self.counts[skill])

    def overall_mean_return(self) -> float:
        total = self.counts.sum()
        if total == 0:
            return 0.0
        return float(self.return_sums.sum() / total)

    def active_count(self) -> int:
        return int((self.counts > 0).sum())

    def cross_skill_return_std(self) -> float:
        """Std of per-skill mean returns across active skills (skill diversity)."""
        active = self.counts > 0
        if active.sum() < 2:
            return 0.0
        means = self.return_sums[active] / self.counts[active]
        return float(means.std())


def train(cfg: TrainConfig):
    env = SkillWrapper(
        DMControlHopperWrapper(
            gym.make(cfg.env.env_name, max_episode_steps=cfg.env.max_episode_len)
        ),
        n_skills=cfg.skill.n_skills,
        k=cfg.skill.traj_len,
    )

    obs_dim = env.observation_space["observation"].shape[0]
    act_dim = env.action_space.shape[0]

    agent = Agent(cfg, obs_dim, act_dim)
    buffer = ReplayBuffer(
        capacity=cfg.sac.mem_size,
        n_states=obs_dim,
        n_actions=act_dim,
        np_random=env.np_random,
    )
    logger = RunLogger(cfg)
    tracker = SkillTracker(cfg.skill.n_skills)

    global_step = 0
    train_steps = 0
    best_mean_return = -float("inf")

    for episode in range(1, cfg.env.max_n_episodes + 1):
        obs, info = env.reset()
        ep_return = 0.0
        ep_length = 0
        skill = obs["skill"]
        t0 = time.time()

        while True:
            action = agent.act(obs["observation"], skill)
            next_obs, reward, terminated, truncated, info = env.step(action)

            buffer.add(
                next_obs["observation"],
                action,
                next_obs["skill"],
                terminated,
                next_obs["skill_start"],
            )

            ep_return += reward
            ep_length += 1
            global_step += 1
            obs = next_obs

            # one gradient step per environment step (standard SAC)
            if buffer.ready(cfg.sac.batch_size):
                metrics = agent.train_step(buffer)
                train_steps += 1

                if metrics is not None and train_steps % cfg.log_interval == 0:
                    logger.log_scalars(
                        global_step,
                        {
                            "losses/critic": metrics["loss_critic"],
                            "losses/actor": metrics["loss_pi"],
                            "losses/traj_disc": metrics["loss_traj"],
                            "losses/state_disc": metrics["loss_state"],
                            "losses/alpha": metrics["loss_alpha"],
                            "reward/intrinsic_mean": metrics["intrinsic_reward_mean"],
                            "reward/intrinsic_std": metrics["intrinsic_reward_std"],
                            "discriminator/traj_accuracy": metrics["traj_disc_accuracy"],
                            "discriminator/state_accuracy": metrics["state_disc_accuracy"],
                            "sac/alpha": metrics["alpha"],
                        },
                    )

            if terminated or truncated:
                break

        ep_time = time.time() - t0
        tracker.update(skill, ep_return, ep_length)

        # ---- episode-level logging ----
        ep_metrics: dict[str, float] = {
            "episode/return": ep_return,
            "episode/length": float(ep_length),
            "episode/steps_per_sec": ep_length / max(ep_time, 1e-6),
            "buffer/size": float(len(buffer)),
            "train/global_step": float(global_step),
            "train/train_steps": float(train_steps),
        }

        # ---- per-skill logging ----
        sk = skill
        ep_metrics[f"skills/skill_{sk}/running_mean_return"] = tracker.mean_return(sk)
        ep_metrics[f"skills/skill_{sk}/running_std_return"] = tracker.std_return(sk)
        ep_metrics[f"skills/skill_{sk}/running_mean_length"] = tracker.mean_length(sk)
        ep_metrics[f"skills/skill_{sk}/episode_count"] = float(tracker.counts[sk])

        # ---- aggregate skill diversity ----
        ep_metrics["skills/active_count"] = float(tracker.active_count())
        ep_metrics["skills/cross_skill_return_std"] = tracker.cross_skill_return_std()
        ep_metrics["skills/overall_mean_return"] = tracker.overall_mean_return()

        logger.log_scalars(global_step, ep_metrics)

        # ---- console summary ----
        if episode % cfg.log_interval == 0:
            print(
                f"[ep {episode:>5d} | step {global_step:>8d}] "
                f"ret={ep_return:8.2f}  len={ep_length:4d}  "
                f"skill={sk:3d}  skill_mean={tracker.mean_return(sk):8.2f}  "
                f"overall={tracker.overall_mean_return():8.2f}  "
                f"active={tracker.active_count()}/{cfg.skill.n_skills}  "
                f"buf={len(buffer)}"
            )

        # ---- checkpoint on new best overall mean return ----
        overall = tracker.overall_mean_return()
        if episode >= 10 and overall > best_mean_return:
            best_mean_return = overall
            logger.save_checkpoint(
                episode,
                agent.state_dicts(),
                extra={
                    "global_step": global_step,
                    "best_mean_return": best_mean_return,
                },
            )

    logger.close()
    env.close()
    print("Training complete.")


def main():
    cfg = get_train_config()
    train(cfg)


if __name__ == "__main__":
    main()
