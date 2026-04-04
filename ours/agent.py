"""SAC agent with pluggable trajectory-centric intrinsic rewards."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from buffer import ReplayBuffer
from config import TrainConfig
from models import (
    CriticNetwork,
    MLPDiscriminator,
    OneStateDiscriminator,
    PolicyNetwork,
)
from torch import Tensor, from_numpy
from torch.optim import Adam

log_state_z_coef = 1.0
log_state_mean_coef = 1.0
p_z_coef = 1.0


class Agent:
    def __init__(
        self,
        cfg: TrainConfig,
        n_states: int,
        n_actions: int,
        action_scale: float = 1,
        action_bias: float = 0,
    ) -> None:
        self.cfg = cfg
        self.n_states = n_states
        self.n_skills = cfg.skill.n_skills
        self.skill_embed_dim = cfg.skill.skill_embed_dim
        self.n_actions = n_actions
        self.policy_obs_dim = n_states + self.skill_embed_dim
        sac = cfg.sac
        self.target_update_interval = sac.target_update_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(sac.seed)

        nh = sac.n_hiddens

        self.skill_embedding = nn.Embedding(self.n_skills, self.skill_embed_dim).to(
            self.device
        )
        self.skill_embedding.weight.data.normal_(0, 0.02)
        self.skill_embedding.weight.data.clamp_(-0.04, 0.04)

        self.policy = PolicyNetwork(
            self.policy_obs_dim,
            n_actions,
            action_scale,
            action_bias,
            nh,
        ).to(self.device)

        self.critic1 = CriticNetwork(self.policy_obs_dim, n_actions, nh).to(self.device)
        self.critic2 = CriticNetwork(self.policy_obs_dim, n_actions, nh).to(self.device)

        self.critic1_target = CriticNetwork(self.policy_obs_dim, n_actions, nh).to(
            self.device
        )
        self.critic2_target = CriticNetwork(self.policy_obs_dim, n_actions, nh).to(
            self.device
        )

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_target.eval()
        self.critic2_target.eval()
        for p in self.critic1_target.parameters():
            p.requires_grad_(False)
        for p in self.critic2_target.parameters():
            p.requires_grad_(False)

        self.state_disc = OneStateDiscriminator(n_states, self.n_skills, nh).to(
            self.device
        )
        self.traj_disc = MLPDiscriminator(
            cfg.skill.traj_len, n_states, self.n_skills, nh
        ).to(self.device)

        ## ===== NEED UPDATE =====
        # self.reward_module: IntrinsicReward = build_reward_module(
        #     cfg.skill.reward_type,  # type: ignore[arg-type]
        #     self.state_disc,
        #     self.traj_disc,
        #     self.n_skills,
        #     cfg.skill.n_negatives,
        #     self.device,
        # )
        ## ===== NEED UPDATE =====

        if sac.learning_alpha:
            self.log_alpha = nn.Parameter(
                torch.log(torch.tensor(sac.alpha, device=self.device)),
                requires_grad=True,
            )
        else:
            self.log_alpha = torch.tensor(
                np.log(sac.alpha), device=self.device, requires_grad=False
            )

        self.skill_embedding_opt = Adam(
            self.skill_embedding.parameters(), lr=sac.actor_lr
        )  # TODO: add skill embedding learning rate

        self.policy_opt = Adam(self.policy.parameters(), lr=sac.actor_lr)

        self.critic_opt = Adam(
            [*self.critic1.parameters(), *self.critic2.parameters()], lr=sac.critic_lr
        )

        self.discriminator_opt = Adam(
            [*self.state_disc.parameters(), *self.traj_disc.parameters()],
            lr=sac.discriminator_lr,
        )

        if sac.learning_alpha:
            self.alpha_opt = Adam([self.log_alpha], lr=sac.alpha_lr)
        else:
            self.alpha_opt = None

        # uniform entropy of n skills
        self.p_z = torch.log(torch.tensor(float(self.n_skills), device=self.device))

        self.time_step = 0

        self.target_entropy = -self.n_skills

    def act(self, obs_np: np.ndarray, skill: int) -> np.ndarray:
        obs = from_numpy(obs_np).float().unsqueeze(0).to(self.device)
        skill_emb: Tensor = self.skill_embedding(
            torch.tensor([skill], device=self.device).long()
        )
        obs_aug = torch.cat([obs, skill_emb], dim=-1)
        action, _ = self.policy.sample_and_log_prob(obs_aug)
        return action.squeeze(0).detach().cpu().numpy()

    def _soft_update_target(self) -> None:
        tau = self.cfg.sac.tau
        with torch.no_grad():
            self.critic1_target.soft_update(self.critic1, tau)
            self.critic2_target.soft_update(self.critic2, tau)

    def train_step(self, memory: ReplayBuffer) -> dict[str, float] | None:
        """One SAC + discriminator update."""
        self.time_step += 1

        sac = self.cfg.sac

        batch = memory.sample(sac.batch_size, self.cfg.skill.traj_len)

        if batch is None:
            return None

        # B, [St, St+1, ..., St+k]
        traj = from_numpy(batch.trajectory).float().to(self.device)
        # B, At (action at St)
        actions = from_numpy(batch.actions).float().to(self.device)
        skills = from_numpy(batch.skills).long().to(self.device)  # B,
        skills_embed = self.skill_embedding(skills)  # B, skill_embed_dim
        dones = from_numpy(batch.dones).reshape(-1, 1).float().to(self.device)

        St = traj[:, 0, :]  # St
        Stk = traj[:, 0:-1, :]  # Stk = [St, St+1, ..., St+k-1]
        St1 = traj[:, 1, :]  # St1 = St+1
        St1k1 = traj[:, 1:, :]  # St1k1 = [St+1, St+2, ..., St+k]

        current_alpha = self.log_alpha.exp().detach()

        obs_and_skill = torch.cat([St, skills_embed], dim=-1)
        next_obs_and_skill = torch.cat(
            [St1, skills_embed], dim=-1
        )  # because next skill is the same as current skill
        # action: B, n_actions
        # log_prob: B, 1

        with torch.no_grad():
            target_actions, target_log_probs = self.policy.sample_and_log_prob(
                next_obs_and_skill
            )
            critic1_target: Tensor = self.critic1_target(
                next_obs_and_skill, target_actions
            )
            critic2_target: Tensor = self.critic2_target(
                next_obs_and_skill, target_actions
            )

            traj_logits = self.traj_disc(St1k1)  # B, n_skills
            log_traj = F.log_softmax(traj_logits, dim=-1)  # B, n_skills
            log_traj_z = log_traj.gather(-1, skills.unsqueeze(-1))  # B, 1

            state_logits = self.state_disc(St1)  # B, n_skills
            log_state = F.log_softmax(state_logits, dim=-1)  # B, n_skills
            log_state_z = log_state.gather(-1, skills.unsqueeze(-1))  # B, 1

            log_state_mean = log_state.mean(dim=-1).unsqueeze(-1)  # B, 1

            # B, 1
            intrinsic_reward = (
                log_traj_z
                - log_state_z_coef * log_state_z
                + log_state_mean_coef * log_state_mean
                - p_z_coef * self.p_z
            )
            _intrinsic_reward_mean = intrinsic_reward.mean().item()
            _intrinsic_reward_std = intrinsic_reward.std().item()

            # B, 1
            y_target = intrinsic_reward + sac.gamma * (1.0 - dones) * (
                torch.min(critic1_target, critic2_target)
                - current_alpha * target_log_probs
            )

        # critic loss
        loss_critic = F.mse_loss(
            self.critic1(obs_and_skill, actions), y_target
        ) + F.mse_loss(self.critic2(obs_and_skill, actions), y_target)

        # actor loss
        actions_pi, logp_pi = self.policy.sample_and_log_prob(obs_and_skill)
        critic1_pi: Tensor = self.critic1(obs_and_skill, actions_pi)
        critic2_pi: Tensor = self.critic2(obs_and_skill, actions_pi)
        critic_pi = torch.min(critic1_pi, critic2_pi)
        loss_pi = (current_alpha * logp_pi - critic_pi).mean()

        # actor and critic update
        self.policy_opt.zero_grad()
        self.critic_opt.zero_grad()
        self.skill_embedding_opt.zero_grad()

        (loss_critic + loss_pi).backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1)

        self.policy_opt.step()
        self.critic_opt.step()
        self.skill_embedding_opt.step()

        # alpha update
        _loss_alpha = 0.0
        if sac.learning_alpha:
            loss_alpha = -(
                self.log_alpha.exp() * (logp_pi + self.target_entropy).detach()
            ).mean()
            _loss_alpha = loss_alpha.item()
            self.alpha_opt.zero_grad()
            loss_alpha.backward()
            self.alpha_opt.step()

        # discriminator update
        ## trajectory discriminator loss
        traj_logits = self.traj_disc(Stk)  # B, n_skills
        loss_traj = F.cross_entropy(traj_logits, skills)

        ## state discriminator loss
        state_logits = self.state_disc(St1)  # B, n_skills
        loss_state = F.cross_entropy(state_logits, skills)

        ## discriminator update
        self.discriminator_opt.zero_grad()
        (loss_traj + loss_state).backward()
        self.discriminator_opt.step()

        with torch.no_grad():
            traj_acc = (traj_logits.argmax(-1) == skills).float().mean().item()
            state_acc = (state_logits.argmax(-1) == skills).float().mean().item()

        # target update
        if self.time_step % self.target_update_interval == 0:
            self._soft_update_target()

        return {
            "loss_critic": loss_critic.item(),
            "loss_pi": loss_pi.item(),
            "loss_traj": loss_traj.item(),
            "loss_state": loss_state.item(),
            "loss_alpha": _loss_alpha,
            "alpha": current_alpha.item(),
            "intrinsic_reward_mean": _intrinsic_reward_mean,
            "intrinsic_reward_std": _intrinsic_reward_std,
            "traj_disc_accuracy": traj_acc,
            "state_disc_accuracy": state_acc,
        }

    def state_dicts(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "state_disc": self.state_disc.state_dict(),
            "traj_disc": self.traj_disc.state_dict(),
            "skill_embedding": self.skill_embedding.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "policy_opt": self.policy_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "skill_embedding_opt": self.skill_embedding_opt.state_dict(),
            "discriminator_opt": self.discriminator_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict()
            if self.cfg.sac.learning_alpha
            else None,
        }

    def load_state_dicts(self, ckpt: dict) -> None:
        self.policy.load_state_dict(ckpt["policy"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
        self.state_disc.load_state_dict(ckpt["state_disc"])
        self.traj_disc.load_state_dict(ckpt["traj_disc"])
        self.skill_embedding.load_state_dict(ckpt["skill_embedding"])
        self.policy_opt.load_state_dict(ckpt["policy_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.skill_embedding_opt.load_state_dict(ckpt["skill_embedding_opt"])
        self.discriminator_opt.load_state_dict(ckpt["discriminator_opt"])

        if self.cfg.sac.learning_alpha:
            self.log_alpha.data = nn.Parameter(
                torch.tensor(ckpt["log_alpha"], device=self.device, requires_grad=True)
            )
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        else:
            self.log_alpha.data = torch.tensor(
                ckpt["log_alpha"], device=self.device, requires_grad=False
            )
            self.alpha_opt = None
