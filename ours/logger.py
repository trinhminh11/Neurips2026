"""TensorBoard + optional wandb logging and checkpoints."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig, train_config_to_dict

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


class RunLogger:
    def __init__(self, cfg: TrainConfig, log_root: str = "logs") -> None:
        self.cfg = cfg
        self.base = Path(log_root)
        stamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        env_short = cfg.env.env_name.replace("/", "_").rsplit("-", 1)[0]
        self.run_dir = self.base / f"{env_short}_{cfg.skill.reward_type}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir

        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        flat = train_config_to_dict(cfg)
        for k, v in flat.items():
            self.writer.add_text(f"config/{k}", str(v), 0)

        self._wandb = None
        if cfg.use_wandb and wandb is not None:
            name = cfg.run_name or str(self.run_dir.name)
            self._wandb = wandb.init(project=cfg.wandb_project, name=name, config=flat)
        elif cfg.use_wandb and wandb is None:
            print("wandb not installed; continuing with TensorBoard only.")


    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, step)
                if self._wandb is not None:
                    wandb.log({k: v}, step=step)


    def close(self) -> None:
        self.writer.close()
        if self._wandb is not None:
            wandb.finish()


    def save_checkpoint(self, episode: int, agent_state: dict, extra: dict[str, Any] | None = None) -> Path:
        payload = {"episode": episode, "agent": agent_state}
        if extra:
            payload.update(extra)
        path = self.ckpt_dir / "checkpoint.pt"
        torch.save(payload, path)
        return path


    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        return torch.load(path, map_location="cpu", weights_only=False)
