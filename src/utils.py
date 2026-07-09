"""Shared helpers: deterministic seeding, config loading, optional offline W&B."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_smoke_overrides(config: dict[str, Any], smoke: bool) -> dict[str, Any]:
    """--smoke forces the mock judge and a tiny sample regardless of the passed config file."""
    if not smoke:
        return config
    config = dict(config)
    config["dataset"] = dict(config["dataset"])
    config["dataset"]["max_queries"] = min(config["dataset"].get("max_queries") or 5, 5)
    config["dataset"]["candidate_pool_size"] = min(config["dataset"].get("candidate_pool_size", 5), 5)
    config["judge"] = dict(config["judge"])
    config["judge"]["offline_mock"] = True
    config["seeds"] = config.get("seeds", [config.get("seed", 0)])[:1]
    return config


class WandbLogger:
    """Thin optional/offline-capable W&B wrapper. No-ops if disabled or wandb import fails."""

    def __init__(self, config: dict[str, Any], run_name: str):
        wandb_cfg = config.get("wandb", {})
        self.enabled = wandb_cfg.get("enabled", False)
        self._run = None
        if self.enabled:
            try:
                import wandb

                os.environ.setdefault("WANDB_MODE", wandb_cfg.get("mode", "offline"))
                self._run = wandb.init(
                    project=wandb_cfg.get("project", "zelo-curriculum-study"),
                    name=run_name,
                    config=config,
                )
            except Exception as exc:  # pragma: no cover - defensive, wandb is optional
                print(f"[WandbLogger] disabling W&B logging, init failed: {exc}")
                self.enabled = False

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._run.finish()
