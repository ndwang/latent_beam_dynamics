"""W&B initialization utility."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .logging import LoggingCallback, NoOpCallback, WandbCallback


def init_wandb(
    config: Dict[str, Any],
    run_name: str,
    output_dir: Path,
) -> Tuple[Optional[Any], LoggingCallback]:
    """Initialize Weights & Biases if enabled.

    Returns:
        Tuple of (wandb.Run or None, LoggingCallback).
    """
    wandb_cfg = config.get("training", {}).get("wandb", {})

    if not wandb_cfg.get("enabled", False):
        return None, NoOpCallback()

    try:
        import wandb
    except ImportError:
        print("Warning: wandb not installed. Logging disabled.")
        return None, NoOpCallback()

    if wandb_cfg.get("offline", True):
        os.environ["WANDB_MODE"] = "offline"

    try:
        run = wandb.init(
            project=wandb_cfg.get("project", "latent-beam-dynamics"),
            entity=wandb_cfg.get("entity"),
            name=run_name,
            config=config,
            dir=str(output_dir),
            tags=wandb_cfg.get("tags", []),
            notes=wandb_cfg.get("notes"),
            reinit=True,
        )
        callback = WandbCallback(run)
        mode = "offline" if wandb_cfg.get("offline", True) else "online"
        print(f"W&B initialized ({mode} mode): {run.name}")
        return run, callback
    except Exception as e:
        print(f"Warning: Failed to initialize W&B: {e}")
        return None, NoOpCallback()
