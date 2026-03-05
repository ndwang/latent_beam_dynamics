#!/usr/bin/env python
"""Training script for LatentBeamTransformer.

Usage:
    # Default config
    python scripts/train.py

    # Override hyperparameters
    python scripts/train.py model.d_model=512 training.epochs=300

    # Specify data path
    python scripts/train.py data.path=/path/to/trajectories.npz

    # Resume from checkpoint
    python scripts/train.py --resume runs/my_run/lbd_best.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, random_split

from src.model import ModelConfig, LatentBeamTransformer
from src.data import LatentTrajectoryDataset
from src.training import Trainer
from src.utils import load_config, save_config, generate_run_name, init_wandb


def get_args():
    parser = argparse.ArgumentParser(description="Train LatentBeamTransformer")
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--resume", type=str, default=None)
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def main():
    args = get_args()

    config = load_config(
        config_path=args.config,
        config_dir=args.config_dir,
        overrides=args.overrides,
    )

    model_cfg_dict = config.get("model", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Reproducibility
    seed = training_cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Dataset
    data_path = data_cfg.get("path")
    if not data_path:
        raise ValueError("data.path must be set (e.g. data.path=/path/to/data_dir)")

    full_dataset = LatentTrajectoryDataset(data_path)
    n_samples = len(full_dataset)
    val_split = training_cfg.get("val_split", 0.1)
    val_size = int(val_split * n_samples)
    train_size = n_samples - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Dataset: {n_samples} samples ({train_size} train, {val_size} val)")

    num_workers = training_cfg.get("num_workers", 8)
    batch_size = training_cfg.get("batch_size", 32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # Model
    model_config = ModelConfig(**model_cfg_dict)
    model = LatentBeamTransformer(model_config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get("lr", 3e-4),
        weight_decay=training_cfg.get("weight_decay", 1e-2),
    )

    # Scheduler
    scheduler_cfg = training_cfg.get("scheduler", {})
    scheduler_name = scheduler_cfg.get("name", "cosine")
    epochs = training_cfg.get("epochs", 200)

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 10),
        )
    else:
        scheduler = None

    # Output dir
    run_name = config.get("run_name") or generate_run_name(config)
    output_dir = Path(config.get("output_dir", "./runs")) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    _, logger_callback = init_wandb(config, run_name, output_dir)

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        grad_clip=training_cfg.get("grad_clip", 1.0),
        ss_warmup=training_cfg.get("ss_warmup", 10),
        ss_k=training_cfg.get("ss_k", 0.05),
        logger_callback=logger_callback,
    )

    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    save_config(config, output_dir / "config.yaml")
    print(f"Run: {run_name}")
    print(f"Output: {output_dir}")

    try:
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            max_steps=training_cfg.get("max_steps"),
            save_dir=output_dir,
            model_name=run_name,
            checkpoint_freq=training_cfg.get("checkpoint_freq", 50),
        )
        print("Training complete!")
    finally:
        logger_callback.finish()


if __name__ == "__main__":
    main()
