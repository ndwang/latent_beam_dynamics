"""Trainer for the LatentBeamTransformer."""

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import trajectory_mse_loss

if TYPE_CHECKING:
    from src.utils.logging import LoggingCallback


class Trainer:
    """Training loop for LatentBeamTransformer.

    Args:
        model: LatentBeamTransformer instance.
        optimizer: PyTorch optimizer.
        scheduler: Optional LR scheduler.
        device: Compute device.
        grad_clip: Max gradient norm (0 = disabled).
        ss_warmup: Scheduled-sampling warmup epochs (pure teacher forcing).
        ss_k: Scheduled-sampling ramp rate after warmup.
        logger_callback: Optional W&B / logging callback.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: Optional[torch.device] = None,
        grad_clip: float = 1.0,
        ss_warmup: int = 10,
        ss_k: float = 0.05,
        logger_callback: Optional["LoggingCallback"] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grad_clip = grad_clip
        self.ss_warmup = ss_warmup
        self.ss_k = ss_k

        if logger_callback is None:
            from src.utils.logging import NoOpCallback
            logger_callback = NoOpCallback()
        self.logger_callback = logger_callback

        self.best_val_loss = float("inf")
        self.start_epoch = 0
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}

        self.model.to(self.device)

    def _sampling_prob(self, epoch: int) -> float:
        if epoch < self.ss_warmup:
            return 0.0
        return min(1.0, (epoch - self.ss_warmup) * self.ss_k)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        self.start_epoch = ckpt.get("epoch", 0)
        print(f"Resuming from epoch {self.start_epoch}, val_loss={self.best_val_loss:.6f}")

    def train_epoch(
        self, train_loader: DataLoader, epoch: int, max_steps: Optional[int] = None
    ) -> float:
        self.model.train()
        sampling_prob = self._sampling_prob(epoch)
        total_loss = 0.0
        n_samples = 0

        loop = tqdm(train_loader, desc="Training", leave=False)
        for step, (z0, elements, z_gt) in enumerate(loop):
            if max_steps is not None and step >= max_steps:
                break

            z0 = z0.to(self.device)
            elements = elements.to(self.device)
            z_gt = z_gt.to(self.device)

            self.optimizer.zero_grad()
            z_pred = self.model(z0, elements, z_gt=z_gt, sampling_prob=sampling_prob)
            loss = trajectory_mse_loss(z_pred, z_gt)

            if torch.isnan(loss):
                raise ValueError(f"NaN loss at step {step}")

            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            batch_size = z0.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            loop.set_postfix(loss=f"{loss.item():.4f}", sp=f"{sampling_prob:.2f}")

        return total_loss / n_samples

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        for z0, elements, z_gt in val_loader:
            z0 = z0.to(self.device)
            elements = elements.to(self.device)
            z_gt = z_gt.to(self.device)

            # Validation always uses teacher forcing for a stable metric
            z_pred = self.model(z0, elements, z_gt=z_gt, sampling_prob=0.0)
            loss = trajectory_mse_loss(z_pred, z_gt)

            batch_size = z0.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        return total_loss / n_samples

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        max_steps: Optional[int] = None,
        save_dir: Optional[Path] = None,
        model_name: str = "lbd",
        checkpoint_freq: int = 50,
    ) -> Dict[str, list]:
        epoch_bar = tqdm(range(self.start_epoch, epochs), desc="Epochs", unit="epoch")
        for epoch in epoch_bar:
            train_loss = self.train_epoch(train_loader, epoch, max_steps)
            val_loss = self.validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            sampling_prob = self._sampling_prob(epoch)
            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                lr=f"{current_lr:.1e}",
                sp=f"{sampling_prob:.2f}",
            )

            self.logger_callback.log_metrics(
                {
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "learning_rate": current_lr,
                    "sampling_prob": sampling_prob,
                },
                step=epoch + 1,
            )

            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(
                        save_dir / f"{model_name}_best.pth", epoch + 1, train_loss, val_loss
                    )

                if (epoch + 1) % checkpoint_freq == 0:
                    self._save_checkpoint(
                        save_dir / f"{model_name}_epoch{epoch + 1}.pth",
                        epoch + 1, train_loss, val_loss,
                    )

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            model_path = save_dir / f"{model_name}.pth"
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved: {model_path}")

            history_path = save_dir / f"{model_name}_history.csv"
            self._save_history(history_path)
            print(f"History saved: {history_path}")

        return self.history

    def _save_checkpoint(
        self, path: Path, epoch: int, train_loss: float, val_loss: float
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            path,
        )
        tqdm.write(f"Checkpoint saved: {path}")

    def _save_history(self, path: Path) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            for i, (tl, vl) in enumerate(
                zip(self.history["train_loss"], self.history["val_loss"])
            ):
                writer.writerow([self.start_epoch + i + 1, tl, vl])
