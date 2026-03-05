"""Logging callbacks for training metrics."""

from abc import ABC, abstractmethod
from typing import Dict


class LoggingCallback(ABC):
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass

    @abstractmethod
    def finish(self) -> None:
        pass


class NoOpCallback(LoggingCallback):
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbCallback(LoggingCallback):
    def __init__(self, run):
        self.run = run

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self.run.log(metrics, step=step)

    def finish(self) -> None:
        self.run.finish()
