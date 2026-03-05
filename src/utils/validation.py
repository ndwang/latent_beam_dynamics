"""Configuration validation using Pydantic models."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Validation schema for model configuration."""

    model_config = {"extra": "forbid"}

    latent_dim: int = Field(default=64, ge=1)
    d_model: int = Field(default=256, ge=1)
    n_layers: int = Field(default=6, ge=1)
    n_heads: int = Field(default=8, ge=1)
    n_freq: int = Field(default=32, ge=1)
    element_dim: int = Field(default=7, ge=1)
    lambda_min: float = Field(default=0.01, gt=0.0)
    lambda_max: float = Field(default=1000.0, gt=0.0)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    mlp_ratio: int = Field(default=4, ge=1)

    @field_validator("d_model")
    @classmethod
    def d_model_divisible_by_n_heads(cls, v: int, info) -> int:
        n_heads = info.data.get("n_heads")
        if n_heads and v % n_heads != 0:
            raise ValueError(f"d_model={v} must be divisible by n_heads={n_heads}")
        return v

    @field_validator("lambda_max")
    @classmethod
    def lambda_max_gt_min(cls, v: float, info) -> float:
        lambda_min = info.data.get("lambda_min")
        if lambda_min is not None and v <= lambda_min:
            raise ValueError(f"lambda_max={v} must be greater than lambda_min={lambda_min}")
        return v


class SchedulerConfig(BaseModel):
    """Validation schema for learning rate scheduler configuration."""

    model_config = {"extra": "forbid"}

    name: Literal["cosine", "reduce_on_plateau", "none"] = "cosine"
    factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    patience: int = Field(default=10, ge=1)


class WandbConfig(BaseModel):
    """Validation schema for Weights & Biases configuration."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    project: str = "latent-beam-dynamics"
    entity: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    offline: bool = True


class TrainingConfig(BaseModel):
    """Validation schema for training configuration."""

    model_config = {"extra": "forbid"}

    epochs: int = Field(default=200, ge=1)
    batch_size: int = Field(default=32, ge=1)
    lr: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    val_split: float = Field(default=0.1, gt=0.0, lt=1.0)
    seed: int = 42
    num_workers: int = Field(default=8, ge=0)
    checkpoint_freq: int = Field(default=50, ge=1)
    max_steps: Optional[int] = Field(default=None, ge=1)
    ss_warmup: int = Field(default=10, ge=0)
    ss_k: float = Field(default=0.05, gt=0.0)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class DataConfig(BaseModel):
    """Validation schema for data configuration."""

    model_config = {"extra": "forbid"}

    path: str

    @field_validator("path")
    @classmethod
    def check_path_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("Data path cannot be empty")
        return v


class Config(BaseModel):
    """Top-level configuration validation schema."""

    model_config = {"extra": "forbid"}

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig
    run_name: Optional[str] = None
    output_dir: str = "./runs"


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[Dict[str, Any]]):
        self.errors = errors
        messages = []
        for err in errors:
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "Unknown error")
            messages.append(f"  {loc}: {msg}")
        super().__init__("Configuration validation failed:\n" + "\n".join(messages))


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration dictionary against schema.

    Returns validated and normalized config dict.
    Raises ConfigValidationError on invalid config.
    """
    try:
        validated = Config(**config)
        return validated.model_dump()
    except Exception as e:
        if hasattr(e, "errors"):
            raise ConfigValidationError(e.errors())
        raise ConfigValidationError([{"loc": [], "msg": str(e)}])
