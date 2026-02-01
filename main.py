from pathlib import Path

import yaml

from model import ModelConfig, LatentBeamTransformer


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    model_cfg = ModelConfig(**cfg["model"])
    model = LatentBeamTransformer(model_cfg)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model config: {model_cfg}")
    print(f"Parameters:   {param_count:,}")
    print(f"Training config: {cfg['training']}")


if __name__ == "__main__":
    main()
