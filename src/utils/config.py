"""Configuration loading and management utilities."""

import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict, override: Dict) -> Dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_override(override_str: str) -> tuple[List[str], Any]:
    """Parse 'model.d_model=256' into (['model', 'd_model'], 256)."""
    if "=" not in override_str:
        raise ValueError(f"Override must be 'key=value', got: {override_str}")
    key, value_str = override_str.split("=", 1)
    key_path = key.split(".")
    try:
        value = yaml.safe_load(value_str)
    except yaml.YAMLError:
        value = value_str
    return key_path, value


def apply_override(config: Dict, key_path: List[str], value: Any) -> None:
    current = config
    for key in key_path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[key_path[-1]] = value


def apply_overrides(config: Dict, overrides: List[str]) -> Dict:
    config = copy.deepcopy(config)
    for override_str in overrides:
        key_path, value = parse_override(override_str)
        apply_override(config, key_path, value)
    return config


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    config_dir: Union[str, Path] = "configs",
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load and compose config from YAML files with optional CLI overrides.

    The main config may reference sub-configs by path string, e.g.:
        training: training/default.yaml
    """
    config_dir = Path(config_dir)

    if config_path is None:
        config_path = config_dir / "default.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = config_dir / config_path

    config = load_yaml(config_path)

    composed: Dict[str, Any] = {}
    for key in ["model", "training", "data"]:
        if key in config and isinstance(config[key], str):
            sub_path = config_dir / config[key]
            composed[key] = load_yaml(sub_path)
        elif key in config and isinstance(config[key], dict):
            composed[key] = config[key]

    for key, value in config.items():
        if key not in ["model", "training", "data"]:
            composed[key] = value

    if overrides:
        composed = apply_overrides(composed, overrides)

    from .validation import validate_config
    composed = validate_config(composed)

    return composed


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def generate_run_name(config: Dict[str, Any]) -> str:
    """Generate run name from key params + timestamp.

    Format: lbd{d_model}_L{n_layers}_{YYMMDD}_{HHMM}
    """
    d_model = config.get("model", {}).get("d_model", 0)
    n_layers = config.get("model", {}).get("n_layers", 0)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    return f"lbd_d{d_model}_L{n_layers}_{timestamp}"
