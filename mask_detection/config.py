import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

DEFAULTS = {
    "model_path": str(BASE_DIR / "model" / "mask_detector.h5"),
    "host": "0.0.0.0",
    "port": 8000,
    "enable_frontend": False,
    "allowed_origins": ["*"],
    "images_dir": ""
}

def load_config():
    cfg_path_env = os.getenv("CONFIG_PATH")
    cfg_path = Path(cfg_path_env) if cfg_path_env else BASE_DIR / "config.json"
    cfg = DEFAULTS.copy()
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                cfg.update(data)
    model_env = os.getenv("MODEL_PATH")
    if model_env:
        cfg["model_path"] = model_env
    return cfg
