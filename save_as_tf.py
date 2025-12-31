import tensorflow as tf
from pathlib import Path
from mask_detection.config import load_config, BASE_DIR

CFG = load_config()
_mp = Path(CFG.get("model_path", ""))
MODEL_PATH = _mp if _mp.is_absolute() else (BASE_DIR / _mp)

model = tf.keras.models.load_model(str(MODEL_PATH))
save_path = "mask_detection_saved_model"
tf.saved_model.save(model, save_path)
print(f"Saved as SavedModel to {save_path}")
