import argparse
from pathlib import Path
from tensorflow.keras.models import load_model
import tf2onnx
import onnx
from .config import BASE_DIR

def model2onnx():
    ap = argparse.ArgumentParser()
    default_model = str(BASE_DIR / "model" / "mask_detector.h5")
    default_out = str(BASE_DIR / "model" / "mask_detector.onnx")
    ap.add_argument("-m", "--model", type=str, default=default_model)
    ap.add_argument("-o", "--output", type=str, default=default_out)
    args = ap.parse_args()

    model = load_model(args.model)
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "?"
    onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "?"
    onnx.save(onnx_model, args.output)

if __name__ == "__main__":
    model2onnx()
