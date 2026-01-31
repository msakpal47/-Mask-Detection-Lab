import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from .config import BASE_DIR

def load_face_detector(face_dir: str):
    if face_dir:
        prototxtPath = os.path.join(face_dir, "deploy.prototxt")
        weightsPath = os.path.join(face_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.exists(prototxtPath) and os.path.exists(weightsPath):
            net = cv2.dnn.readNet(prototxtPath, weightsPath)
            return ("dnn", net)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return ("haar", cascade)

def detect_faces(image, detector):
    kind, model = detector
    h, w = image.shape[:2]
    boxes = []
    if kind == "dnn":
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        model.setInput(blob)
        detections = model.forward()
        for i in range(0, detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            boxes.append((startX, startY, endX, endY, confidence))
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, ww, hh) in faces:
            boxes.append((x, y, x + ww, y + hh, 1.0))
    return boxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-f", "--face", type=str, default="", help="path to face detector model directory (Caffe). Optional.")
    default_model = str(BASE_DIR / "model" / "mask_detector.h5")
    ap.add_argument("-m", "--model", type=str, default=default_model, help="path to trained mask detector (.h5)")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="min confidence for DNN detections")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        raise RuntimeError(f"Input image not found: {args.image}")
    if not os.path.isfile(args.model):
        raise RuntimeError(f"Mask model not found: {args.model}")

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError("Failed to read image")
    orig = image.copy()

    det = load_face_detector(args.face)
    boxes = detect_faces(image, det)

    model = load_model(args.model)

    for (startX, startY, endX, endY, conf) in boxes:
        face = image[startY:endY, startX:endX]
        if face.size == 0:
            continue
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        mask, withoutMask = model.predict(face)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label_text = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100.0)
        cv2.putText(image, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
