import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from .config import load_config, BASE_DIR

CFG = load_config()
_mp = Path(CFG.get("model_path", ""))
MODEL_PATH = _mp if _mp.is_absolute() else (BASE_DIR / _mp)
model = load_model(str(MODEL_PATH))

LABELS = ["With Mask", "Without Mask"]
IMG_SIZE = 224

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            continue
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = preprocess_input(face.astype("float32"))
        face = np.expand_dims(face, axis=0)
        scores = model.predict(face)[0]
        idx = int(np.argmax(scores))
        label = LABELS[idx] if idx < len(LABELS) else f"class_{idx}"
        conf = float(scores[idx])
        color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {conf*100:.1f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
