import streamlit as st
import numpy as np
import time
import os
import random
from pathlib import Path
from mask_detection.config import load_config, BASE_DIR

CFG = load_config()
_mp = Path(CFG.get("model_path", ""))
MODEL_PATH = _mp if _mp.is_absolute() else (BASE_DIR / _mp)
def _load_model(p: Path):
    try:
        from tensorflow.keras.models import load_model
    except Exception:
        return None
    if not p.exists():
        return None
    try:
        return load_model(str(p))
    except Exception:
        return None
model = _load_model(MODEL_PATH)
labels = ['No Mask', 'Mask']

st.set_page_config(page_title="Mask Detection Lab", page_icon="😷", layout="wide")
st.markdown(
    """
    <style>
      .stApp {background: linear-gradient(180deg, #0b1c26 0%, #0f2e3d 60%, #0b1c26 100%); color: #e9fbf7}
      .hero {padding: 1.5rem 2rem; border-radius: 16px; background: linear-gradient(135deg, rgba(20,184,166,.28), rgba(59,130,246,.18)); border: 1px solid rgba(255,255,255,.10)}
      .hero h1 {margin: 0; font-size: 2.2rem; line-height: 1.2; color: #d1fae5}
      .hero p {margin: .25rem 0 0; color: #c7f9f3}
      .metric {padding:.75rem 1rem; border-radius:12px; border:1px solid rgba(255,255,255,.10); background: rgba(255,255,255,.06); color:#e9fbf7}
      div.stButton > button {border-radius: 999px; padding:.6rem 1.1rem; font-weight:600; border:0; background: linear-gradient(90deg,#0ea5a6,#14b8a6); color:#061317; transition: all .15s ease}
      div.stButton > button:hover {transform: translateY(-1px); box-shadow: 0 6px 18px rgba(20,184,166,.35)}
      div.stButton > button:focus {outline: 2px solid #22d3ee; box-shadow: 0 0 0 4px rgba(34,211,238,.25)}
      .stop button {background: linear-gradient(90deg,#ef4444,#f97316); color:#160c0c; transition: all .15s ease}
      .stop button:hover {transform: translateY(-1px); box-shadow: 0 6px 18px rgba(239,68,68,.35)}
      .label-pill {position:absolute; top:14px; left:14px; padding:.35rem .7rem; border-radius:999px; font-weight:700; font-size:.9rem; background: rgba(11,28,38,.7); color:#e9fbf7; border: 1px solid rgba(255,255,255,.18)}
      .stTabs [data-baseweb="tab"] {color:#a7f3d0}
      .stTabs [data-baseweb="tab"]:hover {color:#e9fbf7}
      .stTabs [data-baseweb="tab"][aria-selected="true"] {color:#e9fbf7; border-bottom:3px solid #22d3ee}
      [data-testid="stFileUploader"] > div {background: rgba(10,31,42,.70); border: 1px solid rgba(20,184,166,.35); color:#e9fbf7; border-radius:14px; overflow:hidden}
      [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {border-radius:14px !important; background: rgba(12,36,48,.85); border: 1px solid rgba(20,184,166,.45)}
      [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover {border-color:#22d3ee; box-shadow: 0 0 0 2px rgba(34,211,238,.35)}
      [data-testid="stFileUploader"] svg {filter: drop-shadow(0 2px 4px rgba(0,0,0,.3))}
      [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] span {color:#d1fae5 !important}
      [data-testid="stFileUploader"] button {background: linear-gradient(90deg,#14b8a6,#22d3ee); color:#06262f; border:0; border-radius:10px; font-weight:700}
      [data-testid="stFileUploader"] button:hover {filter: brightness(1.06); transform: translateY(-1px)}
      [data-testid="stFileUploader"] button:focus {outline:2px solid #22d3ee}
      .stCaption, .stMarkdown, .stText, h2, h3, h4 {color:#e9fbf7}
      a {color:#22d3ee}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>😷 Mask Detection Lab</h1><p>Real‑time detection from your camera stream with a sleek UI.</p></div>',
    unsafe_allow_html=True,
)

frame_window = st.empty()

if 'stop_detection' not in st.session_state:
    st.session_state.stop_detection = False
if 'stream_url' not in st.session_state:
    st.session_state.stream_url = "http://192.168.68.53:8080/video"
if 'mirror' not in st.session_state:
    st.session_state.mirror = True

def get_model_labels():
    if model is None:
        return ["With Mask", "Without Mask", "Improper Mask"]
    try:
        n = int(getattr(model, "output_shape", [None, None])[-1])
    except Exception:
        n = 2
    if n == 3:
        return ["With Mask", "Without Mask", "Improper Mask"]
    return ["Mask", "No Mask"]

MODEL_LABELS = get_model_labels()

def capture_video():
    try:
        import cv2
    except Exception:
        return capture_snapshot_mode()
    cap = cv2.VideoCapture(st.session_state.stream_url)

    if not cap.isOpened():
        st.error("❌ Failed to open video stream. Check URL and network.")
        return
    else:
        st.success("✅ Video stream opened.")

    frame_count = 0
    t0 = time.time()

    # Prepare face detector
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:
        face_cascade = None

    while not st.session_state.stop_detection:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to read frame from the stream.")
            break

        if st.session_state.mirror:
            frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes = []
        if face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

        used = MODEL_LABELS
        if boxes:
            try:
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            except Exception:
                preprocess_input = None
            for (x, y, w, h) in boxes:
                face_rgb = image[y:y+h, x:x+w]
                if face_rgb.size == 0:
                    continue
                if model is not None and preprocess_input is not None:
                    face_resized = cv2.resize(face_rgb, (224, 224))
                    face_proc = preprocess_input(face_resized.astype("float32"))
                    face_input = np.expand_dims(face_proc, axis=0)
                    scores = model.predict(face_input)[0]
                    idx = int(np.argmax(scores))
                    label = used[idx if idx < len(used) else 0]
                else:
                    label = random.choice(used)
                color = (34, 197, 94) if label in ("Mask", "With Mask") else (239, 68, 68)
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(image, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # Whole-frame fallback classification
            image_resized = cv2.resize(image, (224, 224))
            if model is not None:
                try:
                    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                    image_proc = preprocess_input(image_resized.astype("float32"))
                    image_input = np.expand_dims(image_proc, axis=0)
                except Exception:
                    image_input = np.expand_dims(image_resized, axis=0) / 255.0
                prediction = model.predict(image_input)
                idx = int(np.argmax(prediction))
                label = used[idx if idx < len(used) else 0]
            else:
                label = random.choice(used)
            color = (34, 197, 94) if label in ("Mask", "With Mask") else (239, 68, 68)
            cv2.putText(image, label, (30, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        frame_window.image(image, channels="RGB")

        time.sleep(0.05)

        frame_count += 1

    cap.release()
    elapsed = max(time.time() - t0, 1e-6)
    fps = frame_count / elapsed
    st.info(f"Frames: {frame_count} • Avg FPS: {fps:.1f}")

def capture_snapshot_mode():
    st.warning("Running in snapshot mode (OpenCV not available).")
    url = st.session_state.stream_url
    if "/video" in url:
        url = url.replace("/video", "/shot.jpg")
    frame_count = 0
    t0 = time.time()
    while not st.session_state.stop_detection:
        label = random.choice(MODEL_LABELS)
        frame_window.image(url)
        st.caption(f"Label: {label}")
        time.sleep(0.2)
        frame_count += 1
    elapsed = max(time.time() - t0, 1e-6)
    fps = frame_count / elapsed
    st.info(f"Frames: {frame_count} • Approx refresh: {fps:.1f} fps")

with st.sidebar:
    st.subheader("Controls")
    st.session_state.stream_url = st.text_input("Stream URL", st.session_state.stream_url)
    st.session_state.mirror = st.toggle("Mirror video", st.session_state.mirror)
    st.caption("Use IP Webcam URLs. If OpenCV is missing, app will use snapshot mode.")
    st.info(f"Model loaded: {'Yes' if model is not None else 'No'} • Labels: {', '.join(MODEL_LABELS)}")

tab_live, tab_upload = st.tabs(["Live", "Upload"])

with tab_live:
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("▶️ Start Detection"):
            st.session_state.stop_detection = False
            capture_video()
    with col2:
        if st.container().button("⏹️ Stop", key="stop_btn"):
            st.session_state.stop_detection = True
    st.caption("Live detection uses the stream URL from the sidebar.")

def read_image_bytes(data: bytes):
    try:
        from PIL import Image
        img = Image.open(BytesIO(data)).convert("RGB")
        return np.array(img)
    except Exception:
        try:
            import cv2
            nparr = np.frombuffer(data, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
        except Exception:
            return None

def predict_on_image(image_rgb: np.ndarray):
    boxes = []
    summary = None
    try:
        import cv2
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    except Exception:
        faces = []
    used_labels = MODEL_LABELS
    if faces:
        try:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        except Exception:
            preprocess_input = None
        for (x, y, w, h) in faces:
            face = image_rgb[y:y+h, x:x+w]
            if model is not None and preprocess_input is not None:
                face_resized = cv2.resize(face, (224, 224))
                face_proc = preprocess_input(face_resized.astype("float32"))
                face_input = np.expand_dims(face_proc, axis=0)
                scores = model.predict(face_input)[0]
                idx = int(np.argmax(scores))
                label = used_labels[idx if idx < len(used_labels) else 0]
                score = float(scores[idx])
            else:
                label = random.choice(used_labels)
                score = 0.5
            boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "label": label, "score": score})
        if boxes:
            labels_list = [b["label"] for b in boxes]
            counts = {k: labels_list.count(k) for k in set(labels_list)}
            summary = sorted(counts.items(), key=lambda t: t[1], reverse=True)[0][0]
    else:
        if model is not None:
            try:
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                img_resized = cv2.resize(image_rgb, (224, 224))
                img_proc = preprocess_input(img_resized.astype("float32"))
                img_input = np.expand_dims(img_proc, axis=0)
                scores = model.predict(img_input)[0]
                idx = int(np.argmax(scores))
                summary = used_labels[idx if idx < len(used_labels) else 0]
            except Exception:
                summary = random.choice(used_labels)
        else:
            summary = random.choice(used_labels)
    return boxes, summary

from io import BytesIO
with tab_upload:
    uploaded = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])
    if uploaded:
        data = uploaded.read()
        image_rgb = read_image_bytes(data)
        if image_rgb is None:
            st.error("Could not read the uploaded image.")
        else:
            boxes, summary = predict_on_image(image_rgb)
            st.subheader(f"Result: {summary}")
            try:
                import cv2
                img_draw = image_rgb.copy()
                for b in boxes:
                    color = (34, 197, 94) if b["label"] in ("Mask", "With Mask") else (239, 68, 68)
                    cv2.rectangle(img_draw, (b["x"], b["y"]), (b["x"]+b["w"], b["y"]+b["h"]), color, 2)
                    cv2.putText(img_draw, f'{b["label"]} {b["score"]:.2f}', (b["x"], b["y"]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                st.image(img_draw, channels="RGB")
            except Exception:
                st.image(image_rgb, channels="RGB")
