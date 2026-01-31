
import cv2
import os

print(f"CV2 Version: {cv2.__version__}")
try:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    print(f"Cascade Path: {path}")
    if os.path.exists(path):
        print("Cascade file exists.")
        face_cascade = cv2.CascadeClassifier(path)
        if face_cascade.empty():
            print("Error: Failed to load cascade classifier (empty).")
        else:
            print("Success: Cascade classifier loaded.")
    else:
        print("Error: Cascade file does NOT exist at that path.")
except Exception as e:
    print(f"Exception: {e}")
