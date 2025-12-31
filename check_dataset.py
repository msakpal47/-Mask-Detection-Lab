import os

DATA_DIR = r"E:\Mask_detection\mask_detection\Mask_detection_images\images"

for root, dirs, files in os.walk(DATA_DIR):
    print("FOLDER:", root)
    for f in files[:5]:
        print("  ", f)
