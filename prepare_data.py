import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image


# Paths
BASE_DIR = r"E:\Mask_detection\mask_detection\Mask_detection_images"
IMAGES_DIR = os.path.join(BASE_DIR, "dataset")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_dataset")

# Classes
CLASSES = ["with_mask", "without_mask", "improper_mask"] # standardized names
# XML labels map to these
LABEL_MAP = {
    "with_mask": "with_mask",
    "without_mask": "without_mask",
    "mask_weared_incorrect": "improper_mask"
}

def setup_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for c in CLASSES:
        path = os.path.join(OUTPUT_DIR, c)
        if not os.path.exists(path):
            os.makedirs(path)

def find_image(filename):
    # Check root images folder
    p = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(p): return p
    
    # Check subfolders (if previously moved)
    for c in CLASSES:
        p = os.path.join(IMAGES_DIR, c, filename)
        if os.path.exists(p): return p
    
    return None

def process():
    setup_dirs()
    xml_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.xml"))
    print(f"Found {len(xml_files)} XML files.")

    count = 0
    
    for i, xml_file in enumerate(xml_files):
        if i % 100 == 0:
            print(f"Processing {i}/{len(xml_files)}")
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename = root.find("filename").text
        # Fix filename if it doesn't match pattern
        if not filename.endswith(".png"):
            # try to guess
            base = os.path.basename(xml_file).replace(".xml", ".png")
            filename = base
            
        img_path = find_image(filename)
        if not img_path:
            # try replacing extension
            img_path = find_image(filename.replace(".xml", ".png"))
        
        if not img_path:
            print(f"Image not found: {filename}")
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            continue
            
        for member in root.findall("object"):
            label = member.find("name").text
            if label not in LABEL_MAP:
                continue
                
            target_class = LABEL_MAP[label]
            
            bndbox = member.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            
            # Crop
            try:
                crop = img.crop((xmin, ymin, xmax, ymax))
                # Save
                save_name = f"{os.path.splitext(filename)[0]}_{count}.png"
                save_path = os.path.join(OUTPUT_DIR, target_class, save_name)
                crop.save(save_path)
                count += 1
            except Exception as e:
                print(f"Error cropping {filename}: {e}")

    print(f"✅ Processed {count} faces into {OUTPUT_DIR}")

if __name__ == "__main__":
    process()
