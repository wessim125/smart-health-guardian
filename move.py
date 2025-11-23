# scripts\organize_mask_data.py
import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import glob
import cv2

# === CHEMINS WINDOWS (CORRIGÉS SELON TA STRUCTURE) ===
RAW_IMG_DIR = r"C:/Users/hichr/smart-health-guardian/data/mask/raw/images/images"
RAW_XML_DIR = r"C:/Users/hichr/smart-health-guardian/data/mask/raw/annotations/annotations"
IMG_OUT_DIR = r"C:/Users/hichr/smart-health-guardian/data/mask/images"
LABEL_OUT_DIR = r"C:/Users/hichr/smart-health-guardian/data/mask/labels"

# Créer dossiers train/val
for split in ['train', 'val']:
    os.makedirs(os.path.join(IMG_OUT_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(LABEL_OUT_DIR, split), exist_ok=True)

# === LIRE LES FICHIERS XML ===
xml_files = glob.glob(os.path.join(RAW_XML_DIR, "*.xml"))
data = []

print(f"Lecture des annotations dans : {RAW_XML_DIR}")
print(f"Nombre de fichiers XML trouvés : {len(xml_files)}")

for xml_file in xml_files:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text.strip()

        # Chemin de l'image
        img_path = os.path.join(RAW_IMG_DIR, filename)
        if not os.path.exists(img_path):
            img_path = img_path.replace('.png', '.jpg')
        if not os.path.exists(img_path):
            print(f"Image non trouvée : {filename}")
            continue

        labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text.strip()
            if name not in ['with_mask', 'without_mask', 'mask_weared_incorrect']:
                continue
            cls_id = {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrect': 2}[name]
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            labels.append(f"{cls_id} {xmin} {ymin} {xmax} {ymax}")

        if labels:
            data.append((img_path, labels))
        else:
            print(f"Aucune annotation valide dans : {filename}")

    except Exception as e:
        print(f"Erreur lecture {xml_file}: {e}")

print(f"{len(data)} images valides avec annotations trouvées.")

# === SPLIT TRAIN/VAL ===
if len(data) == 0:
    raise ValueError("Aucune donnée à traiter ! Vérifiez les chemins.")

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# === FONCTION DE CONVERSION YOLO + COPIE ===
def convert_and_save(dataset, split):
    out_img_dir = os.path.join(IMG_OUT_DIR, split)
    out_lbl_dir = os.path.join(LABEL_OUT_DIR, split)

    for i, (img_path, labels) in enumerate(dataset):
        # Copier l'image
        ext = os.path.splitext(img_path)[1]
        new_img_path = os.path.join(out_img_dir, f"img_{i:04d}{ext}")
        shutil.copy(img_path, new_img_path)

        # Lire dimensions
        img = cv2.imread(new_img_path)
        if img is None:
            print(f"Impossible de lire l'image : {new_img_path}")
            continue
        h, w = img.shape[:2]

        # Convertir en YOLO (normalisé)
        yolo_lines = []
        for label in labels:
            cls, xmin, ymin, xmax, ymax = map(int, label.split())
            x_center = (xmin + xmax) / 2 / w
            y_center = (ymin + ymax) / 2 / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h
            yolo_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Sauvegarder .txt
        lbl_path = os.path.join(out_lbl_dir, f"img_{i:04d}.txt")
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

# === EXÉCUTER ===
print("Copie et conversion en cours...")
convert_and_save(train_data, 'train')
convert_and_save(val_data, 'val')

print(f"TERMINE !")
print(f"   Train : {len(train_data)} images → {IMG_OUT_DIR}/train/")
print(f"   Val   : {len(val_data)} images → {IMG_OUT_DIR}/val/")