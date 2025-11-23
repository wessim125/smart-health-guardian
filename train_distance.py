# scripts/train_distance_from_zero.py
# ENTRAÎNEMENT DE 0 → 7 ÉPOQUES : HYPERPARAMÈTRES SOTA
# Objectif : mAP@0.5 > 99% | mAP@0.5:0.95 > 90% | ±2cm précision

from ultralytics import YOLO
import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==============================
# === CHARGER MODÈLE DE BASE ===
# ==============================
# Commence de zéro (pas de resume)
model = YOLO('yolov8s.pt')  # s = meilleur équilibre vitesse/précision

# ==============================
# === HYPERPARAMÈTRES SOTA ===
# ==============================
results = model.train(
    # === DATASET ===
    data='distance.yaml',       # Ton dataset (1m/2m)
    epochs=7,                   # ← SEULEMENT 7 ÉPOQUES
    imgsz=640,
    batch=32,                   # 32 = optimal GPU 6GB+
    workers=0,                  # 0 = stable Windows
    cache='disk',               # SSD = +50% vitesse

    # === OPTIMISEUR (AdamW + Warmup long) ===
    optimizer='AdamW',
    lr0=0.001,                  # ← HAUT pour 7 époques (apprentissage rapide)
    lrf=0.01,                   # Decay final
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,          # ← 3/7 = warmup agressif
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # === LOSS (PRIORITÉ LOCALISATION) ===
    box=15.0,                   # ← TRÈS HAUT = boîtes parfaites
    cls=0.5,
    dfl=2.0,                    # ← DFL boosté
    pose=0.0,
    kobj=1.0,

    # === AUGMENTATION ULTRA-AGRESSIVE ===
    hsv_h=0.03, hsv_s=0.9, hsv_v=0.6,   # Lumières variables
    degrees=25.0,                       # Rotations
    translate=0.4,                      # Déplacements
    scale=1.0,                          # Zoom
    shear=10.0,                         # Cisaillement
    perspective=0.002,                  # 3D
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,                         # Essentiel
    mixup=0.6,                          # Généralisation
    copy_paste=0.8,                     # Plus de personnes

    # === PERFORMANCE ===
    amp=True,                   # Mixed precision = +70% vitesse
    cos_lr=True,                # Pic optimal
    close_mosaic=0,             # OFF (7 époques = pas besoin)
    single_cls=True,            # 1 classe = plus rapide

    # === EARLY STOPPING (SÉCURITÉ) ===
    patience=3,                 # Arrête si stagnation

    # === SAUVEGARDE ===
    name='yolov8_distance_7epochs_sota',
    project='runs/distance',
    exist_ok=True,
    plots=True,
    save_period=1,              # Sauvegarde chaque époque (1 à 7)
    val=True,                   # Évalue chaque époque → best.pt

    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# ==============================
# === ÉVALUATION FINALE ===
# ==============================
print("\n" + "="*70)
print(" ENTRAÎNEMENT TERMINÉ – 7 ÉPOQUES SOTA")
print("="*70)
metrics = model.val()
print(f"mAP@0.5       : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")
print(f"Précision     : {metrics.box.p.mean():.4f}")
print(f"Rappel        : {metrics.box.r.mean():.4f}")
print(f"Modèle final  : runs/distance/yolov8_distance_7epochs_sota/weights/best.pt")
print("="*70)

# === EXPORT ONNX ===
model.export(format='onnx', imgsz=640, simplify=True)
print("Export ONNX terminé : best.onnx")