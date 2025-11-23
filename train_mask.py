# scripts/train_mask.py
# Entraînement YOLOv8 local (masques uniquement)
# CORRIGÉ POUR WINDOWS + MULTIPROCESSING

import os
import torch

# === RÉSOUDRE CONFLIT OPENMP ===
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# === VÉRIFIER GPU ===
print("CUDA disponible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
else:
    print("Mode CPU activé")

# === FONCTION PRINCIPALE ===
def main():
    from ultralytics import YOLO

    # Charger modèle
    model = YOLO('yolov8s.pt')  # ou yolov8n.pt

    # Entraînement
    results = model.train(
        data='mask.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        augment=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.0,
        name='yolov8_mask_local',
        project='runs/mask',
        exist_ok=True,
        plots=True,
        save=True,
        device='cpu' if not torch.cuda.is_available() else 0,
        workers=0  # CRUCIAL : 0 sur Windows pour éviter le bug
    )

    # Évaluation
    metrics = model.val()
    print("\n" + "="*50)
    print("RÉSULTATS FINAUX")
    print("="*50)
    print(f"mAP@0.5       : {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95  : {metrics.box.map:.3f}")
    print("="*50)

    # Export
    print("Export en cours...")
    model.export(format='onnx', imgsz=640)
    model.export(format='torchscript')

    print("ENTRAÎNEMENT TERMINÉ !")
    print("Modèle → runs/mask/yolov8_mask_local/weights/best.pt")

# === PROTECTION WINDOWS (OBLIGATOIRE) ===
if __name__ == '__main__':
    main()