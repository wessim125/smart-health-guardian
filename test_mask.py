from ultralytics import YOLO
import cv2
import os

# =========================
# CONFIGURATION MODÈLE
# =========================
MODEL_PATH = "runs/mask/yolov8_mask_local/weights/best.pt"
if not os.path.exists(MODEL_PATH):
    print(f"⚠ Modèle manquant : {MODEL_PATH}")
    exit()

model = YOLO(MODEL_PATH)
classes = ['with_mask', 'without_mask', 'incorrect_mask']

# =========================
# FONCTION DE DÉTECTION
# =========================
def detect_masks(source, save_dir="C:/Users/hichr/smart-health-guardian/results"):
    out = None

    # Préparer dossier de sauvegarde
    os.makedirs(save_dir, exist_ok=True)

    # Si c'est une vidéo, préparer l'enregistrement
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Erreur : impossible d'ouvrir la vidéo.")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        filename = f"result_{os.path.basename(source)}"
        save_path = os.path.join(save_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"Vidéo annotée sera sauvegardée : {save_path}")

    results = model(source, stream=True, conf=0.5, iou=0.45)

    for result in results:
        img = result.orig_img.copy()

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = box.conf[0].item()
            label = f"{classes[cls_id]} {conf:.2f}"

            # Couleurs
            color = (0, 255, 0)   # vert = masque correct
            if cls_id == 1: color = (0, 0, 255)   # rouge = sans masque
            if cls_id == 2: color = (255, 255, 0) # bleu = masque mal porté

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Affichage et écriture vidéo
        cv2.imshow('Detection Masques', img)
        if out:
            out.write(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    if out:
        out.release()
    print("✅ Traitement terminé !")

# =========================
# MENU INTERACTIF
# =========================
print("\n" + "="*60)
print("     DETECTION MASQUES - SMART HEALTH GUARDIAN")
print("="*60)
print("1. Webcam (temps réel)")
print("2. Image")
print("3. Vidéo")
print("q. Quitter")
print("-"*60)

choice = input("Choisis (1/2/3/q) : ").strip()

if choice == '1':
    detect_masks(0)

elif choice == '2':
    path = input("Chemin image (ex: test.jpg) : ").strip()
    if os.path.exists(path):
        detect_masks(path)
    else:
        print("Image non trouvée !")

elif choice == '3':
    path = input("Nom ou chemin vidéo (ex: masque.mp4) : ").strip()
    if os.path.exists(path):
        detect_masks(path)  # La vidéo sera automatiquement sauvegardée dans results
    else:
        print("Vidéo non trouvée !")

elif choice.lower() == 'q':
    print("Au revoir !")
else:
    print("Choix invalide")
