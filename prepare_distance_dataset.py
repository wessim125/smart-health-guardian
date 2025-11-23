import cv2
import numpy as np
from ultralytics import YOLO
import os

# ==============================
# CONFIGURATION
# ==============================
VIDEO_PATH = "C:/Users/hichr/Downloads/metro.mp4"
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45
DISTANCE_THRESHOLD = 100  # seuil pour alerter (en pixels)
OUTPUT_PATH = "C:/Users/hichr/smart-health-guardian/results/analyzed_video.mp4"

# ==============================
# CHARGER LE MODÈLE YOLO
# ==============================
model = YOLO(MODEL_PATH)

# ==============================
# TRAITEMENT VIDÉO
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0

# Récupérer les dimensions et fps de la vidéo
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Créer le VideoWriter pour sauvegarder la vidéo
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    results = model(frame)[0]  # Détection

    persons = []
    person_id = 0
    for det in results.boxes:
        cls = int(det.cls[0])
        conf = float(det.conf[0])
        if cls == 0 and conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            person_id += 1
            persons.append({"id": person_id, "bbox": [x1, y1, x2, y2], "center": (cx, cy)})

    # ==============================
    # CALCUL DES DISTANCES ET ALERTES
    # ==============================
    n = len(persons)
    for i in range(n):
        for j in range(i + 1, n):
            (x1, y1) = persons[i]["center"]
            (x2, y2) = persons[j]["center"]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            if distance < DISTANCE_THRESHOLD:
                color = (0, 0, 255)  # rouge = danger
                alert_text = "RISQUE SECURITE ÉLEVÉ"
            else:
                color = (0, 255, 0)  # vert = ok
                alert_text = ""

            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            dist_text = f"{int(distance)} px"
            (w, h), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (mid_x - 2, mid_y - h - 2), (mid_x + w + 2, mid_y + 2), (0,0,0), -1)
            cv2.putText(frame, dist_text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if alert_text:
                (w2, h2), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (mid_x - 2, mid_y + 5), (mid_x + w2 + 2, mid_y + h2 + 5), (0,0,255), -1)
                cv2.putText(frame, alert_text, (mid_x, mid_y + h2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # ==============================
    # BOUNDING BOX ET LABEL PERSONNE
    # ==============================
    for person in persons:
        x1, y1, x2, y2 = person["bbox"]
        cx, cy = person["center"]
        pid = person["id"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        label = f"Person {pid}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), (0,0,0), -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # ==============================
    # AFFICHAGE FRAME
    # ==============================
    cv2.putText(frame, f"Frame: {frame_id}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Distance entre personnes", frame)

    # Écrire la frame dans la vidéo
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Vidéo enregistrée dans :", OUTPUT_PATH)
