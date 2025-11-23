import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque

# ============================== 
# CONFIGURATION
# ============================== 
VIDEO_PATH = "C:/Users/hichr/Downloads/dd.mp4"
OUTPUT_PATH = "C:/Users/hichr/Downloads/vdi_analyzed.mp4"

# Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Paramètres de détection
NOSE_THRESHOLD = 120         # Distance max nez-main (pixels)
MOUTH_THRESHOLD = 140        # Distance max bouche-main (pixels)
ELBOW_NOSE_THRESHOLD = 180   # Distance max nez-coude (pixels)
ELBOW_MOUTH_THRESHOLD = 200  # Distance max bouche-coude (pixels)
FOREARM_NOSE_THRESHOLD = 150 # Distance max nez-avant-bras (pixels)
FOREARM_MOUTH_THRESHOLD = 170 # Distance max bouche-avant-bras (pixels)
ELBOW_ANGLE_MIN = 30         # Angle min du coude pour validation
ELBOW_ANGLE_MAX = 170        # Angle max du coude pour validation
TEMPORAL_WINDOW = 15         # Nombre de frames pour détection d'éternuement
MIN_CORRECT_FRAMES = 5       # Minimum de frames correctes pour valider

# Stockage des résultats pour analyse finale
all_frames_data = []
sneeze_events = []  # Liste des événements d'éternuement détectés

# ============================== 
# FONCTIONS UTILITAIRES
# ============================== 

def distance(p1, p2):
    """Calcule la distance euclidienne entre deux points"""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def calculate_angle(p1, p2, p3):
    """
    Calcule l'angle au point p2 formé par p1-p2-p3
    Retourne l'angle en degrés
    """
    a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def analyze_sneeze_posture(landmarks, frame_shape):
    """
    Analyse complète de la posture
    Détecte: main OU coude/avant-bras devant bouche/nez
    """
    h, w = frame_shape[:2]
    
    # Extraire les points clés
    nose = np.array([landmarks[0].x * w, landmarks[0].y * h])
    mouth = np.array([landmarks[10].x * w, landmarks[10].y * h])
    
    left_shoulder = np.array([landmarks[11].x * w, landmarks[11].y * h])
    left_elbow = np.array([landmarks[13].x * w, landmarks[13].y * h])
    left_wrist = np.array([landmarks[15].x * w, landmarks[15].y * h])
    
    right_shoulder = np.array([landmarks[12].x * w, landmarks[12].y * h])
    right_elbow = np.array([landmarks[14].x * w, landmarks[14].y * h])
    right_wrist = np.array([landmarks[16].x * w, landmarks[16].y * h])
    
    # Point milieu de l'avant-bras
    left_forearm_mid = (left_elbow + left_wrist) / 2
    right_forearm_mid = (right_elbow + right_wrist) / 2
    
    # === DISTANCES ===
    dist_left_wrist_nose = distance(nose, left_wrist)
    dist_right_wrist_nose = distance(nose, right_wrist)
    dist_left_wrist_mouth = distance(mouth, left_wrist)
    dist_right_wrist_mouth = distance(mouth, right_wrist)
    
    dist_left_elbow_nose = distance(nose, left_elbow)
    dist_right_elbow_nose = distance(nose, right_elbow)
    dist_left_elbow_mouth = distance(mouth, left_elbow)
    dist_right_elbow_mouth = distance(mouth, right_elbow)
    
    dist_left_forearm_nose = distance(nose, left_forearm_mid)
    dist_right_forearm_nose = distance(nose, right_forearm_mid)
    dist_left_forearm_mouth = distance(mouth, left_forearm_mid)
    dist_right_forearm_mouth = distance(mouth, right_forearm_mid)
    
    # Calcul des angles des coudes
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # === VÉRIFICATIONS MAIN ===
    left_hand_covers = (dist_left_wrist_nose < NOSE_THRESHOLD or 
                        dist_left_wrist_mouth < MOUTH_THRESHOLD)
    right_hand_covers = (dist_right_wrist_nose < NOSE_THRESHOLD or 
                         dist_right_wrist_mouth < MOUTH_THRESHOLD)
    
    # === VÉRIFICATIONS COUDE/AVANT-BRAS ===
    left_elbow_covers = (dist_left_elbow_nose < ELBOW_NOSE_THRESHOLD or 
                         dist_left_elbow_mouth < ELBOW_MOUTH_THRESHOLD or
                         dist_left_forearm_nose < FOREARM_NOSE_THRESHOLD or
                         dist_left_forearm_mouth < FOREARM_MOUTH_THRESHOLD)
    
    right_elbow_covers = (dist_right_elbow_nose < ELBOW_NOSE_THRESHOLD or 
                          dist_right_elbow_mouth < ELBOW_MOUTH_THRESHOLD or
                          dist_right_forearm_nose < FOREARM_NOSE_THRESHOLD or
                          dist_right_forearm_mouth < FOREARM_MOUTH_THRESHOLD)
    
    # === VÉRIFICATIONS ANGLE ===
    left_elbow_bent = ELBOW_ANGLE_MIN < left_elbow_angle < ELBOW_ANGLE_MAX
    right_elbow_bent = ELBOW_ANGLE_MIN < right_elbow_angle < ELBOW_ANGLE_MAX
    
    # === VALIDATION FINALE ===
    # Correct si MAIN devant visage OU COUDE/AVANT-BRAS devant visage (avec bras plié)
    left_correct = (left_hand_covers and left_elbow_bent) or (left_elbow_covers and left_elbow_bent)
    right_correct = (right_hand_covers and right_elbow_bent) or (right_elbow_covers and right_elbow_bent)
    
    is_correct = left_correct or right_correct
    
    # Déterminer la méthode utilisée
    method_used = []
    if left_hand_covers and left_elbow_bent:
        method_used.append("Main gauche")
    elif left_elbow_covers and left_elbow_bent:
        method_used.append("Coude/Avant-bras gauche")
    
    if right_hand_covers and right_elbow_bent:
        method_used.append("Main droite")
    elif right_elbow_covers and right_elbow_bent:
        method_used.append("Coude/Avant-bras droite")
    
    details = {
        'left_wrist_nose': dist_left_wrist_nose,
        'right_wrist_nose': dist_right_wrist_nose,
        'left_elbow_nose': dist_left_elbow_nose,
        'right_elbow_nose': dist_right_elbow_nose,
        'left_elbow_angle': left_elbow_angle,
        'right_elbow_angle': right_elbow_angle,
        'method_used': method_used,
        'left_correct': left_correct,
        'right_correct': right_correct
    }
    
    return is_correct, details

def detect_sneeze_events(all_frames_data):
    """
    Analyse tous les mouvements pour détecter les événements d'éternuement
    Un éternuement est détecté quand il y a une séquence de mouvement de bras vers le visage
    """
    events = []
    i = 0
    
    while i < len(all_frames_data):
        frame_data = all_frames_data[i]
        
        if frame_data['pose_detected']:
            # Chercher une séquence de frames avec mouvement vers le visage
            sequence_correct = []
            sequence_frames = []
            j = i
            
            # Collecter une fenêtre de frames
            while j < min(i + TEMPORAL_WINDOW * 2, len(all_frames_data)):
                if all_frames_data[j]['pose_detected']:
                    sequence_correct.append(all_frames_data[j]['is_correct'])
                    sequence_frames.append(j)
                j += 1
            
            # Si on a assez de frames correctes dans la séquence, c'est un éternuement
            if len(sequence_correct) >= MIN_CORRECT_FRAMES:
                correct_count = sum(sequence_correct)
                
                if correct_count >= MIN_CORRECT_FRAMES:
                    # Éternuement détecté et correct
                    start_frame = sequence_frames[0]
                    end_frame = sequence_frames[-1]
                    
                    # Collecter les méthodes utilisées
                    methods = []
                    for frame_idx in sequence_frames:
                        methods.extend(all_frames_data[frame_idx]['details']['method_used'])
                    
                    unique_methods = list(set(methods))
                    
                    events.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'is_correct': True,
                        'methods': unique_methods,
                        'correct_frames': correct_count,
                        'total_frames': len(sequence_correct)
                    })
                    
                    i = end_frame + 1
                    continue
                elif correct_count > 0:
                    # Éternuement détecté mais incorrect (pas assez de protection)
                    start_frame = sequence_frames[0]
                    end_frame = sequence_frames[-1]
                    
                    events.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'is_correct': False,
                        'methods': [],
                        'correct_frames': correct_count,
                        'total_frames': len(sequence_correct)
                    })
                    
                    i = end_frame + 1
                    continue
        
        i += 1
    
    return events

# ============================== 
# TRAITEMENT VIDÉO
# ============================== 

cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0

# Configuration pour sauvegarder la vidéo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

print(f"🎬 Analyse de la vidéo: {VIDEO_PATH}")
print(f"📊 Résolution: {frame_width}x{frame_height}, FPS: {fps}")
print(f"⏳ Traitement en cours...\n")

# PHASE 1: Collecter toutes les données
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_id += 1
    
    # Conversion RGB pour Mediapipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    frame_data = {
        'frame_id': frame_id,
        'pose_detected': False,
        'is_correct': False,
        'details': {}
    }
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        is_correct, details = analyze_sneeze_posture(landmarks, frame.shape)
        
        frame_data['pose_detected'] = True
        frame_data['is_correct'] = is_correct
        frame_data['details'] = details
    
    all_frames_data.append(frame_data)
    
    if frame_id % 30 == 0:
        print(f"   Traité: {frame_id} frames...")

cap.release()

print(f"\n✅ Collecte terminée: {frame_id} frames analysées")
print(f"🔍 Analyse des mouvements et détection des éternuements...\n")

# PHASE 2: Détecter les événements d'éternuement
sneeze_events = detect_sneeze_events(all_frames_data)

# PHASE 3: Générer la vidéo avec résultats
cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_id += 1
    frame_data = all_frames_data[frame_id - 1]
    
    # Conversion RGB pour Mediapipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Dessiner les landmarks
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        # Vérifier si cette frame fait partie d'un événement d'éternuement
        current_event = None
        for event in sneeze_events:
            if event['start_frame'] <= frame_id <= event['end_frame']:
                current_event = event
                break
        
        if current_event:
            if current_event['is_correct']:
                methods_text = ", ".join(current_event['methods'])
                status = f"✓ ÉTERNUEMENT CORRECT"
                method_info = f"Méthode: {methods_text}"
                color = (0, 255, 0)  # Vert
            else:
                status = "✗ ÉTERNUEMENT INCORRECT"
                method_info = "Protection insuffisante"
                color = (0, 0, 255)  # Rouge
            
            cv2.putText(frame, status, (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, method_info, (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, "Position normale", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Aucune personne détectée", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    
    # Afficher le numéro de frame
    cv2.putText(frame, f"Frame: {frame_id}/{len(all_frames_data)}", 
                (frame_width - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# ============================== 
# RAPPORT FINAL
# ============================== 

print("\n" + "="*60)
print("📋 RAPPORT D'ANALYSE FINALE")
print("="*60)
print(f"\n📹 Vidéo: {VIDEO_PATH}")
print(f"📊 Total de frames: {len(all_frames_data)}")
print(f"🎯 Événements d'éternuement détectés: {len(sneeze_events)}\n")

if len(sneeze_events) == 0:
    print("❌ Aucun éternuement détecté dans la vidéo")
else:
    correct_sneezes = sum(1 for e in sneeze_events if e['is_correct'])
    incorrect_sneezes = len(sneeze_events) - correct_sneezes
    
    print(f"✅ Éternuements CORRECTS: {correct_sneezes}")
    print(f"❌ Éternuements INCORRECTS: {incorrect_sneezes}")
    print(f"📈 Taux de conformité: {(correct_sneezes/len(sneeze_events)*100):.1f}%\n")
    
    print("-" * 60)
    print("DÉTAILS DES ÉVÉNEMENTS:")
    print("-" * 60)
    
    for idx, event in enumerate(sneeze_events, 1):
        status = "✅ CORRECT" if event['is_correct'] else "❌ INCORRECT"
        start_time = event['start_frame'] / fps
        end_time = event['end_frame'] / fps
        duration = end_time - start_time
        
        print(f"\n🔸 Éternuement #{idx} - {status}")
        print(f"   Frames: {event['start_frame']} → {event['end_frame']}")
        print(f"   Temps: {start_time:.2f}s → {end_time:.2f}s (durée: {duration:.2f}s)")
        print(f"   Frames correctes: {event['correct_frames']}/{event['total_frames']}")
        
        if event['is_correct']:
            methods = ", ".join(event['methods']) if event['methods'] else "Non identifiée"
            print(f"   Méthode(s) utilisée(s): {methods}")
        else:
            print(f"   Raison: Protection insuffisante ou absente")

print("\n" + "="*60)
print(f"✅ Vidéo analysée sauvegardée: {OUTPUT_PATH}")
print("="*60 + "\n")