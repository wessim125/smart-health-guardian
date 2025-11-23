import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import sys
import os
import glob
import json
import time
import argparse

# ==============================
# CONFIGURATION GLOBALE & MEDIAPIPE
# ==============================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Re-use pose/hands objects for efficiency
POSE_DETECTOR = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                             min_detection_confidence=0.6, min_tracking_confidence=0.6)
HANDS_DETECTOR = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Decision thresholds (ajustables)
WINDOW_SIZE = 12                 # fenêtre pour voter la couverture
MOVEMENT_THRESHOLD = 45.0        # pic de mouvement minimal (empirique)
HEAD_DROP_SUM_THRESHOLD = 0.06   # somme minimale des chutes de tête dans la fenêtre
COVERAGE_RATIO_THRESHOLD = 0.6   # ratio de frames couvertes pour considérer l'éternuement bien couvert
MIN_LANDMARK_VISIBILITY = 0.35   # visibilité minimale pour considérer landmarks fiables

# Precision improvements
MIN_FRAMES_BETWEEN_EVENTS = 18    # éviter doublons trop proches
DEPTH_FRONT_THRESHOLD = 0.08     # z difference to consider hand in front
FACE_TURN_THRESHOLD = 0.12       # normalized x offset of nose vs shoulders to consider face turned
PALM_VEL_THRESHOLD = 0.02        # normalized palm velocity threshold to increase confidence


# ==============================
# HELPERS
# ==============================

def calculate_angle(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    cos_angle = np.dot(a, b) / denom
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle


def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def norm_dist(p1, p2, diag):
    """Distance normalisée par la diagonale du cadre pour invariante d'échelle."""
    return dist(p1, p2) / (diag + 1e-6)


def is_hand_covering(nose, mouth, wrist, palm_center, wrist_z, nose_z, hand_closed, diag):
    """Décide si la main couvre le visage (retourne bool et méthode)."""
    if palm_center is not None:
        if norm_dist(palm_center, nose, diag) < 0.15 or norm_dist(palm_center, mouth, diag) < 0.17:
            return True, "Main (paume)"
        if hand_closed and norm_dist(palm_center, nose, diag) < 0.22:
            return True, "Main fermée"
    if norm_dist(wrist, nose, diag) < 0.18 or norm_dist(wrist, mouth, diag) < 0.20:
        return True, "Poignet"
    if wrist_z is not None and nose_z is not None and (nose_z - wrist_z) > DEPTH_FRONT_THRESHOLD:
        return True, "Main devant (3D)"
    return False, ""


def is_elbow_covering(elbow, nose, angle, diag):
    if norm_dist(elbow, nose, diag) < 0.26 and 40 < angle < 165:
        return True, "Coude"
    return False, ""


def is_reliable_pose(landmarks):
    """Vérifie si les landmarks ont une visibilité suffisante (si disponible)."""
    try:
        vis = [getattr(lm, 'visibility', 1.0) for lm in landmarks.landmark]
        return np.median(vis) >= MIN_LANDMARK_VISIBILITY
    except Exception:
        return True


def _palm_velocity(frames_buf, diag):
    if len(frames_buf) < 2:
        return 0.0
    vel_l = []
    vel_r = []
    for a, b in zip(list(frames_buf)[:-1], list(frames_buf)[1:]):
        if a.get('palm_l') is not None and b.get('palm_l') is not None:
            vel_l.append(norm_dist(a['palm_l'], b['palm_l'], diag))
        if a.get('palm_r') is not None and b.get('palm_r') is not None:
            vel_r.append(norm_dist(a['palm_r'], b['palm_r'], diag))
    mean_l = np.mean(vel_l) if vel_l else 0.0
    mean_r = np.mean(vel_r) if vel_r else 0.0
    return max(mean_l, mean_r)


def coverage_vote(frames_buf, diag, peak_index=None):
    """Vote pondéré sur la fenêtre: retourne weighted_ratio et méthodes majoritaires."""
    n = len(frames_buf)
    if n == 0:
        return 0.0, []
    center = peak_index if peak_index is not None else (n - 1) // 2
    covered_weight = 0.0
    total_weight = 0.0
    methods = []

    palm_vel = _palm_velocity(frames_buf, diag)

    for i, snap in enumerate(frames_buf):
        if not snap.get('pose') or not is_reliable_pose(snap['pose']):
            continue
        wgt = 1.0 - (abs(i - center) / (center + 1e-6))
        wgt = max(0.0, wgt)
        try:
            shoulders_mid_x = (snap['pose'].landmark[11].x + snap['pose'].landmark[12].x) / 2.0
            nose_x = snap['pose'].landmark[0].x
            turn_norm = abs(nose_x - shoulders_mid_x)
            if turn_norm > FACE_TURN_THRESHOLD:
                wgt *= 0.6
        except Exception:
            pass

        cl, ml = is_hand_covering(snap['nose'], snap['mouth'], snap['wrist_l'], snap['palm_l'], snap['wrist_l_z'], snap['nose_z'], snap['left_closed'], diag)
        cr, mr = is_hand_covering(snap['nose'], snap['mouth'], snap['wrist_r'], snap['palm_r'], snap['wrist_r_z'], snap['nose_z'], snap['right_closed'], diag)
        cel, _ = is_elbow_covering(snap['elbow_l'], snap['nose'], snap['elbow_l_angle'], diag)
        cer, _ = is_elbow_covering(snap['elbow_r'], snap['nose'], snap['elbow_r_angle'], diag)

        depth_bonus = 1.0
        try:
            if snap.get('wrist_l_z') is not None and snap.get('nose_z') is not None:
                if snap['wrist_l_z'] < snap['nose_z'] - DEPTH_FRONT_THRESHOLD:
                    depth_bonus = 1.2
            if snap.get('wrist_r_z') is not None and snap.get('nose_z') is not None:
                if snap['wrist_r_z'] < snap['nose_z'] - DEPTH_FRONT_THRESHOLD:
                    depth_bonus = 1.2
        except Exception:
            pass

        covered = cl or cr or cel or cer
        if covered:
            total_weight += wgt
            covered_weight += wgt * depth_bonus
            if cl and ml: methods.append(ml)
            if cr and mr: methods.append(mr)
            if cel: methods.append('Coude gauche')
            if cer: methods.append('Coude droit')

    weighted_ratio = (covered_weight / (total_weight + 1e-6)) if total_weight > 0 else 0.0
    if palm_vel > PALM_VEL_THRESHOLD:
        weighted_ratio = min(1.0, weighted_ratio * 1.12)

    methods_summary = []
    if methods:
        uniq = {}
        for m in methods:
            uniq[m] = uniq.get(m, 0) + 1
        methods_summary = sorted(uniq.items(), key=lambda x: x[1], reverse=True)
        methods_summary = [m for m, _ in methods_summary[:3]]

    return weighted_ratio, methods_summary


# ==============================
# PROCESSING
# ==============================

def process_video(video_path, output_path=None, show_window=False):
    """Traite une vidéo et retourne un résumé."""
    print(f"Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERREUR: impossible d'ouvrir la source vidéo '{video_path}'")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = base + '_analyzed.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # local buffers
    sneeze_events_local = []
    MOVEMENT_BUFFER_local = deque(maxlen=WINDOW_SIZE)
    HEAD_DROP_BUFFER_local = deque(maxlen=WINDOW_SIZE)
    frames_buffer_local = deque(maxlen=WINDOW_SIZE)
    prev_nose_y_local = None
    prev_landmarks_local = None
    last_event_frame = -999
    frame_logs = []

    frame_id = 0
    diag = math.hypot(w, h)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = POSE_DETECTOR.process(rgb)
        hand_results = HANDS_DETECTOR.process(rgb)

        display = frame.copy()
        nose_pos = mouth_pos = None
        nose_z = None

        movement = 0.0
        head_drop = 0.0

        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            hh, ww = frame.shape[:2]

            # Points clés
            nose = [lm[0].x * ww, lm[0].y * hh]; nose_z = lm[0].z
            mouth = [lm[10].x * ww, lm[10].y * hh]
            l_wrist = [lm[15].x * ww, lm[15].y * hh]; l_wrist_z = lm[15].z
            r_wrist = [lm[16].x * ww, lm[16].y * hh]; r_wrist_z = lm[16].z
            l_elbow = [lm[13].x * ww, lm[13].y * hh]
            r_elbow = [lm[14].x * ww, lm[14].y * hh]
            l_shoulder = [lm[11].x * ww, lm[11].y * hh]
            r_shoulder = [lm[12].x * ww, lm[12].y * hh]

            nose_pos = np.array(nose)
            mouth_pos = np.array(mouth)

            mp_drawing.draw_landmarks(display, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # mains
            left_palm = right_palm = None
            left_closed = right_closed = False
            left_hand_lms = right_hand_lms = None
            if hand_results.multi_hand_landmarks:
                for hand_lms, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    label = handedness.classification[0].label
                    wrist_pt = hand_lms.landmark[0]
                    mcp_pt = hand_lms.landmark[9]
                    palm_x = (wrist_pt.x * ww + mcp_pt.x * ww) / 2
                    palm_y = (wrist_pt.y * hh + mcp_pt.y * hh) / 2
                    palm = np.array([palm_x, palm_y])
                    closed = sum(1 for i in [8,12,16,20] if hand_lms.landmark[i].y > hand_lms.landmark[i-2].y) >= 3
                    if label == 'Left':
                        left_palm, left_closed, left_hand_lms = palm, closed, hand_lms
                    else:
                        right_palm, right_closed, right_hand_lms = palm, closed, hand_lms
                    mp_drawing.draw_landmarks(display, hand_lms, mp_hands.HAND_CONNECTIONS)

            # mouvement
            if prev_landmarks_local:
                for idx in [13,14,15,16]:
                    curr = lm[idx]
                    prev = prev_landmarks_local.landmark[idx]
                    movement += abs(curr.x - prev.x) + abs(curr.y - prev.y)
            MOVEMENT_BUFFER_local.append(movement * 1000.0)

            # head drop
            if prev_nose_y_local is not None:
                head_drop = lm[0].y - prev_nose_y_local
            HEAD_DROP_BUFFER_local.append(head_drop)
            prev_nose_y_local = lm[0].y

            # snapshot
            elbow_l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            elbow_r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            snapshot = {
                'pose': pose_results.pose_landmarks,
                'nose': nose_pos,
                'mouth': mouth_pos,
                'wrist_l': np.array(l_wrist),
                'wrist_r': np.array(r_wrist),
                'palm_l': left_palm,
                'palm_r': right_palm,
                'wrist_l_z': l_wrist_z,
                'wrist_r_z': r_wrist_z,
                'left_closed': left_closed,
                'right_closed': right_closed,
                'elbow_l': np.array(l_elbow),
                'elbow_r': np.array(r_elbow),
                'elbow_l_angle': elbow_l_angle,
                'elbow_r_angle': elbow_r_angle,
                'nose_z': nose_z,
            }
            frames_buffer_local.append(snapshot)

            # decision window
            if len(MOVEMENT_BUFFER_local) == MOVEMENT_BUFFER_local.maxlen:
                peak = max(MOVEMENT_BUFFER_local)
                head_drop_sum = sum(HEAD_DROP_BUFFER_local)
                if peak > MOVEMENT_THRESHOLD and head_drop_sum > HEAD_DROP_SUM_THRESHOLD:
                    peak_idx = int(np.argmax(np.array(MOVEMENT_BUFFER_local)))
                    coverage_ratio, methods_summary = coverage_vote(list(frames_buffer_local), diag, peak_index=peak_idx)
                    is_covered_final = coverage_ratio >= COVERAGE_RATIO_THRESHOLD
                    if frame_id - last_event_frame > MIN_FRAMES_BETWEEN_EVENTS:
                        sneeze_events_local.append({
                            'frame': frame_id,
                            'covered': is_covered_final,
                            'coverage_ratio': float(coverage_ratio),
                            'methods': methods_summary,
                            'peak_movement': float(peak),
                            'head_drop_sum': float(head_drop_sum)
                        })
                        last_event_frame = frame_id
                        label = 'ETERNELLEMENT !'
                        color = (0,255,0) if is_covered_final else (0,0,255)
                        cv2.putText(display, label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

            prev_landmarks_local = pose_results.pose_landmarks

        # write frame and small per-frame log
        out.write(display)
        frame_logs.append({
            'frame': frame_id,
            'movement': float(movement),
            'head_drop': float(head_drop),
            'sneeze_count': len(sneeze_events_local)
        })

        if show_window:
            cv2.imshow('Analyse Finale - Éternuement Correct ?', display)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    out.release()
    if show_window:
        cv2.destroyAllWindows()

    # résumé et sauvegardes
    total_sneezes = len(sneeze_events_local)
    correct_sneezes = sum(1 for e in sneeze_events_local if e['covered'])
    rate = (correct_sneezes / total_sneezes * 100) if total_sneezes else 0.0

    print(f"Processed {video_path}: frames={frame_id}, sneezes={total_sneezes}, correct={correct_sneezes}, rate={rate:.1f}%")

    # save events and per-frame logs
    base = os.path.splitext(os.path.basename(video_path))[0]
    events_path = base + '_sneeze_events.json'
    frame_log_path = base + '_frame_log.json'
    try:
        with open(events_path, 'w', encoding='utf-8') as f:
            json.dump({'video': video_path, 'events': sneeze_events_local}, f, ensure_ascii=False, indent=2)
        with open(frame_log_path, 'w', encoding='utf-8') as f:
            json.dump({'frame_logs': frame_logs}, f, ensure_ascii=False, indent=2)
        print(f"Saved: {events_path}, {frame_log_path}, video out: {output_path}")
    except Exception as e:
        print('Error saving logs:', e)

    return {'video': video_path, 'frames': frame_id, 'sneezes': total_sneezes, 'correct': correct_sneezes, 'rate': rate}


# ==============================
# CLI / BATCH
# ==============================
if __name__ == '__main__':
    # === Mets ton chemin ici ===
    video_path = "C:/Users/hichr/Downloads/ff.mp4"
    outdir = "C:/Users/hichr/smart-health-guardian/results"            # <-- dossier sortie
    
    os.makedirs(outdir, exist_ok=True)

    # Générer nom sortie
    out_name = os.path.join(outdir,
                            os.path.splitext(os.path.basename(video_path))[0] + "_analyzed.mp4")

    # Lancer le traitement
    res = process_video(video_path, out_name, show_window=True)

    print("\nRésumé :")
    print(res)
