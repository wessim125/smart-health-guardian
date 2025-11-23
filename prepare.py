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
import subprocess

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

# Default input path: change this to a video file path to use it when you don't pass
# --input on the command line. Set to None to require --input or --webcam.
DEFAULT_INPUT = "C:/Users/hichr/Downloads/dd.mp4"
# You can add multiple default inputs here (useful for testing multiple videos).
# The script will use the first path in this list when no --input is provided.
DEFAULT_INPUTS = [
    # r"C:\Users\hichr\Downloads\vdi.mp4",
]

# If you prefer to hardcode a single default video, set DEFAULT_INPUT above.
# The launcher below will call `prepare_realtime.py` with this input when you
# run `prepare.py` without arguments.


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher helper: runs the realtime analyzer with a default path set in this file')
    parser.add_argument('--input', '-i', required=False, default=None, help='Video file to analyze (overrides DEFAULT_INPUT)')
    parser.add_argument('--outdir', '-o', required=False, default='results/video_analysis', help='Output folder to pass to analyzer')
    parser.add_argument('--show', action='store_true', help='Show visualization window during analysis')
    parser.add_argument('--webcam', action='store_true', help='Use webcam instead of a file')
    args = parser.parse_args()

    # Resolve input: CLI -> DEFAULT_INPUT -> DEFAULT_INPUTS
    inp = args.input or DEFAULT_INPUT
    if not inp and DEFAULT_INPUTS:
        inp = DEFAULT_INPUTS[0]

    if args.webcam:
        inp = 'webcam'

    if not inp:
        print('Aucun chemin vidéo fourni. Éditez DEFAULT_INPUT or DEFAULT_INPUTS in this file or pass --input <path>')
        sys.exit(1)

    # Build command to call the real analyzer (prepare_realtime.py)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    analyzer = os.path.join(this_dir, 'prepare_realtime.py')
    cmd = [sys.executable, analyzer, '--input', str(inp), '--outdir', args.outdir]
    if args.show:
        cmd.append('--show')
    if args.webcam:
        # prepare_realtime accepts --webcam instead of --input
        cmd = [sys.executable, analyzer, '--webcam', '--outdir', args.outdir]

    print('Launching analyzer:')
    print(' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print('Analyzer exited with code', e.returncode)