import os
# Reduce TensorFlow / tflite startup verbosity early to avoid noisy warnings
# Must be set before importing tensorflow/mediapipe so the delegate logs are quieter.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
try:
    # Try to silence absl if available
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass
# Set GLOG_minloglevel to 2 to hide INFO/WARNING messages from some native libs
os.environ.setdefault('GLOG_minloglevel', '2')

import cv2
import mediapipe as mp
import math
import numpy as np
import argparse
import time
from collections import deque

# This script is a lightweight real-time/front-facing adaptation of the
# existing prepare_sneez logic to allow testing with the webcam (device 0)

# ==============================
# CONFIGURATION
# ==============================
TEMPORAL_WINDOW = 15
MIN_CORRECT_FRAMES = 5
# Default input path: change this to the video you want to analyze by default.
# If set, this path will be used when you don't pass --input on the command line.
DEFAULT_INPUT = r"C:/Users/hichr/Downloads/ff.mp4"

# Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Hands detector (for palm position + open/closed check)
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ==============================
# UTILITIES (from prepare_sneez)
# ==============================

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def calculate_angle(p1, p2, p3):
    a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def analyze_sneeze_posture(landmarks, frame_shape):
    h, w = frame_shape[:2]
    nose = np.array([landmarks[0].x * w, landmarks[0].y * h])
    mouth = np.array([landmarks[10].x * w, landmarks[10].y * h])

    left_shoulder = np.array([landmarks[11].x * w, landmarks[11].y * h])
    left_elbow = np.array([landmarks[13].x * w, landmarks[13].y * h])
    left_wrist = np.array([landmarks[15].x * w, landmarks[15].y * h])

    right_shoulder = np.array([landmarks[12].x * w, landmarks[12].y * h])
    right_elbow = np.array([landmarks[14].x * w, landmarks[14].y * h])
    right_wrist = np.array([landmarks[16].x * w, landmarks[16].y * h])

    left_forearm_mid = (left_elbow + left_wrist) / 2
    right_forearm_mid = (right_elbow + right_wrist) / 2

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

    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # thresholds tuned for typical webcam scale (pixels)
    NOSE_THRESHOLD = max(w, h) * 0.12
    MOUTH_THRESHOLD = max(w, h) * 0.14
    ELBOW_NOSE_THRESHOLD = max(w, h) * 0.18
    ELBOW_MOUTH_THRESHOLD = max(w, h) * 0.20
    FOREARM_NOSE_THRESHOLD = max(w, h) * 0.15
    FOREARM_MOUTH_THRESHOLD = max(w, h) * 0.17
    ELBOW_ANGLE_MIN = 30
    ELBOW_ANGLE_MAX = 170

    left_hand_covers = (dist_left_wrist_nose < NOSE_THRESHOLD or dist_left_wrist_mouth < MOUTH_THRESHOLD)
    right_hand_covers = (dist_right_wrist_nose < NOSE_THRESHOLD or dist_right_wrist_mouth < MOUTH_THRESHOLD)

    left_elbow_covers = (dist_left_elbow_nose < ELBOW_NOSE_THRESHOLD or dist_left_elbow_mouth < ELBOW_MOUTH_THRESHOLD or dist_left_forearm_nose < FOREARM_NOSE_THRESHOLD or dist_left_forearm_mouth < FOREARM_MOUTH_THRESHOLD)
    right_elbow_covers = (dist_right_elbow_nose < ELBOW_NOSE_THRESHOLD or dist_right_elbow_mouth < ELBOW_MOUTH_THRESHOLD or dist_right_forearm_nose < FOREARM_NOSE_THRESHOLD or dist_right_forearm_mouth < FOREARM_MOUTH_THRESHOLD)

    left_elbow_bent = ELBOW_ANGLE_MIN < left_elbow_angle < ELBOW_ANGLE_MAX
    right_elbow_bent = ELBOW_ANGLE_MIN < right_elbow_angle < ELBOW_ANGLE_MAX

    left_correct = (left_hand_covers and left_elbow_bent) or (left_elbow_covers and left_elbow_bent)
    right_correct = (right_hand_covers and right_elbow_bent) or (right_elbow_covers and right_elbow_bent)

    is_correct = left_correct or right_correct

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
    events = []
    i = 0
    while i < len(all_frames_data):
        frame_data = all_frames_data[i]
        if frame_data['pose_detected']:
            sequence_correct = []
            sequence_frames = []
            j = i
            while j < min(i + TEMPORAL_WINDOW * 2, len(all_frames_data)):
                if all_frames_data[j]['pose_detected']:
                    sequence_correct.append(all_frames_data[j]['is_correct'])
                    sequence_frames.append(j)
                j += 1
            if len(sequence_correct) >= MIN_CORRECT_FRAMES:
                correct_count = sum(sequence_correct)
                if correct_count >= MIN_CORRECT_FRAMES:
                    start_frame = sequence_frames[0]
                    end_frame = sequence_frames[-1]
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
# MAIN - real-time / webcam
# ==============================

def main():
    parser = argparse.ArgumentParser(description='Real-time sneeze coverage tester (webcam or file)')
    parser.add_argument('--webcam', action='store_true', help='Use webcam device 0')
    parser.add_argument('--input', '-i', required=False, default=None, help='Video file input')
    parser.add_argument('--outdir', '-o', required=False, default='results/video_analysis', help='Output directory for logs/video')
    parser.add_argument('--show', action='store_true', help='Show window while processing')
    args = parser.parse_args()

    # Prefer CLI args, then DEFAULT_INPUT, then require user input
    if args.webcam:
        src = 0
    else:
        src = args.input or DEFAULT_INPUT
        if not src:
            print('Provide --webcam or --input <file> (or set DEFAULT_INPUT in the script)')
            return

    os.makedirs(args.outdir, exist_ok=True)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print('Cannot open source:', src)
        return

    frame_id = 0
    all_frames_data = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = 'webcam_' + time.strftime('%Y%m%d_%H%M%S') if args.webcam else os.path.splitext(os.path.basename(src))[0]
    out_path = os.path.join(args.outdir, base + '_analyzed.mp4')
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    print('Starting real-time analysis. Press q to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
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
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        all_frames_data.append(frame_data)

        # detect events on collected frames so far (lightweight)
        events = detect_sneeze_events(all_frames_data)
        current_event = None
        for event in events:
            if event['start_frame'] <= frame_id <= event['end_frame']:
                current_event = event
                break

        if current_event:
            if current_event['is_correct']:
                status = '✓ ÉTERN. CORRECT'
                color = (0, 255, 0)
                methods_text = ', '.join(current_event['methods'])
                cv2.putText(frame, status + ' ' + methods_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                status = '✗ ÉTERN. INCORRECT'
                color = (0, 0, 255)
                cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            cv2.putText(frame, 'En attente...', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, f'Frame: {frame_id}', (frame_width-180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        out.write(frame)

        if args.show:
            cv2.imshow('Real-time sneeze tester', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save events and frame logs
    events = detect_sneeze_events(all_frames_data)
    events_path = os.path.join(args.outdir, base + '_sneeze_events.json')
    frame_log_path = os.path.join(args.outdir, base + '_frame_log.json')
    import json

    # Helper to convert numpy / mediapipe numeric types into native Python types
    def _to_serializable(o):
        try:
            import numpy as _np
        except Exception:
            _np = None
        # numpy scalar
        if _np is not None and isinstance(o, (_np.integer,)):
            return int(o)
        if _np is not None and isinstance(o, (_np.floating,)):
            return float(o)
        if _np is not None and isinstance(o, (_np.bool_ ,)):
            return bool(o)
        # numpy arrays
        if _np is not None and isinstance(o, _np.ndarray):
            return o.tolist()
        # Mediapipe landmark objects sometimes contain numpy types inside; try .tolist
        if hasattr(o, 'tolist') and not isinstance(o, (str, bytes)):
            try:
                return o.tolist()
            except Exception:
                pass
        # Fallback: try to cast to native python via int/float/str
        try:
            return int(o)
        except Exception:
            pass
        try:
            return float(o)
        except Exception:
            pass
        try:
            return str(o)
        except Exception:
            return None

    with open(events_path, 'w', encoding='utf-8') as f:
        json.dump({'video': str(src), 'events': events}, f, indent=2, ensure_ascii=False, default=_to_serializable)
    with open(frame_log_path, 'w', encoding='utf-8') as f:
        json.dump({'frame_logs': all_frames_data}, f, indent=2, ensure_ascii=False, default=_to_serializable)

    print('Saved:', out_path, events_path, frame_log_path)


if __name__ == '__main__':
    main()
