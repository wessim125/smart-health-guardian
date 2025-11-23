import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import os
import json
import math
from collections import deque
import tempfile
from pathlib import Path
import time

# ==============================
# CONFIGURATION PAGE STREAMLIT
# ==============================
st.set_page_config(
    page_title="Smart Health Guardian",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CSS PERSONNALISÉ MODERNE
# ==============================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Background with Gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        box-shadow: 4px 0 15px rgba(0,0,0,0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Main Content Container */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        backdrop-filter: blur(10px);
    }
    
    /* Header Gradient Card */
    .header-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .header-card h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-card p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(17, 153, 142, 0.4);
    }
    
    /* Info/Success/Error Boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid;
        padding: 1rem;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Success Box */
    [data-baseweb="notification"][kind="success"] {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-left: 5px solid #0d7a6f;
    }
    
    /* Info Box */
    [data-baseweb="notification"][kind="info"] {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-left: 5px solid #3b8bc9;
    }
    
    /* Warning Box */
    [data-baseweb="notification"][kind="warning"] {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border-left: 5px solid #f85c7f;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #e0e7ff 0%, #c3cfe2 100%);
        transform: translateX(5px);
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Radio Buttons */
    .stRadio > div {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #764ba2;
        background: #f8f9ff;
    }
    
    /* Video Player */
    video {
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Text Input */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e7ff;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e0e7ff;
    }
    
    /* Stats Card */
    .stats-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border-left: 5px solid #667eea;
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Risk Badge */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(250, 112, 154, 0.4);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(235, 51, 73, 0.4);
    }
    
    /* Sidebar Icons */
    .sidebar-icon {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.1rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .sidebar-icon:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(5px);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        box-shadow: 0 -5px 20px rgba(0,0,0,0.1);
    }
    
    /* Chart */
    .stPlotlyChart {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        background: white;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# INITIALISATION SESSION STATE
# ==============================
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None

# ==============================
# CONFIGURATION GLOBALE
# ==============================
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# CLASSE POUR DÉTECTION ÉTERNUEMENTS
# ==============================
class SneezeDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Paramètres par défaut
        self.WINDOW_SIZE = 12
        self.MOVEMENT_THRESHOLD = 45.0
        self.HEAD_DROP_SUM_THRESHOLD = 0.06
        self.COVERAGE_RATIO_THRESHOLD = 0.6
        self.MIN_LANDMARK_VISIBILITY = 0.35
        self.MIN_FRAMES_BETWEEN_EVENTS = 18
        self.DEPTH_FRONT_THRESHOLD = 0.08
        self.FACE_TURN_THRESHOLD = 0.12
        self.PALM_VEL_THRESHOLD = 0.02
        
        # Détecteurs
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
    
    def calculate_angle(self, p1, p2, p3):
        a = np.array(p1) - np.array(p2)
        b = np.array(p3) - np.array(p2)
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        cos_angle = np.dot(a, b) / denom
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle
    
    def dist(self, p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
    
    def norm_dist(self, p1, p2, diag):
        return self.dist(p1, p2) / (diag + 1e-6)
    
    def is_hand_covering(self, nose, mouth, wrist, palm_center, wrist_z, nose_z, hand_closed, diag):
        if palm_center is not None:
            if self.norm_dist(palm_center, nose, diag) < 0.15 or self.norm_dist(palm_center, mouth, diag) < 0.17:
                return True, "Main (paume)"
            if hand_closed and self.norm_dist(palm_center, nose, diag) < 0.22:
                return True, "Main fermée"
        if self.norm_dist(wrist, nose, diag) < 0.18 or self.norm_dist(wrist, mouth, diag) < 0.20:
            return True, "Poignet"
        if wrist_z is not None and nose_z is not None and (nose_z - wrist_z) > self.DEPTH_FRONT_THRESHOLD:
            return True, "Main devant (3D)"
        return False, ""
    
    def is_elbow_covering(self, elbow, nose, angle, diag):
        if self.norm_dist(elbow, nose, diag) < 0.26 and 40 < angle < 165:
            return True, "Coude"
        return False, ""
    
    def is_reliable_pose(self, landmarks):
        try:
            vis = [getattr(lm, 'visibility', 1.0) for lm in landmarks.landmark]
            return np.median(vis) >= self.MIN_LANDMARK_VISIBILITY
        except Exception:
            return True
    
    def coverage_vote(self, frames_buf, diag, peak_index=None):
        n = len(frames_buf)
        if n == 0:
            return 0.0, []
        center = peak_index if peak_index is not None else (n - 1) // 2
        covered_weight = 0.0
        total_weight = 0.0
        methods = []
        
        for i, snap in enumerate(frames_buf):
            if not snap.get('pose') or not self.is_reliable_pose(snap['pose']):
                continue
            wgt = 1.0 - (abs(i - center) / (center + 1e-6))
            wgt = max(0.0, wgt)
            
            cl, ml = self.is_hand_covering(snap['nose'], snap['mouth'], snap['wrist_l'], 
                                          snap['palm_l'], snap['wrist_l_z'], snap['nose_z'], 
                                          snap['left_closed'], diag)
            cr, mr = self.is_hand_covering(snap['nose'], snap['mouth'], snap['wrist_r'], 
                                          snap['palm_r'], snap['wrist_r_z'], snap['nose_z'], 
                                          snap['right_closed'], diag)
            cel, _ = self.is_elbow_covering(snap['elbow_l'], snap['nose'], snap['elbow_l_angle'], diag)
            cer, _ = self.is_elbow_covering(snap['elbow_r'], snap['nose'], snap['elbow_r_angle'], diag)
            
            covered = cl or cr or cel or cer
            if covered:
                total_weight += wgt
                covered_weight += wgt
                if cl and ml: methods.append(ml)
                if cr and mr: methods.append(mr)
                if cel: methods.append('Coude gauche')
                if cer: methods.append('Coude droit')
        
        weighted_ratio = (covered_weight / (total_weight + 1e-6)) if total_weight > 0 else 0.0
        
        methods_summary = []
        if methods:
            uniq = {}
            for m in methods:
                uniq[m] = uniq.get(m, 0) + 1
            methods_summary = sorted(uniq.items(), key=lambda x: x[1], reverse=True)
            methods_summary = [m for m, _ in methods_summary[:3]]
        
        return weighted_ratio, methods_summary
    
    def process_video(self, video_path, output_path, progress_bar=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        sneeze_events = []
        movement_buffer = deque(maxlen=self.WINDOW_SIZE)
        head_drop_buffer = deque(maxlen=self.WINDOW_SIZE)
        frames_buffer = deque(maxlen=self.WINDOW_SIZE)
        prev_nose_y = None
        prev_landmarks = None
        last_event_frame = -999
        
        frame_id = 0
        diag = math.hypot(w, h)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            if progress_bar and total_frames > 0:
                progress_bar.progress(frame_id / total_frames)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose_detector.process(rgb)
            hand_results = self.hands_detector.process(rgb)
            
            display = frame.copy()
            movement = 0.0
            head_drop = 0.0
            
            if pose_results.pose_landmarks:
                lm = pose_results.pose_landmarks.landmark
                hh, ww = frame.shape[:2]
                
                nose = [lm[0].x * ww, lm[0].y * hh]
                nose_z = lm[0].z
                mouth = [lm[10].x * ww, lm[10].y * hh]
                l_wrist = [lm[15].x * ww, lm[15].y * hh]
                l_wrist_z = lm[15].z
                r_wrist = [lm[16].x * ww, lm[16].y * hh]
                r_wrist_z = lm[16].z
                l_elbow = [lm[13].x * ww, lm[13].y * hh]
                r_elbow = [lm[14].x * ww, lm[14].y * hh]
                l_shoulder = [lm[11].x * ww, lm[11].y * hh]
                r_shoulder = [lm[12].x * ww, lm[12].y * hh]
                
                self.mp_drawing.draw_landmarks(display, pose_results.pose_landmarks, 
                                              self.mp_pose.POSE_CONNECTIONS)
                
                left_palm = right_palm = None
                left_closed = right_closed = False
                
                if hand_results.multi_hand_landmarks:
                    for hand_lms, handedness in zip(hand_results.multi_hand_landmarks, 
                                                    hand_results.multi_handedness):
                        label = handedness.classification[0].label
                        wrist_pt = hand_lms.landmark[0]
                        mcp_pt = hand_lms.landmark[9]
                        palm_x = (wrist_pt.x * ww + mcp_pt.x * ww) / 2
                        palm_y = (wrist_pt.y * hh + mcp_pt.y * hh) / 2
                        palm = np.array([palm_x, palm_y])
                        closed = sum(1 for i in [8,12,16,20] 
                                   if hand_lms.landmark[i].y > hand_lms.landmark[i-2].y) >= 3
                        
                        if label == 'Left':
                            left_palm, left_closed = palm, closed
                        else:
                            right_palm, right_closed = palm, closed
                        
                        self.mp_drawing.draw_landmarks(display, hand_lms, 
                                                      self.mp_hands.HAND_CONNECTIONS)
                
                if prev_landmarks:
                    for idx in [13,14,15,16]:
                        curr = lm[idx]
                        prev = prev_landmarks.landmark[idx]
                        movement += abs(curr.x - prev.x) + abs(curr.y - prev.y)
                
                movement_buffer.append(movement * 1000.0)
                
                if prev_nose_y is not None:
                    head_drop = lm[0].y - prev_nose_y
                head_drop_buffer.append(head_drop)
                prev_nose_y = lm[0].y
                
                elbow_l_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
                elbow_r_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
                
                snapshot = {
                    'pose': pose_results.pose_landmarks,
                    'nose': np.array(nose),
                    'mouth': np.array(mouth),
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
                frames_buffer.append(snapshot)
                
                if len(movement_buffer) == movement_buffer.maxlen:
                    peak = max(movement_buffer)
                    head_drop_sum = sum(head_drop_buffer)
                    
                    if peak > self.MOVEMENT_THRESHOLD and head_drop_sum > self.HEAD_DROP_SUM_THRESHOLD:
                        peak_idx = int(np.argmax(np.array(movement_buffer)))
                        coverage_ratio, methods_summary = self.coverage_vote(
                            list(frames_buffer), diag, peak_index=peak_idx)
                        is_covered = coverage_ratio >= self.COVERAGE_RATIO_THRESHOLD
                        
                        if frame_id - last_event_frame > self.MIN_FRAMES_BETWEEN_EVENTS:
                            sneeze_events.append({
                                'frame': frame_id,
                                'covered': is_covered,
                                'coverage_ratio': float(coverage_ratio),
                                'methods': methods_summary,
                                'peak_movement': float(peak),
                                'head_drop_sum': float(head_drop_sum)
                            })
                            last_event_frame = frame_id
                            # Dessiner un label simple indiquant l'état d'éternuement
                            label = 'ÉTERNUEMENT CORRECT' if is_covered else 'ÉTERNUEMENT NON COUVERT'
                            color = (0, 255, 0) if is_covered else (0, 0, 255)
                            cv2.putText(display, label, (20, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Afficher numéro de frame et écrire la frame annotée
            cv2.putText(display, f"Frame: {frame_id}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            out.write(display)
            # mettre à jour prev_landmarks pour le calcul du mouvement
            prev_landmarks = pose_results.pose_landmarks
        
        cap.release()
        out.release()

        # Compile sneeze statistics
        total_sneezes = len(sneeze_events)
        correct_sneezes = sum(1 for e in sneeze_events if e.get('covered'))
        rate = (correct_sneezes / total_sneezes * 100) if total_sneezes else 0.0

        # Save events JSON
        events_path = output_path.replace('.mp4', '_events.json')
        try:
            with open(events_path, 'w', encoding='utf-8') as f:
                json.dump({'video': video_path, 'events': sneeze_events}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return {
            'frames': frame_id,
            'sneezes': total_sneezes,
            'correct': correct_sneezes,
            'rate': rate,
            'events': sneeze_events,
            'events_path': events_path
        }

# ==============================
# CLASSE POUR DÉTECTION MASQUES
# ==============================
class MaskDetector:
    def __init__(self, model_path, confidence=0.5):
        if not os.path.exists(model_path):
            st.error(f"⚠ Modèle manquant : {model_path}")
            self.model = None
        else:
            self.model = YOLO(model_path)
        
        self.classes = ['with_mask', 'without_mask', 'incorrect_mask']
        self.confidence = confidence
    
    def process_video(self, video_path, output_path, progress_bar=None):
        if self.model is None:
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_id = 0
        mask_stats = {'with_mask': 0, 'without_mask': 0, 'incorrect_mask': 0}
        
        results = self.model(video_path, stream=True, conf=self.confidence, iou=0.45)
        
        for result in results:
            frame_id += 1
            if progress_bar and total_frames > 0:
                progress_bar.progress(min(frame_id / total_frames, 1.0))
            
            img = result.orig_img.copy()
            
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = box.conf[0].item()
                label = f"{self.classes[cls_id]} {conf:.2f}"
                
                mask_stats[self.classes[cls_id]] += 1
                
                color = (0, 255, 0)   # vert = masque correct
                if cls_id == 1: color = (0, 0, 255)   # rouge = sans masque
                if cls_id == 2: color = (255, 255, 0) # jaune = masque mal porté
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(img)
        
        cap.release()
        out.release()
        
        return {
            'frames': frame_id,
            'mask_stats': mask_stats,
            'compliance_rate': (mask_stats['with_mask'] / 
                              (sum(mask_stats.values()) + 1e-6)) * 100
        }

# ==============================
# INTERFACE STREAMLIT PRINCIPALE
# ==============================
def main():
    # En-tête moderne avec animation
    st.markdown("""
    <div class='header-card'>
        <h1>🏥 Smart Health Guardian</h1>
        <p>Système Intelligent de Surveillance Sanitaire | Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar avec design moderne
    with st.sidebar:
        st.markdown("<div style='text-align: center; margin-bottom: 2rem;'>", unsafe_allow_html=True)
        st.image("https://img.icons8.com/fluency/96/health-graph.png", width=100)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Configuration")
        
        module = st.selectbox(
            "📋 Module d'Analyse",
            ["🤧 Détection d'Éternuements", 
             "👥 Distanciation Sociale", 
             "😷 Détection de Masques"],
            help="Choisissez le type d'analyse à effectuer"
        )
        
        st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
        
        # Configuration selon le module
        if "Éternuements" in module:
            st.markdown("#### 🎯 Paramètres d'Éternuements")
            window_size = st.slider("🔲 Taille fenêtre d'analyse", 8, 20, 12, help="Nombre de frames à analyser")
            movement_threshold = st.slider("📊 Seuil de mouvement", 20.0, 80.0, 45.0, help="Sensibilité de détection")
            coverage_threshold = st.slider("✋ Seuil de couverture", 0.3, 0.9, 0.6, help="Niveau de couverture requis")
            
        elif "Distanciation" in module:
            st.markdown("#### 🎯 Paramètres de Distanciation")
            distance_threshold = st.slider("📏 Distance minimale (px)", 50, 200, 100, help="Distance de sécurité en pixels")
            confidence = st.slider("🎯 Confiance de détection", 0.3, 0.9, 0.45, help="Précision du modèle")
            
        elif "Masques" in module:
            st.markdown("#### 🎯 Paramètres de Masques")
            mask_model_path = st.text_input(
                "📂 Chemin du modèle",
                "C:/Users/hichr/smart-health-guardian/runs/mask/yolov8_mask_local/weights/best.pt",
                help="Chemin vers le modèle YOLO entraîné"
            )
            mask_confidence = st.slider("🎯 Confiance de détection", 0.3, 0.9, 0.5, help="Précision du modèle")
        
        st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.info("💡 **Conseil**: Uploadez une vidéo claire pour de meilleurs résultats")
        
        st.markdown("<div style='margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;'>", unsafe_allow_html=True)
        st.markdown("**📈 Statistiques**")
        st.markdown(f"- Module: {module.split()[1]}")
        st.markdown(f"- Version: 2.0")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Zone principale avec design amélioré
    st.markdown(f"### 📹 {module}")
    
    # Section Upload avec style moderne
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        input_type = st.radio(
            "Type d'entrée",
            ["📁 Fichier vidéo", "📹 Webcam"],
            horizontal=True
        )
        
        video_file = None
        
        if input_type == "📁 Fichier vidéo":
            video_file = st.file_uploader(
                "Glissez-déposez votre vidéo ici",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Formats acceptés: MP4, AVI, MOV, MKV"
            )
            
            if video_file:
                st.success("✅ Vidéo chargée avec succès!")
                st.markdown("#### 📹 Aperçu de la vidéo d'origine")
                st.video(video_file)
    
    with col_info:
        if video_file:
            file_details = {
                "📄 Nom": video_file.name,
                "📦 Taille": f"{video_file.size / (1024*1024):.2f} MB",
                "🎬 Type": video_file.type
            }
            st.markdown("#### 📊 Informations")
            for key, value in file_details.items():
                st.markdown(f"**{key}:** {value}")
        else:
            st.info("ℹ️ Aucune vidéo sélectionnée")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Bouton de traitement avec style moderne
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button("🚀 LANCER L'ANALYSE", type="primary", use_container_width=True)
    
    if analyze_button:
        if video_file is None:
            st.error("⚠️ Veuillez sélectionner une vidéo avant de lancer l'analyse")
            return
        
        # Préparer la vidéo
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        video_path = tfile.name
        
        # Nom du fichier de sortie
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"{module.split()[1].lower()}_{timestamp}.mp4"
        output_path = os.path.join(RESULTS_DIR, output_filename)
        
        # Animation de chargement
        with st.spinner('🔄 Analyse en cours...'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Container pour les résultats
            st.markdown("<hr>", unsafe_allow_html=True)
            results_container = st.container()
            
            try:
                if "Éternuements" in module:
                    status_text.info("🔍 Analyse des éternuements en cours...")
                    detector = SneezeDetector()
                    detector.WINDOW_SIZE = window_size
                    detector.MOVEMENT_THRESHOLD = movement_threshold
                    detector.COVERAGE_RATIO_THRESHOLD = coverage_threshold
                    
                    results = detector.process_video(video_path, output_path, progress_bar)
                    
                    if results:
                        progress_bar.empty()
                        status_text.success("✅ Analyse terminée avec succès!")
                        
                        with results_container:
                            # Badge de résultats
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;'>
                                <h2 style='color: white; margin: 0;'>🎥 RÉSULTATS DE L'ANALYSE</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Vidéo analysée
                            st.video(output_path)
                            
                            st.markdown("<hr>", unsafe_allow_html=True)
                            
                            # Statistiques principales avec design moderne
                            st.markdown("### 📊 Statistiques Globales")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown("""
                                <div class='stats-card'>
                                    <p style='font-size: 0.9rem; color: #666; margin: 0;'>🎬 FRAMES</p>
                                    <h2 style='color: #667eea; margin: 0.5rem 0;'>{}</h2>
                                </div>
                                """.format(results['frames']), unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("""
                                <div class='stats-card' style='border-left-color: #fa709a;'>
                                    <p style='font-size: 0.9rem; color: #666; margin: 0;'>🤧 ÉTERNUEMENTS</p>
                                    <h2 style='color: #fa709a; margin: 0.5rem 0;'>{}</h2>
                                </div>
                                """.format(results['sneezes']), unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown("""
                                <div class='stats-card' style='border-left-color: #11998e;'>
                                    <p style='font-size: 0.9rem; color: #666; margin: 0;'>✅ CORRECTS</p>
                                    <h2 style='color: #11998e; margin: 0.5rem 0;'>{}</h2>
                                </div>
                                """.format(results['correct']), unsafe_allow_html=True)
                            
                            with col4:
                                st.markdown("""
                                <div class='stats-card' style='border-left-color: #764ba2;'>
                                    <p style='font-size: 0.9rem; color: #666; margin: 0;'>📈 CONFORMITÉ</p>
                                    <h2 style='color: #764ba2; margin: 0.5rem 0;'>{:.1f}%</h2>
                                </div>
                                """.format(results['rate']), unsafe_allow_html=True)
                            
                            if results['sneezes'] > 0:
                                st.markdown("<hr>", unsafe_allow_html=True)
                                st.markdown("### 📋 Détails des Événements Détectés")
                                
                                for i, event in enumerate(results['events'], 1):
                                    status_icon = "✅" if event['covered'] else "❌"
                                    status_text_label = "CONFORME" if event['covered'] else "NON CONFORME"
                                    border_color = "#11998e" if event['covered'] else "#eb3349"
                                    
                                    with st.expander(f"{status_icon} **Éternuement #{i}** - Frame {event['frame']} - Status: {status_text_label}", expanded=(i==1)):
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.metric("📊 Taux de couverture", f"{event['coverage_ratio']*100:.1f}%")
                                            st.metric("💨 Mouvement pic", f"{event['peak_movement']:.2f}")
                                        with col_b:
                                            st.metric("📉 Chute de tête", f"{event['head_drop_sum']:.4f}")
                                            if event['methods']:
                                                st.success(f"**✋ Méthodes:** {', '.join(event['methods'])}")
                            
                            # Section téléchargements avec design moderne
                            st.markdown("<hr>", unsafe_allow_html=True)
                            st.markdown("### 💾 Télécharger les Résultats")
                            col_dl1, col_dl2 = st.columns(2)
                            
                            with col_dl1:
                                with open(output_path, 'rb') as f:
                                    st.download_button(
                                        "⬇️ Vidéo Analysée",
                                        f,
                                        file_name=output_filename,
                                        mime="video/mp4",
                                        use_container_width=True
                                    )
                            
                            with col_dl2:
                                with open(results['events_path'], 'rb') as f:
                                    st.download_button(
                                        "📄 Rapport JSON",
                                        f,
                                        file_name=f"events_{timestamp}.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                
                elif "Distanciation" in module:
                    status_text.info("🔍 Analyse de la distanciation sociale en cours...")
                    detector = SocialDistancingDetector(
                        distance_threshold=distance_threshold,
                        confidence=confidence
                    )
                    
                    results = detector.process_video(video_path, output_path, progress_bar)
                    
                    if results:
                        progress_bar.empty()
                        status_text.success("✅ Analyse terminée avec succès!")
                        
                        with results_container:
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;'>
                                <h2 style='color: white; margin: 0;'>🎥 RÉSULTATS DE L'ANALYSE</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.video(output_path)
                            
                            st.markdown("<hr>", unsafe_allow_html=True)
                            
                            # Statistiques
                            st.markdown("### 📊 Statistiques Globales")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("""
                                <div class='stats-card'>
                                    <p style='font-size: 0.9rem; color: #666; margin: 0;'>🎬 FRAMES ANALYSÉES</p>
                                    <h2 style='color: #667eea; margin: 0.5rem 0;'>{}</h2>
                                </div>
                                """.format(results['frames']), unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("""
                                <div class='stats-card' style='border-left-color: #eb3349;'>
                                    <p style='font-size: 0.9rem; color: #666; margin: 0;'>⚠️ VIOLATIONS TOTALES</p>
                                    <h2 style='color: #eb3349; margin: 0.5rem 0;'>{}</h2>
                                </div>
                                """.format(results['total_violations']), unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown("""
                                <div class='stats-card' style='border-left-color: #fa709a;'>
                                    <p style='font-size: 0.9rem; color: #666; margin: 0;'>📊 VIOLATIONS/FRAME</p>
                                    <h2 style='color: #fa709a; margin: 0.5rem 0;'>{:.2f}</h2>
                                </div>
                                """.format(results['avg_violations_per_frame']), unsafe_allow_html=True)
                            
                            st.markdown("<hr>", unsafe_allow_html=True)
                            
                            # Indicateur de risque stylisé
                            avg_violations = results['avg_violations_per_frame']
                            if avg_violations < 1:
                                risk_level = "FAIBLE"
                                risk_class = "risk-low"
                                risk_emoji = "🟢"
                            elif avg_violations < 3:
                                risk_level = "MOYEN"
                                risk_class = "risk-medium"
                                risk_emoji = "🟡"
                            else:
                                risk_level = "ÉLEVÉ"
                                risk_class = "risk-high"
                                risk_emoji = "🔴"
                            
                            st.markdown(f"""
                            <div style='text-align: center; margin: 2rem 0;'>
                                <h3>Niveau de Risque Sanitaire</h3>
                                <div class='risk-badge {risk_class}'>
                                    {risk_emoji} RISQUE {risk_level}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Barre de progression du risque
                            risk_percentage = min(avg_violations / 5 * 100, 100)
                            st.progress(risk_percentage / 100)
                            
                            # Téléchargement
                            st.markdown("<hr>", unsafe_allow_html=True)
                            st.markdown("### 💾 Télécharger les Résultats")
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    "⬇️ Vidéo Analysée",
                                    f,
                                    file_name=output_filename,
                                    mime="video/mp4",
                                    use_container_width=True
                                )
                
                elif "Masques" in module:
                    status_text.info("🔍 Analyse du port de masques en cours...")
                    detector = MaskDetector(
                        model_path=mask_model_path,
                        confidence=mask_confidence
                    )
                    
                    if detector.model is None:
                        st.error("❌ Impossible de charger le modèle de détection de masques")
                        return
                    
                    results = detector.process_video(video_path, output_path, progress_bar)
                    
                    if results:
                        progress_bar.empty()
                        status_text.success("✅ Analyse terminée avec succès!")
                        
                        with results_container:
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;'>
                                <h2 style='color: white; margin: 0;'>🎥 RÉSULTATS DE L'ANALYSE</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.video(output_path)
                            
                            st.markdown("<hr>", unsafe_allow_html=True)
                            
                            stats = results['mask_stats']
                            total = sum(stats.values())
                            
                            if total > 0:
                                # Statistiques principales
                                st.markdown("### 📊 Statistiques Globales")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.markdown("""
                                    <div class='stats-card'>
                                        <p style='font-size: 0.9rem; color: #666; margin: 0;'>🎬 FRAMES</p>
                                        <h2 style='color: #667eea; margin: 0.5rem 0;'>{}</h2>
                                    </div>
                                    """.format(results['frames']), unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("""
                                    <div class='stats-card' style='border-left-color: #11998e;'>
                                        <p style='font-size: 0.9rem; color: #666; margin: 0;'>✅ AVEC MASQUE</p>
                                        <h2 style='color: #11998e; margin: 0.5rem 0;'>{}</h2>
                                        <p style='color: #999; font-size: 0.85rem;'>{:.1f}%</p>
                                    </div>
                                    """.format(stats['with_mask'], stats['with_mask']/total*100), unsafe_allow_html=True)
                                
                                with col3:
                                    st.markdown("""
                                    <div class='stats-card' style='border-left-color: #eb3349;'>
                                        <p style='font-size: 0.9rem; color: #666; margin: 0;'>❌ SANS MASQUE</p>
                                        <h2 style='color: #eb3349; margin: 0.5rem 0;'>{}</h2>
                                        <p style='color: #999; font-size: 0.85rem;'>{:.1f}%</p>
                                    </div>
                                    """.format(stats['without_mask'], stats['without_mask']/total*100), unsafe_allow_html=True)
                                
                                with col4:
                                    st.markdown("""
                                    <div class='stats-card' style='border-left-color: #fa709a;'>
                                        <p style='font-size: 0.9rem; color: #666; margin: 0;'>⚠️ MAL PORTÉ</p>
                                        <h2 style='color: #fa709a; margin: 0.5rem 0;'>{}</h2>
                                        <p style='color: #999; font-size: 0.85rem;'>{:.1f}%</p>
                                    </div>
                                    """.format(stats['incorrect_mask'], stats['incorrect_mask']/total*100), unsafe_allow_html=True)
                                
                                st.markdown("<hr>", unsafe_allow_html=True)
                                
                                # Taux de conformité avec design moderne
                                compliance = results['compliance_rate']
                                st.markdown(f"""
                                <div style='text-align: center; margin: 2rem 0;'>
                                    <h3>📈 Taux de Conformité Global</h3>
                                    <div style='font-size: 3rem; font-weight: 700; 
                                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                                -webkit-background-clip: text;
                                                -webkit-text-fill-color: transparent;
                                                margin: 1rem 0;'>
                                        {compliance:.1f}%
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.progress(compliance / 100)
                                
                                st.markdown("<hr>", unsafe_allow_html=True)
                                
                                # Graphique de répartition
                                st.markdown("### 📊 Répartition Détaillée des Détections")
                                import pandas as pd
                                df = pd.DataFrame({
                                    'Catégorie': ['✅ Avec masque', '❌ Sans masque', '⚠️ Mal porté'],
                                    'Nombre': [stats['with_mask'], stats['without_mask'], stats['incorrect_mask']],
                                    'Pourcentage': [
                                        f"{stats['with_mask']/total*100:.1f}%",
                                        f"{stats['without_mask']/total*100:.1f}%",
                                        f"{stats['incorrect_mask']/total*100:.1f}%"
                                    ]
                                })
                                
                                # Tableau stylisé
                                st.dataframe(df, use_container_width=True, hide_index=True)
                                
                                # Graphique en barres
                                st.bar_chart(df.set_index('Catégorie')['Nombre'])
                            
                            # Téléchargement
                            st.markdown("<hr>", unsafe_allow_html=True)
                            st.markdown("### 💾 Télécharger les Résultats")
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    "⬇️ Vidéo Analysée",
                                    f,
                                    file_name=output_filename,
                                    mime="video/mp4",
                                    use_container_width=True
                                )
            
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            
            finally:
                progress_bar.empty()
                if os.path.exists(video_path):
                    try:
                        os.unlink(video_path)
                    except:
                        pass
    
    # Pied de page moderne
    st.markdown("""
    <div class='footer'>
        <h3 style='margin: 0; font-size: 1.5rem;'>🏥 Smart Health Guardian</h3>
        <p style='margin: 0.5rem 0; font-size: 1rem;'>Système de Surveillance Sanitaire Intelligent</p>
        <p style='margin: 0.5rem 0; font-size: 0.9rem; opacity: 0.9;'>
            Développé avec ❤️ utilisant MediaPipe, YOLO v8 et Streamlit
        </p>
        <p style='margin: 1rem 0 0 0; font-size: 0.85rem; opacity: 0.8;'>
            © 2024 - Tous droits réservés | Version 2.0
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# CLASSE POUR DISTANCIATION SOCIALE
# ==============================
class SocialDistancingDetector:
    def __init__(self, model_path="yolov8n.pt", distance_threshold=100, confidence=0.45):
        self.model = YOLO(model_path)
        self.distance_threshold = distance_threshold
        self.confidence = confidence
    
    def process_video(self, video_path, output_path, progress_bar=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_id = 0
        total_violations = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            if progress_bar and total_frames > 0:
                progress_bar.progress(frame_id / total_frames)
            
            results = self.model(frame)[0]
            
            persons = []
            person_id = 0
            for det in results.boxes:
                cls = int(det.cls[0])
                conf = float(det.conf[0])
                if cls == 0 and conf >= self.confidence:
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    person_id += 1
                    persons.append({
                        "id": person_id,
                        "bbox": [x1, y1, x2, y2],
                        "center": (cx, cy)
                    })
            
            frame_violations = 0
            n = len(persons)
            for i in range(n):
                for j in range(i + 1, n):
                    (x1, y1) = persons[i]["center"]
                    (x2, y2) = persons[j]["center"]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    if distance < self.distance_threshold:
                        color = (0, 0, 255)
                        alert_text = "RISQUE SECURITE ELEVE"
                        frame_violations += 1
                        # compter la violation globale
                        total_violations += 1
                    else:
                        color = (0, 255, 0)
                        alert_text = ""
                    
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)

            # Dessiner boîtes et centres des personnes
            for person in persons:
                x1, y1, x2, y2 = person["bbox"]
                cx, cy = person["center"]
                pid = person["id"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(frame, f"Person {pid}", (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.putText(frame, f"Frame: {frame_id}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            out.write(frame)

        cap.release()
        out.release()

        return {
            'frames': frame_id,
            'total_violations': total_violations,
            'avg_violations_per_frame': (total_violations / frame_id) if frame_id > 0 else 0
        }

# Entrée principale déplacée en fin de fichier
if __name__ == "__main__":
    main()
