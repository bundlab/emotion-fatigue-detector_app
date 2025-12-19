# app.py
import streamlit as st
import cv2
import numpy as np
from fer import FER
import mediapipe as mp
import time

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Emotion + Fatigue Detector", layout="centered")
st.title("Real-Time Emotion & Drowsiness Detector")
st.markdown("Built by **Abdullahi Bundi** • Powered by MediaPipe + FER • 2025")

# -----------------------------
# Initialize Detectors
# -----------------------------
@st.cache_resource
def load_detectors():
    emo_detector = FER(mtcnn=True, min_face_size=50)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return emo_detector, face_mesh

emotion_detector, face_mesh = load_detectors()

# -----------------------------
# Fatigue Variables
# -----------------------------
if 'blink_count' not in st.session_state:
    st.session_state.blink_count = 0
if 'last_blink' not in st.session_state:
    st.session_state.last_blink = time.time()

EAR_THRESHOLD = 0.22
FRAME_SKIP = 2  # Process every 2nd frame for speed

def calculate_EAR(eye):
    """Compute Eye Aspect Ratio (EAR) for blink detection"""
    return (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / \
           (2.0 * np.linalg.norm(eye[0] - eye[3]))

# -----------------------------
# Streamlit Layout
# -----------------------------
col1, col2 = st.columns([3, 1])
frame_window = col1.empty()
stats = col2.empty()

# -----------------------------
# Stop Webcam Button
# -----------------------------
stop_camera = st.button("Stop Webcam")
if stop_camera:
    st.warning("Webcam stopped!")
    st.stop()  # safely exit Streamlit script

# -----------------------------
# OpenCV Video Capture
# -----------------------------
cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Cannot access webcam")
        break

    frame_count += 1

    # Resize for faster FER processing
    small_frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # Emotion Detection (FER)
    # -----------------------------
    emo_results = []
    if frame_count % FRAME_SKIP == 0:
        try:
            emo_results = emotion_detector.detect_emotions(rgb_frame)
        except Exception as e:
            st.warning(f"FER skipped for this frame: {e}")

    emotion = "Neutral"
    confidence = 0.0
    if emo_results:
        top_emotion = emo_results[0]["emotions"]
        emotion = max(top_emotion, key=top_emotion.get)
        confidence = top_emotion[emotion]

    # -----------------------------
    # Fatigue / Blink Detection (MediaPipe)
    # -----------------------------
    blink_status = "👀 Normal"
    mesh_results = face_mesh.process(rgb_frame)
    if mesh_results.multi_face_landmarks:
        landmarks = mesh_results.multi_face_landmarks[0].landmark
        left_eye = np.array([[landmarks[p].x, landmarks[p].y] for p in [33, 160, 158, 133, 153, 144]])
        ear = calculate_EAR(left_eye)

        if ear < EAR_THRESHOLD:
            blink_status = "Blink detected"
            st.session_state.blink_count += 1
            st.session_state.last_blink = time.time()
        elif time.time() - st.session_state.last_blink > 6:
            blink_status = "Possible fatigue 😴"

    # -----------------------------
    # Overlay Text on Frame
    # -----------------------------
    cv2.putText(frame, f"{emotion} ({confidence:.2f})", (10, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 3)
    cv2.putText(frame, blink_status, (10, 100),
                cv2.FONT_HERSHEY_DUPLEX, 1.2,
                (0, 0, 255) if "fatigue" in blink_status.lower() else (255, 255, 0), 3)

    # Convert BGR to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame_rgb)

    # -----------------------------
    # Live Stats
    # -----------------------------
    stats.markdown(f"""
    ### Live Stats
    - **Emotion**  {emotion}
    - **Confidence** {confidence:.1%}
    - **Blink Status** {blink_status}
    - **Total Blinks** {st.session_state.blink_count}
    """)

# Release webcam when done
cap.release()
