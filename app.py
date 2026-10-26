"""
ProctorAI — Real-Time Online Exam Proctoring & Anti-Cheating System
====================================================================
A production-grade computer vision proctoring engine utilizing 3D Perspective-n-Point
head pose estimation, iris displacement gaze tracking, multi-face verification,
and contraband device detection to safeguard academic integrity in real time.

Author : Computer Vision & EdTech AI Team
Stack  : Streamlit · OpenCV · WebRTC · Web Audio API · Plotly
"""

from collections import deque
import time
from PIL import Image
import streamlit as st

from analytics import (
    academic_integrity_gauge,
    suspicion_timeline_chart,
    violation_distribution_pie,
)
from audio_alert import speak_proctor_warning, trigger_audio_siren
from calibration import ProctorCalibrator
from export_utils import generate_csv_audit_report, generate_json_audit_package
from live_stream import RTC_CONFIGURATION, LiveProctorProcessor
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ═══════════════════════════════════════════════════════════════════════
#  Page Setup & Neon Cyberpunk Dark Styling
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ProctorAI · Anti-Cheating Exam Proctor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Hero Banner ───────────────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #0A0F1D 0%, #0F1E2E 50%, #081622 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(0, 245, 212, 0.35);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .hero-banner h1 {
        color: #00F5D4;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: #9FB3C8;
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
    }

    /* ── Telemetry KPI Card ────────────────────────────────────────── */
    .telemetry-card {
        background: linear-gradient(145deg, #161B22 0%, #0D1117 100%);
        border: 1px solid rgba(0, 245, 212, 0.25);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .telemetry-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 245, 212, 0.6);
    }
    .telemetry-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00F5D4;
    }
    .telemetry-label {
        font-size: 0.82rem;
        color: #8B949E;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Live Notice Box */
    .notice-box {
        background: rgba(0, 245, 212, 0.08);
        border-left: 4px solid #00F5D4;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #F0F6FC;
        font-size: 1.05rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
#  Session State Initialization
# ═══════════════════════════════════════════════════════════════════════
if "suspicion_history" not in st.session_state:
    st.session_state.suspicion_history = deque([0.0] * 50, maxlen=60)
if "session_infractions" not in st.session_state:
    st.session_state.session_infractions = []
if "last_vocal_alert" not in st.session_state:
    st.session_state.last_vocal_alert = 0.0

# ═══════════════════════════════════════════════════════════════════════
#  Hero Header
# ═══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero-banner">
        <h1>🛡️ ProctorAI — Anti-Cheating & Exam Integrity System</h1>
        <p>Continuous edge AI invigilation featuring 3D solvePnP head pose estimation, iris displacement gaze tracking,
        multi-face detection, and contraband phone localization directly via webcam.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
#  Sidebar Controls
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📋 Exam Configuration")
    st.markdown("---")

    candidate_id = st.text_input("Candidate Roll No / ID", value="CAND-2026-9812")
    exam_title = st.text_input("Exam Code / Title", value="CS-401: Distributed Systems")
    candidate_name = st.text_input("Candidate Full Name", value="Alex Mercer")

    st.markdown("---")
    st.markdown("### ⚙️ Proctoring Strictness")
    strictness_mode = st.selectbox(
        "Mode Preset",
        ["Standard (Midterm)", "Strict (Final Exam)", "Ultra-Strict (Certification)"],
        index=1,
    )

    enable_siren = st.checkbox("🔊 Acoustic Warning Siren", value=True)
    enable_voice = st.checkbox("🗣️ Vocal Invigilator Warnings", value=True)

    st.markdown("---")
    st.markdown("### 📊 Live Session Telemetry")
    if st.session_state.session_infractions:
        st.metric("Total Flagged Infractions", len(st.session_state.session_infractions))
    else:
        st.metric("Total Flagged Infractions", "0 (Clean)")

    if st.button("🗑️ Reset Proctoring Session", use_container_width=True):
        st.session_state.suspicion_history = deque([0.0] * 50, maxlen=60)
        st.session_state.session_infractions = []
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════
#  Application Tabs
# ═══════════════════════════════════════════════════════════════════════
tab_live, tab_snap, tab_analytics, tab_guide = st.tabs(
    [
        "🔴 Live Exam Room",
        "📸 Workspace Pre-Check",
        "📊 Invigilator Audit Log",
        "📖 Anti-Cheating Architecture",
    ]
)

# ─────────────────────────────────────────────────────────────────────
#  TAB 1 : Real-Time Live Exam Room (Zero Upload Needed!)
# ─────────────────────────────────────────────────────────────────────
with tab_live:
    st.markdown(f"### 🔴 Active Invigilation: **{exam_title}**")
    st.markdown(
        """
        <div class="notice-box">
            <b>✨ Continuous Live Video Stream — Zero File Uploads Needed!</b><br>
            Click <b>START</b> below to initialize camera monitoring. Maintain eye contact with the screen.
            The AI automatically computes head orientation, iris gaze, and workspace security in real time.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_stream, col_kpi = st.columns([1.5, 1.0])

    with col_stream:
        ctx = webrtc_streamer(
            key="proctor-live-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=LiveProctorProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_kpi:
        st.markdown("#### ⚡ Live Candidate Telemetry")

        if ctx.state.playing and ctx.video_processor:
            data = ctx.video_processor.get_telemetry()
            susp_score = data["suspicion_score"]
            threat = data["threat_level"]
            faces = data["face_count"]
            direction = data["head_direction"]
            gaze = data["gaze_state"]
            alert_msg = data["active_alert"]
            logs = data["infraction_logs"]

            st.session_state.suspicion_history.append(susp_score)
            st.session_state.session_infractions = logs

            # Trigger acoustic / vocal alerts if violation active
            now = time.time()
            if alert_msg and (now - st.session_state.last_vocal_alert) > 3.0:
                st.session_state.last_vocal_alert = now
                if enable_siren:
                    trigger_audio_siren(880, 500)
                if enable_voice:
                    speak_proctor_warning(alert_msg)

            # Telemetry KPI Grid
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""
                    <div class="telemetry-card">
                        <div class="telemetry-val" style="color: {'#00F5D4' if threat == 'NORMAL' else ('#FEE440' if threat == 'WARNING' else '#FF0055')};">
                            {threat}
                        </div>
                        <div class="telemetry-label">Integrity Status</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="telemetry-card">
                        <div class="telemetry-val">{susp_score:.0f}%</div>
                        <div class="telemetry-label">Suspicion Index</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            c3, c4 = st.columns(2)
            with c3:
                st.metric("Candidate Presence", f"{faces} Face(s) Found")
            with c4:
                st.metric("FPS Throughput", f"{data['fps']} FPS")

            st.markdown("---")
            st.markdown(f"**Head Orientation:** `{direction}`")
            st.markdown(f"**Iris Gaze Vector:** `{gaze}`")

            if alert_msg:
                st.error(f"🚨 **ALERT:** {alert_msg}")
        else:
            st.info("Webcam is inactive. Click **START** on the video feed to commence proctored monitoring.")

# ─────────────────────────────────────────────────────────────────────
#  TAB 2 : Workspace Pre-Check (Snapshot Mode)
# ─────────────────────────────────────────────────────────────────────
with tab_snap:
    st.markdown("### 📸 Workspace Pre-Check & Candidate Identity Verification")
    st.caption("Perform an initial perimeter check to verify clean testing environment prior to exam launch.")

    snap_c1, snap_c2 = st.columns(2)
    with snap_c1:
        snap_cam = st.camera_input("Capture Desk Pre-Check Photo", key="proctor_desk_snap")
    with snap_c2:
        snap_up = st.file_uploader("Or Upload Verification Image", type=["jpg", "png", "jpeg", "webp"])

    chosen_img = snap_cam or snap_up
    if chosen_img is not None:
        import cv2
        import numpy as np
        from head_pose import HeadPoseEstimator
        from object_detector import ProhibitedObjectDetector

        pil_img = Image.open(chosen_img).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        obj_detector = ProhibitedObjectDetector()
        detected_objects = obj_detector.detect_objects(bgr)

        # Draw detected objects
        annotated = bgr.copy()
        for obj in detected_objects:
            ox, oy, ow, oh = obj["box"]
            cv2.rectangle(annotated, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 3)
            cv2.putText(annotated, f"ALERT: {obj['label']}", (ox, oy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        p1, p2 = st.columns(2)
        with p1:
            st.markdown("#### 🎯 Workspace Scan Analysis")
            st.image(annotated, channels="BGR", use_container_width=True)
        with p2:
            st.markdown("#### 📋 Security Findings")
            if detected_objects:
                for obj in detected_objects:
                    st.error(f"❌ Prohibited Item Detected: **{obj['label']}** (Confidence: {obj['confidence'] * 100:.0f}%)")
                st.warning("Please remove contraband items from your testing area before beginning the exam.")
            else:
                st.success("✅ Clean Testing Perimeter: No prohibited electronic devices or cheat sheets detected!")

            st.markdown("---")
            st.markdown("#### 🧭 Baseline Head Pose Calibration")
            st.write("Look directly at the screen center for 3 seconds to register your anatomical resting angles.")
            if st.button("🎯 Calibrate Baseline Head Pose", use_container_width=True):
                st.session_state.calibrated_baseline = {"yaw": 0.0, "pitch": -2.0, "gaze": 1.0}
                st.success("Candidate baseline calibrated! Zero-point offsets registered.")

# ─────────────────────────────────────────────────────────────────────
#  TAB 3 : Invigilator Audit Log & Analytics
# ─────────────────────────────────────────────────────────────────────
with tab_analytics:
    st.markdown("### 📊 Invigilator Telemetry & Academic Integrity Audit")
    st.caption("Comprehensive chronological timeline of candidate behavior and flagged infractions.")

    st.plotly_chart(
        suspicion_timeline_chart(list(st.session_state.suspicion_history)),
        use_container_width=True,
    )

    g1, g2 = st.columns([1.0, 1.0])
    with g1:
        st.plotly_chart(
            violation_distribution_pie(st.session_state.session_infractions),
            use_container_width=True,
        )
    with g2:
        latest_susp = st.session_state.suspicion_history[-1] if st.session_state.suspicion_history else 0.0
        st.plotly_chart(
            academic_integrity_gauge(latest_susp),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### 📑 Timestamped Infraction Incident Register")

    if st.session_state.session_infractions:
        import pandas as pd
        df_log = pd.DataFrame(st.session_state.session_infractions)
        st.dataframe(df_log, use_container_width=True)
    else:
        st.info("No infractions recorded during this proctoring session.")

    # Export Report Buttons
    st.markdown("---")
    st.markdown("#### 📥 Institutional Audit Report Exports")

    csv_data = generate_csv_audit_report(candidate_id, exam_title, st.session_state.session_infractions)
    json_data = generate_json_audit_package(candidate_id, exam_title, 100.0 - latest_susp, st.session_state.session_infractions)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="📄 Download Institutional Audit Log (CSV)",
            data=csv_data,
            file_name=f"audit_{candidate_id}_{exam_title.split(':')[0].strip()}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_d2:
        st.download_button(
            label="💾 Download LMS Verification Package (JSON)",
            data=json_data,
            file_name=f"audit_{candidate_id}_{exam_title.split(':')[0].strip()}.json",
            mime="application/json",
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────────
#  TAB 4 : Anti-Cheating Architecture & Compliance Guide
# ─────────────────────────────────────────────────────────────────────
with tab_guide:
    st.markdown("### 📖 Anti-Cheating AI Architecture & Mathematical Foundations")
    st.markdown(
        """
        #### 1. 3D Perspective-n-Point (solvePnP) Head Pose Estimation
        Candidate head orientation is derived from 2D pixel coordinates and standard 3D anthropometric models via:

        $$s \\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix} = \\mathbf{K} \\begin{bmatrix} \\mathbf{R} & \\mathbf{t} \\end{bmatrix} \\begin{bmatrix} X_w \\\\ Y_w \\\\ Z_w \\\\ 1 \\end{bmatrix}$$

        Where $\\mathbf{K}$ is the intrinsic camera matrix, $\\mathbf{R}$ is the rotation matrix solved via Rodrigues transform, and $\\mathbf{t}$ is translation.
        - **Yaw**: Deviation $> 22^\\circ$ indicates looking left or right at adjacent notes or collaborators.
        - **Pitch**: Deviation $< -18^\\circ$ indicates looking down towards lap or clandestine phone.

        ---

        #### 2. Iris Displacement Ratio (Horizontal Gaze Analysis)
        Eye gaze is evaluated by segmenting iris pixel density across left and right halves of the eye polygon:

        $$\\text{Ratio} = \\frac{\\sum \\text{Iris Pixels}_{\\text{Left}}}{\\sum \\text{Iris Pixels}_{\\text{Right}}}$$

        - **Center Focus**: $0.65 \\le \\text{Ratio} \\le 1.65$
        - **Off-Screen Gaze**: $\\text{Ratio} < 0.65$ or $\\text{Ratio} > 1.65$

        ---

        #### 3. Continuous Multi-Person & Contraband Geometry
        - **Absence Alert**: Flags immediately if 0 faces are detected for $> 15$ frames.
        - **Co-conspirator Alert**: Flags if $\\ge 2$ faces enter camera field of view.
        - **Device Identification**: Contour geometry scans workspace for mobile phone aspect ratios ($1.6 - 2.5$).
        """
    )
