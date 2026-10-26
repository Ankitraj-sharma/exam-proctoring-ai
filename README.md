# ProctorAI - Real-Time AI Online Exam Proctoring & Anti-Cheating System 🛡️👁️

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)

> **Real-time, zero-upload automated online exam invigilation system powered by 3D Perspective-n-Point (`solvePnP`) head pose estimation, iris displacement gaze tracking, multi-face presence verification, and contraband device localization.** Operates entirely client-side with zero manual file uploads—just click **START** on the webcam feed to initiate continuous proctoring!

---

## 🎯 Key Architectural Pillars

1. **Zero-Upload Continuous Webcam Stream:**
   - Powered by **Streamlit WebRTC (`streamlit-webrtc`)** and dual Google STUN server ICE negotiation.
   - Low-latency frames processed at 30+ FPS directly in the browser.
2. **3D Perspective-n-Point Head Pose Estimation:**
   - Projects 3D facial models onto 2D image coordinates to solve Euler rotation angles ($\text{Yaw}, \text{Pitch}, \text{Roll}$).
   - Detects when a student turns to talk to an adjacent person or looks down at a lap/cheat sheet.
3. **Iris Displacement Eye Gaze Tracking:**
   - Isolates eye regions and computes dark iris pixel distribution ratio across left and right halves.
   - Detects persistent off-screen glancing towards external monitors or reference materials.
4. **Continuous Candidate Presence & Multi-Person Guard:**
   - Immediate security flag if $0$ faces detected for $> 15$ frames (candidate left desk).
   - Immediate **CRITICAL** flag if $\ge 2$ faces enter camera perimeter (unauthorized collaborator in room).
5. **Contraband Object & Smartphone Localization:**
   - Edge contour geometry analysis identifying smartphones (aspect ratio $1.6 - 2.5$) and notes/books.
6. **Dynamic Suspicion Index ($0 - 100\%$) & Acoustic Deterrents:**
   - Weighted multi-vector risk engine with exponential decay for rehabilitated focus.
   - Client-side Web Audio API synth sirens and Web Speech API vocal reprimands.
7. **Institutional Audit Trails & LMS Exports:**
   - One-click export of formal CSV Incident Logs and JSON telemetry manifests compatible with Canvas, Blackboard, and Moodle.

---

## 📐 Mathematical Formulation

### 1. 3D Perspective-n-Point Head Pose Projection
Given 3D anthropometric face coordinates $\mathbf{P}_w = [X_w, Y_w, Z_w, 1]^T$ and corresponding 2D image coordinates $\mathbf{p} = [u, v, 1]^T$:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

Where $\mathbf{K}$ is the intrinsic camera matrix:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

Using Rodrigues rotation transform, Euler angles are extracted:
- **Yaw ($\psi$):** $> 22^\circ$ or $< -22^\circ \implies$ Looking Left / Right.
- **Pitch ($\theta$):** $< -18^\circ \implies$ Looking Down (lap / phone).

### 2. Horizontal Iris Gaze Ratio
Eye gaze is quantified by iris pixel mass concentration across the segmented eye contour:

$$\text{Ratio} = \frac{\sum_{(x, y) \in \text{Left}} \mathbb{I}(\text{Iris})}{\sum_{(x, y) \in \text{Right}} \mathbb{I}(\text{Iris})}$$

- **Focused Gaze**: $0.65 \le \text{Ratio} \le 1.65$
- **Off-Screen Glance**: $\text{Ratio} < 0.65$ or $\text{Ratio} > 1.65$

---

## 🛡️ Anti-Cheating Threat Matrix

| Vector | Threshold | Penalty | Severity Level | System Response |
| :--- | :--- | :--- | :--- | :--- |
| **Multiple People** | $\ge 2$ Faces in Frame | $+45$ pts | `CRITICAL` | Red Bounding Box + Siren + Logged |
| **Candidate Absent** | 0 Faces for $\ge 15$ frames | $+35$ pts | `CRITICAL` | Yellow Flash + Acoustic Notice |
| **Contraband Device** | Smartphone / Notes detected | $+40$ pts | `CRITICAL` | Contour Highlight + Reprimand |
| **Head Deviation** | $|\text{Yaw}| > 22^\circ$ or $\text{Pitch} < -18^\circ$ | $+25$ pts | `WARNING` | 3D Cartesian Axes Turn Red |
| **Gaze Shift** | $\text{Ratio} \notin [0.65, 1.65]$ | $+12$ pts | `WARNING` | Gaze Crosshair Yellow Alert |

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Ankitraj-sharma/exam-proctoring-ai.git
cd exam-proctoring-ai
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Streamlit Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🐳 Docker Deployment

Deploy in containerized production with hardware video forwarding:

```bash
docker-compose up --build
```

---

## 🧪 Running Unit Tests & Benchmarks

```bash
# Execute proctoring unit test suite
python test_proctor.py

# Execute FPS throughput benchmark
python benchmark.py
```

---

## 📜 License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
