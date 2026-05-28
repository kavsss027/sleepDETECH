# 👁️ SleepDETECH — Real-Time Driver Drowsiness Detection

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)](https://opencv.org/)
[![dlib](https://img.shields.io/badge/dlib-19.24-orange.svg)](http://dlib.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

A real-time computer vision system that monitors a person's eyes through the laptop camera, calculates the Eye Aspect Ratio (EAR) on every frame, and determines whether the person is **AWAKE** or **SLEEPING** — displaying a fully annotated live feed with a traffic-light color system, landmark overlays, and a frame-accurate drowsiness counter.

---

## ⚡ What It Does

SleepDETECH solves a genuine safety problem: detecting human drowsiness in real time using only a standard laptop camera and a mathematically transparent, explainable algorithm. No cloud, no GPU, no black boxes.

### 1. Real-Time Eye Aspect Ratio (EAR) Measurement
- **Per-frame EAR calculation** using the Soukupova & Cech (2016) formula — the academic standard for eye-state detection
- **68-point facial landmark detection** via a pre-trained dlib regression forest model
- **Bilateral averaging** — both eyes measured independently and averaged for robustness against partial occlusion
- **Scale-invariant formula** — produces consistent EAR values regardless of the person's distance from the camera
- **Personally calibrated threshold** — adjusted from the academic default of 0.25 to 0.30 based on observed eye geometry, demonstrating real-world calibration methodology

### 2. Traffic-Light Annotation System
The entire visual output is color-coded as a coherent system — not individual elements changing independently:

| State | Trigger | Color | Label |
|:---|:---|:---|:---|
| **AWAKE** | Counter below warning zone | 🟢 Green | `AWAKE` (bottom-left) |
| **WARNING** | Counter > 50% of frame limit | 🟠 Orange | `AWAKE` (color changes, label holds) |
| **SLEEPING** | Counter ≥ frame limit | 🔴 Red | `SLEEPING` (centered, large) |
| **NO FACE** | No face detected | 🟡 Yellow | `NO FACE DETECTED` (centered) |

### 3. Live Annotation Layers
Every frame renders six annotation layers in a strict drawing order:
- **Face bounding rectangle** — confirms face lock, color-coded to current state
- **Eye landmark dots** — 12 filled circles (6 per eye) tracing the eyelid contours in real time
- **EAR value display** — live numerical EAR reading to 2 decimal places
- **Counter progress** — `Closed: X / 20` showing exactly how close the system is to triggering
- **Main state label** — large, prominently positioned AWAKE or SLEEPING determination
- **Semi-transparent text backgrounds** — `cv2.addWeighted()` overlay technique for readability against any background

### 4. Robust Edge Case Handling
Ten edge cases are explicitly specified and handled — including counter-hold on face disappearance (so the alert cannot be defeated by briefly looking away), multiple-face disambiguation, division-by-zero EAR guard, missing model file detection, and invalid camera index detection with descriptive error messages.

---

## 🛠️ Technology Stack

| Layer | Library | Version | Role |
| :--- | :--- | :--- | :--- |
| **Camera & Vision** | [OpenCV](https://opencv.org/) (`opencv-python`) | `4.9.0.80` | Camera capture, frame processing, all annotation drawing |
| **Face Detection** | [dlib](http://dlib.net/) | `19.24.2` | HOG-based face detection + 68-point facial landmark prediction |
| **Convenience** | [imutils](https://github.com/jrosebr1/imutils) | `0.5.4` | Frame resizing, dlib-to-NumPy conversion |
| **Mathematics** | [SciPy](https://scipy.org/) | `1.13.0` | Euclidean distance calculation for EAR formula |
| **Arrays** | [NumPy](https://numpy.org/) | `1.26.4` | Underlying array structure for all frame and coordinate data |
| **Language** | [Python](https://www.python.org/) | `3.11+` | Runtime environment |

### Pre-Trained Model

| File | Size | Purpose |
|:---|:---|:---|
| `shape_predictor_68_face_landmarks.dat` | ~100 MB | Trained regression forest — predicts 68 facial landmark coordinates from a detected face bounding box |

---

## 🧠 How the Detection Works

### The Eye Aspect Ratio (EAR) Formula

The EAR is computed using 6 landmark points per eye, arranged clockwise around the eye socket:

```
        P2 ---- P3
       /            \
     P1              P4
       \            /
        P6 ---- P5

EAR = ( ||P2-P6|| + ||P3-P5|| ) / ( 2.0 × ||P1-P4|| )
```

The numerator averages two vertical distances (eyelid height). The denominator is twice the horizontal eye width. Dividing by width makes the ratio **scale-invariant** — the same person produces the same EAR value regardless of camera distance.

### Observed EAR Calibration Data

| Eye State | Observed EAR Range |
|:---|:---|
| Fully open (normal) | `0.34 – 0.46` |
| Slow deliberate blink (minimum) | `~0.196` |
| Sustained closure (average) | `~0.165` |
| Sustained closure (minimum) | `~0.110` |

### The State Machine

One integer counter drives the entire system. Two rules. No timers, no threads, no external state:

```
if avg_ear < EAR_THRESHOLD (0.30):
    counter += 1                      # eye closing — count up
else:
    counter = 0                       # eye open — reset immediately

if counter >= FRAME_LIMIT (20):
    state = SLEEPING                  # ~0.67s sustained closure
else:
    state = AWAKE
```

A normal blink (~100–400ms) never accumulates enough frames to reach the limit. Only sustained closure triggers the state change.

---

## 📂 Project Structure

```
sleepDETECH/
├── drowsiness_detector.py                   ← Single application file (entire system)
├── shape_predictor_68_face_landmarks.dat    ← Pre-trained dlib model (downloaded separately)
├── requirements.txt                         ← Pinned library versions
└── README.md                                ← You are here
```

---

## 🚀 Quick Start

### Prerequisites

- Python `3.11` or higher
- Git
- **Windows only:** [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the **Desktop development with C++** workload (required to build dlib)
- **macOS only:** `brew install cmake`
- **Linux only:** `sudo apt-get install cmake build-essential`

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/kavsss027/sleepDETECH.git
cd sleepDETECH
```

#### 2. Download the Pre-Trained Model
Download the dlib landmark model file and place it in the project root:
```
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```
Extract the `.bz2` archive. The resulting `.dat` file (~100MB) must sit alongside `drowsiness_detector.py`.

> **Windows tip:** If dlib fails to compile from source, download a precompiled wheel from
> https://github.com/z-mahmud22/Dlib_Windows_Python3.x/releases
> and install it directly: `pip install dlib-19.24.2-cp311-cp311-win_amd64.whl`

#### 3. Create a Virtual Environment
```bash
# Create
python -m venv venv

# Activate — Windows PowerShell
.\venv\Scripts\Activate.ps1

# Activate — macOS / Linux
source venv/bin/activate
```

#### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 5. Run the Application
```bash
python drowsiness_detector.py
```

The system prints a startup summary to the terminal, then opens the annotated camera feed window. Press **Q** to quit cleanly at any time.

### Expected Startup Output
```
Drowsiness Detection System starting...
EAR Threshold : 0.30
Frame Limit   : 20 frames (~0.67s at 30fps)
Camera Index  : 0
Press Q to quit.
```

### Port / Window Reference

```
Camera Feed Window:   Opens automatically on launch
Terminal Output:      Startup config + any error messages
Exit:                 Press Q or q inside the camera window
```

---

## ⚙️ Configuration

All tunable parameters are defined as named constants at the top of `drowsiness_detector.py`. No magic numbers exist anywhere else in the code.

| Constant | Default | Effect | Formula |
|:---|:---|:---|:---|
| `EAR_THRESHOLD` | `0.30` | EAR value below which eye is considered closed | Calibrate by observing your personal open-eye EAR range |
| `FRAME_LIMIT` | `20` | Consecutive closed frames before SLEEPING triggers | `desired_seconds × camera_fps` |
| `WARNING_ZONE_RATIO` | `0.5` | Point at which annotations turn orange | `0.0` = always orange, `1.0` = no warning zone |
| `CAMERA_INDEX` | `0` | Which camera to use | Try `1` or `2` for external USB cameras |
| `FRAME_RESIZE_WIDTH` | `450` | Processing resolution in pixels | Lower = faster, higher = more accurate |

**Example: changing alert trigger to 5 seconds**
```python
FRAME_LIMIT = 150   # 5 seconds × 30 fps = 150 frames
```

---

## 🧪 Testing

The project was developed across 5 phases with a structured manual verification test suite — 33 tests total across all phases, 33 passing.

| Phase | Tests Run | Tests Passed | Key Verification |
|:---|:---|:---|:---|
| 1 — Environment Setup | 5 | 5 ✅ | All libraries import, camera opens, model file found |
| 2 — Face Detection | 5 | 5 ✅ | Bounding box tracks face, no-face banner appears |
| 3 — Landmark & EAR | 7 | 7 ✅ | Eye dots trace correctly, EAR drops on blink |
| 4 — State Detection | 8 | 8 ✅ | Traffic-light transitions, two full AWAKE→SLEEPING→AWAKE cycles |
| 5 — Edge Cases & Polish | 8 | 8 ✅ | All 10 edge cases verified, 60-second end-to-end demo |

**Total: 33 / 33 PASS ✅**

To verify the system manually after setup:
1. Run `python drowsiness_detector.py`
2. Confirm green annotations and `AWAKE` label with eyes open
3. Close eyes slowly — watch counter climb and color shift to orange then red
4. Open eyes — confirm immediate reset to green `AWAKE`
5. Press Q — confirm clean exit

---

## 🔬 Key Design Decisions

**EAR threshold calibration** — The academic default of 0.25 was adjusted to 0.30 after measuring a personal open-eye EAR range of 0.34–0.46. This demonstrates that EAR thresholds must be calibrated per individual, not taken from literature defaults.

**Counter-hold on face disappearance** — When no face is detected, the counter holds its current value rather than resetting. This prevents the alert from being defeated by briefly looking away from the camera — a non-obvious correctness requirement.

**Frame counter vs. time-based timer** — A simple integer counter is used rather than `time.time()` subtraction. This is simpler, naturally synchronized with the processing loop, and sufficient for this use case. The relationship `seconds = counter / fps` makes the timing explicit and adjustable via one parameter.

**Single-file architecture** — The entire system lives in one Python file with four clearly separated sections (imports → configuration → helper functions → main loop). This makes every decision visible and traceable, which is the right architecture for a focused demonstration project.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

*Disclaimer: SleepDETECH is a computer vision demonstration project. It is not a certified safety device and should not be relied upon as the sole drowsiness mitigation system in any safety-critical application.*
