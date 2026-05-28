import os
import sys
import time
import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np

# =============================================================================
# SECTION 2 — CONFIGURATION CONSTANTS
# =============================================================================

# --- Detection Parameters ---
EAR_THRESHOLD = 0.30
# The Eye Aspect Ratio value below which an eye is considered CLOSED.
# Range: 0.18 (less sensitive, only catches very closed eyes) to
#        0.30 (more sensitive, triggers on slightly droopy eyes).
# Default 0.25 sits between normal open-eye EAR (0.25–0.32) and
# drowsy/closed-eye EAR (0.00–0.22). Do not adjust without testing.

FRAME_LIMIT = 20
# Number of consecutive frames the EAR must stay below EAR_THRESHOLD
# before the state transitions to SLEEPING.
# Formula: FRAME_LIMIT = desired_seconds × camera_fps
# Examples: 0.67s = 20 frames | 1.0s = 30 frames | 3.0s = 90 frames
#           5.0s = 150 frames
# Default 20 frames ≈ 0.67 seconds at 30fps.

WARNING_ZONE_RATIO = 0.5
# When counter exceeds (FRAME_LIMIT × WARNING_ZONE_RATIO), annotations
# transition from GREEN to ORANGE as an early warning signal.
# Default 0.5 means the warning zone starts at 50% of the frame limit.
# Range: 0.3 (earlier warning) to 0.8 (later warning).

DETECTOR_UPSCALE_TIMES = 0
# Number of times to upscale the frame when running dlib HOG detector.
# Range: 0 (default, fastest) to 1 (more accurate for small faces, slower).

# --- Camera Parameters ---
CAMERA_INDEX = 0
# Index of the camera to use. 0 = built-in webcam (default).
# Use 1 or 2 for external USB cameras if 0 does not work.

FRAME_RESIZE_WIDTH = 450
# Width in pixels to resize each frame before processing.
# Smaller = faster processing, less accurate landmark detection.
# Larger = slower processing, more accurate landmark detection.
# Range: 320 (minimum for reliable detection) to 640 (maximum useful).
# Default 450 is the community-standard balance point.

# --- Model Path ---
LANDMARK_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
# Path to the dlib pre-trained landmark model file.
# Must be in the same directory as drowsiness_detector.py.
# If placed elsewhere, update this path accordingly.

# --- Annotation Visual Parameters ---
# Colors are in BGR format (Blue, Green, Red) — NOT RGB.
COLOR_AWAKE   = (0, 255, 0)      # Green  — normal state
COLOR_WARNING = (0, 165, 255)    # Orange — counter in warning zone
COLOR_SLEEPING = (0, 0, 255)     # Red    — SLEEPING state active
COLOR_NO_FACE  = (0, 255, 255)   # Yellow — no face detected in frame
COLOR_WHITE    = (255, 255, 255) # White  — secondary text
COLOR_BLACK    = (0, 0, 0)       # Black  — text background rectangles

FONT = cv2.FONT_HERSHEY_SIMPLEX
# OpenCV font used for all text annotations.
# FONT_HERSHEY_SIMPLEX is the clearest for small sizes on video frames.

FONT_SCALE_LARGE  = 1.2   # Used for the main state label (AWAKE / SLEEPING)
FONT_SCALE_MEDIUM = 0.7   # Used for EAR value and counter progress text
FONT_SCALE_SMALL  = 0.55  # Used for secondary information text
FONT_THICKNESS    = 2     # Line thickness for all text rendering

LANDMARK_DOT_RADIUS = 2   # Pixel radius of each eye landmark dot
LANDMARK_DOT_THICKNESS = -1  # -1 means filled circle (solid dot)

FACE_BOX_THICKNESS = 2    # Line thickness for the face bounding rectangle

# --- Window Parameters ---
WINDOW_TITLE = "Drowsiness Detection — Press Q to Quit"
# Title displayed in the OpenCV window title bar.

# --- Execution and Flow Constants ---
WAIT_TIME_MS = 1
# Time to wait for keypress in waitKey, in milliseconds.
# Default 1ms is required for smooth real-time video display.

KEY_MASK = 0xFF
# Bitmask for capturing key presses across different operating systems.

MAX_CONSECUTIVE_FAILURES = 30
# Maximum consecutive frame read failures allowed before exiting the application.

INCREMENT_STEP = 1
# Step value to increment consecutive frame counters.


# --- Layout and Position Offsets ---
# X coordinate offset from the left edge for all top-left text overlays.
TEXT_START_X = 10

# Y coordinate for the first line of text (EAR / detection label).
TEXT_LINE_1_Y = 30

# Y coordinate for the second line of text (Counter / status).
TEXT_LINE_2_Y = 60

# Y-offset from the bottom edge of the frame for the bottom-left state label.
STATE_LABEL_BOTTOM_OFFSET = 20

# Horizontal and vertical padding around text for readability background boxes.
BOX_PADDING = 10

# Blend weights for background opacity overlay
BG_OVERLAY_WEIGHT = 0.5
BG_ORIGINAL_WEIGHT = 0.5
ADDITIVE_SCALAR = 0.0

# Division factor used to calculate horizontal/vertical centers of the frame.
CENTER_DIVISOR = 2

# --- Landmark Index Constants ---
LEFT_EYE_START = 36
LEFT_EYE_END = 42
RIGHT_EYE_START = 42
RIGHT_EYE_END = 48

# --- Eye Landmark Local Indices ---
# Each eye has 6 points (0 to 5) arranged clockwise starting from the leftmost corner.
EYE_P1_INDEX = 0  # Leftmost corner
EYE_P2_INDEX = 1  # Upper-left eyelid point
EYE_P3_INDEX = 2  # Upper-right eyelid point
EYE_P4_INDEX = 3  # Rightmost corner
EYE_P5_INDEX = 4  # Lower-right eyelid point
EYE_P6_INDEX = 5  # Lower-left eyelid point

# --- Math and Zero-division Guard Constants ---
EAR_NORMALIZATION_FACTOR = 2.0
MIN_EYE_WIDTH = 0.001
MAX_CONSECUTIVE_ZERO_EAR_FAILURES = 10

# --- System Exit Code Constants ---
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

# =============================================================================
# SECTION 3 — HELPER FUNCTIONS
# =============================================================================

def calculate_ear(eye):
    """
    Computes the Eye Aspect Ratio (EAR) for a single eye's landmark coordinates.
    Formula: EAR = (||P2 - P6|| + ||P3 - P5||) / (2.0 * ||P1 - P4||)
    """
    # Vertical distances between upper and lower eyelids
    A = dist.euclidean(eye[EYE_P2_INDEX], eye[EYE_P6_INDEX]) # P2 to P6
    B = dist.euclidean(eye[EYE_P3_INDEX], eye[EYE_P5_INDEX]) # P3 to P5

    # Horizontal distance between eye corners
    C = dist.euclidean(eye[EYE_P1_INDEX], eye[EYE_P4_INDEX]) # P1 to P4

    # Division by zero guard (Edge Case 10)
    if C < MIN_EYE_WIDTH:
        return 0.0

    return (A + B) / (EAR_NORMALIZATION_FACTOR * C)


def get_eye_landmarks(shape):
    """
    Extracts the left and right eye coordinate sets from the full 68-point array.
    """
    left_eye = shape[LEFT_EYE_START:LEFT_EYE_END]
    right_eye = shape[RIGHT_EYE_START:RIGHT_EYE_END]
    return left_eye, right_eye

# =============================================================================
# SECTION 4 — MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    # Task 5.8: Print the active configuration summary on startup
    print("Drowsiness Detection System starting...")
    print(f"EAR Threshold : {EAR_THRESHOLD:.2f}")
    print(f"Frame Limit   : {FRAME_LIMIT} frames (~0.67s at 30fps)")
    print(f"Camera Index  : {CAMERA_INDEX}")
    print("Press Q to quit.")

    # Edge Case 8: Validate landmark model file existence
    if not os.path.exists(LANDMARK_MODEL_PATH):
        print("ERROR: Landmark model file not found.")
        print(f"Expected location: {LANDMARK_MODEL_PATH}")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract the .bz2 file and place the .dat file in the same directory as this script.")
        sys.exit(EXIT_FAILURE)

    # Initialize dlib's HOG-based face detector and shape predictor
    print("Initializing face detector...")
    detector = dlib.get_frontal_face_detector()
    
    print(f"Loading landmark predictor from {LANDMARK_MODEL_PATH}...")
    predictor = dlib.shape_predictor(LANDMARK_MODEL_PATH)

    # Initialize video capture stream
    print(f"Opening camera index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    # Edge Case 9: Validate camera availability
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {CAMERA_INDEX}.")
        print("If you have an external camera, try setting CAMERA_INDEX = 1 or CAMERA_INDEX = 2")
        print("in the configuration section at the top of this file.")
        sys.exit(EXIT_FAILURE)

    # State machine and failure counter initializations
    counter = 0 # Persists across frames
    consecutive_failures = 0
    consecutive_zero_ear_failures = 0

    print("Camera stream started. Press 'Q' to quit.")

    while True:
        # 1. Frame Capture and Edge Case 5 Check
        ret, frame = cap.read()
        
        # Edge Case 5: Handle single frame failure and count consecutive failures
        if not ret or frame is None:
            consecutive_failures += INCREMENT_STEP
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"ERROR: Camera feed lost. {MAX_CONSECUTIVE_FAILURES} consecutive frame read failures. Exiting.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(EXIT_FAILURE)
            continue
            
        # Reset consecutive failures count on a successful frame read
        consecutive_failures = 0

        # 2. Frame Resizing and Grayscale Conversion
        frame = imutils.resize(frame, width=FRAME_RESIZE_WIDTH)
        frame_height, frame_width, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. Face Detection and Edge Case 1 Check
        faces = detector(gray, DETECTOR_UPSCALE_TIMES)

        if len(faces) == 0:
            # Edge Case 1: No Face Detected branch (Counter holds its current value)
            text = "NO FACE DETECTED"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, FONT, FONT_SCALE_MEDIUM, FONT_THICKNESS
            )
            
            # Center coordinates
            x = (frame_width - text_width) // CENTER_DIVISOR
            y = (frame_height + text_height) // CENTER_DIVISOR

            # Draw background rectangle overlay for readability (Layer 6)
            overlay = frame.copy()
            cv2.rectangle(
                overlay, 
                (x - BOX_PADDING, y - text_height - BOX_PADDING), 
                (x + text_width + BOX_PADDING, y + baseline + BOX_PADDING), 
                COLOR_BLACK, 
                cv2.FILLED
            )
            cv2.addWeighted(overlay, BG_OVERLAY_WEIGHT, frame, BG_ORIGINAL_WEIGHT, ADDITIVE_SCALAR, frame)

            # Draw "NO FACE DETECTED" text in yellow (COLOR_NO_FACE) (Layer 5)
            cv2.putText(
                frame, 
                text, 
                (x, y), 
                FONT, 
                FONT_SCALE_MEDIUM, 
                COLOR_NO_FACE, 
                FONT_THICKNESS
            )
        else:
            # Face(s) detected - process only faces[0] (Edge Case 2)
            face = faces[0]
            
            # 4. Facial Landmark Prediction and Extraction
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            left_eye, right_eye = get_eye_landmarks(shape)

            # 5. EAR Calculation and Edge Case 10 Check
            left_width = dist.euclidean(left_eye[EYE_P1_INDEX], left_eye[EYE_P4_INDEX])
            right_width = dist.euclidean(right_eye[EYE_P1_INDEX], right_eye[EYE_P4_INDEX])
            
            if left_width < MIN_EYE_WIDTH or right_width < MIN_EYE_WIDTH:
                consecutive_zero_ear_failures += INCREMENT_STEP
                if consecutive_zero_ear_failures > MAX_CONSECUTIVE_ZERO_EAR_FAILURES:
                    print(f"WARNING: More than {MAX_CONSECUTIVE_ZERO_EAR_FAILURES} consecutive frames with zero horizontal eye distance.")
                # Skip the EAR calculation, hold the counter, and continue to next frame
                continue
                
            consecutive_zero_ear_failures = 0

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / CENTER_DIVISOR

            # 6. State Machine and Alert Color Logic
            if avg_ear < EAR_THRESHOLD:
                counter += INCREMENT_STEP
            else:
                counter = 0

            state = "SLEEPING" if counter >= FRAME_LIMIT else "AWAKE"

            if counter >= FRAME_LIMIT:
                active_color = COLOR_SLEEPING
            elif counter > FRAME_LIMIT * WARNING_ZONE_RATIO:
                active_color = COLOR_WARNING
            else:
                active_color = COLOR_AWAKE

            # 7. Overlay Annotations and Drawing Order
            ear_text = f"EAR: {avg_ear:.2f}"
            (ear_w, ear_h), ear_baseline = cv2.getTextSize(
                ear_text, FONT, FONT_SCALE_MEDIUM, FONT_THICKNESS
            )
            
            counter_text = f"Closed: {counter} / {FRAME_LIMIT}"
            (counter_w, counter_h), counter_baseline = cv2.getTextSize(
                counter_text, FONT, FONT_SCALE_MEDIUM, FONT_THICKNESS
            )

            if state == "SLEEPING":
                state_label = "SLEEPING"
                (state_w, state_h), state_baseline = cv2.getTextSize(
                    state_label, FONT, FONT_SCALE_LARGE, FONT_THICKNESS
                )
                x_state = (frame_width - state_w) // CENTER_DIVISOR
                y_state = (frame_height + state_h) // CENTER_DIVISOR

            # 7.1. Draw Background Rectangles (Layer 6)
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (TEXT_START_X - BOX_PADDING, TEXT_LINE_1_Y - ear_h - BOX_PADDING),
                (TEXT_START_X + ear_w + BOX_PADDING, TEXT_LINE_1_Y + ear_baseline + BOX_PADDING),
                COLOR_BLACK,
                cv2.FILLED
            )
            cv2.rectangle(
                overlay,
                (TEXT_START_X - BOX_PADDING, TEXT_LINE_2_Y - counter_h - BOX_PADDING),
                (TEXT_START_X + counter_w + BOX_PADDING, TEXT_LINE_2_Y + counter_baseline + BOX_PADDING),
                COLOR_BLACK,
                cv2.FILLED
            )
            if state == "SLEEPING":
                cv2.rectangle(
                    overlay,
                    (x_state - BOX_PADDING, y_state - state_h - BOX_PADDING),
                    (x_state + state_w + BOX_PADDING, y_state + state_baseline + BOX_PADDING),
                    COLOR_BLACK,
                    cv2.FILLED
                )
            cv2.addWeighted(overlay, BG_OVERLAY_WEIGHT, frame, BG_ORIGINAL_WEIGHT, ADDITIVE_SCALAR, frame)

            # 7.2. Draw face bounding box (Layer 1) using active_color
            left_coord = face.left()
            top_coord = face.top()
            right_coord = face.right()
            bottom_coord = face.bottom()
            
            cv2.rectangle(
                frame, 
                (left_coord, top_coord), 
                (right_coord, bottom_coord), 
                active_color, 
                FACE_BOX_THICKNESS
            )

            # 7.3. Draw eye landmark dots (Layer 2) using active_color
            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), LANDMARK_DOT_RADIUS, active_color, LANDMARK_DOT_THICKNESS)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), LANDMARK_DOT_RADIUS, active_color, LANDMARK_DOT_THICKNESS)

            # 7.4. Display EAR value text (Layer 3) in white
            cv2.putText(
                frame,
                ear_text,
                (TEXT_START_X, TEXT_LINE_1_Y),
                FONT,
                FONT_SCALE_MEDIUM,
                COLOR_WHITE,
                FONT_THICKNESS
            )

            # 7.5. Display Counter progress text (Layer 4) using active_color
            cv2.putText(
                frame,
                counter_text,
                (TEXT_START_X, TEXT_LINE_2_Y),
                FONT,
                FONT_SCALE_MEDIUM,
                active_color,
                FONT_THICKNESS
            )

            # 7.6. Display main state label (Layer 5) last on top
            if state == "SLEEPING":
                cv2.putText(
                    frame, 
                    state_label, 
                    (x_state, y_state), 
                    FONT, 
                    FONT_SCALE_LARGE, 
                    active_color, 
                    FONT_THICKNESS
                )
            else:
                # AWAKE state (drawn bottom-left, using active_color)
                cv2.putText(
                    frame, 
                    "AWAKE", 
                    (TEXT_START_X, frame_height - STATE_LABEL_BOTTOM_OFFSET), 
                    FONT, 
                    FONT_SCALE_LARGE, 
                    active_color, 
                    FONT_THICKNESS
                )

        # 8. Render Frame and Exit Keyboard Check
        cv2.imshow(WINDOW_TITLE, frame)

        # Check for program exit ('q' or 'Q' key press)
        key = cv2.waitKey(WAIT_TIME_MS) & KEY_MASK
        if key == ord('q') or key == ord('Q'):
            break

    # Cleanup and release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Program exited cleanly.")
