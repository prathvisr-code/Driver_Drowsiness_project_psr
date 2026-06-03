import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import csv
import joblib

# =====================================================
# INITIALIZE MEDIAPIPE
# =====================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================================
# LANDMARK INDICES
# =====================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [61, 81, 13, 311, 291, 402, 14, 178]

NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263

# =====================================================
# FUNCTIONS
# =====================================================

# EAR Calculation
def calculate_ear(eye_points):

    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])

    h = np.linalg.norm(eye_points[0] - eye_points[3])

    ear = (v1 + v2) / (2.0 * h)

    return ear


# MAR Calculation
def calculate_mar(mouth_points):

    v1 = np.linalg.norm(mouth_points[1] - mouth_points[7])
    v2 = np.linalg.norm(mouth_points[2] - mouth_points[6])
    v3 = np.linalg.norm(mouth_points[3] - mouth_points[5])

    h = np.linalg.norm(mouth_points[0] - mouth_points[4])

    mar = (v1 + v2 + v3) / (2.0 * h)

    return mar


# Normalization Function
def normalize(value, min_val, max_val):

    value = max(min(value, max_val), min_val)

    return (value - min_val) / (max_val - min_val)


# =====================================================
# THRESHOLDS
# =====================================================

EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.60

FATIGUE_THRESHOLD = 0.45  # Adjusted threshold based on logistic regression model
ALARM_FRAMES = 25
# =====================================================
# COUNTERS
# =====================================================
fatigue_frames = 0
blink_count = 0
yawn_count = 0

start_time = time.time()
last_save_time = time.time()
eye_closed_frames = 0
yawn_frames = 0
fatigue_label = 0
# CREATE CSV FILE
file = open("fatigue_data.csv", "a", newline="")
writer = csv.writer(file)

writer.writerow([
    "EAR",
    "BlinkRate",
    "MAR",
    "YawnRate",
    "VerticalRatio",
    "EyeSlope",
    "HorizontalDiff",
    "Fatigue"
])

# =====================================================
# START WEBCAM
# =====================================================

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # =================================================
            # EYE LANDMARKS
            # =================================================

            left_eye_points = []
            right_eye_points = []

            for idx in LEFT_EYE:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                left_eye_points.append([x, y])

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            for idx in RIGHT_EYE:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                right_eye_points.append([x, y])

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            left_eye_points = np.array(left_eye_points)
            right_eye_points = np.array(right_eye_points)

            # =================================================
            # EAR
            # =================================================

            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)

            ear = (left_ear + right_ear) / 2.0

            # Blink Detection
            if ear < EAR_THRESHOLD:

                eye_closed_frames += 1

            else:

                if eye_closed_frames >= 2:
                    blink_count += 1

                eye_closed_frames = 0

            # =================================================
            # MOUTH LANDMARKS
            # =================================================

            mouth_points = []

            for idx in MOUTH:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                mouth_points.append([x, y])

                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            mouth_points = np.array(mouth_points)

            # =================================================
            # MAR
            # =================================================

            mar = calculate_mar(mouth_points)

            # Yawn Detection
            if mar > MAR_THRESHOLD:

                yawn_frames += 1

                if yawn_frames == 15:
                    yawn_count += 1

            else:
                yawn_frames = 0
            
            # =================================================
            # HEAD POSE FEATURES
            # =================================================

            nose = face_landmarks.landmark[NOSE_TIP]
            chin = face_landmarks.landmark[CHIN]
            forehead = face_landmarks.landmark[FOREHEAD]

            left_eye_corner = face_landmarks.landmark[LEFT_EYE_CORNER]
            right_eye_corner = face_landmarks.landmark[RIGHT_EYE_CORNER]

            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)

            chin_y = int(chin.y * h)
            forehead_y = int(forehead.y * h)

            left_eye_x = int(left_eye_corner.x * w)
            left_eye_y = int(left_eye_corner.y * h)

            right_eye_x = int(right_eye_corner.x * w)
            right_eye_y = int(right_eye_corner.y * h)
        # =================================================
# DRAW HEAD POSE LANDMARKS
# =================================================

# Nose Tip
        cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)

# Forehead
        cv2.circle(frame, (int(forehead.x * w),
                   int(forehead.y * h)),
           5,
           (255, 0, 255),
           -1)

# Chin
        cv2.circle(frame, (int(chin.x * w),
                   int(chin.y * h)),
           5,
           (255, 255, 0),
           -1)

# Left Eye Corner
        cv2.circle(frame, (left_eye_x, left_eye_y),
           5,
           (0, 255, 255),
           -1)

# Right Eye Corner
        cv2.circle(frame, (right_eye_x, right_eye_y),
           5,
           (0, 255, 255),
           -1)

# =================================================
# DRAW HEAD ORIENTATION LINES
# =================================================

#
        eye_center_x = (left_eye_x + right_eye_x) // 2

        horizontal_diff = abs(nose_x - eye_center_x)

            # Vertical Ratio
        vertical_face_length = chin_y - forehead_y

        vertical_diff = nose_y - forehead_y

        vertical_ratio = vertical_diff / vertical_face_length

            # Eye Slope
        eye_slope = abs(right_eye_y - left_eye_y)
        # =================================================
# TIME ELAPSED
# =================================================

        elapsed_time = time.time() - start_time

# Avoid division by zero
        elapsed_time_minutes = max(elapsed_time / 60, 0.01)

# =================================================
# BLINK RATE & YAWN RATE
# =================================================

        blink_rate = blink_count / elapsed_time_minutes
        yawn_rate = yawn_count / elapsed_time_minutes
        # =================================================
       

        
            # =================================================
            # NORMALIZATION
            # =================================================

            # EAR -> lower EAR means higher fatigue
        norm_ear = normalize(ear, 0.15, 0.35)
        eye_closure_score = 1 - norm_ear

        #     # Blink normalization
        norm_blink = normalize(blink_rate, 0, 40)

        #     # MAR normalization
        norm_mar = normalize(mar, 0.2, 1.0)

        #     # Yawn count normalization
        norm_yawn = normalize(yawn_rate, 0, 10)

        #     # Vertical ratio normalization
        norm_vertical = normalize(vertical_ratio, 0.3, 0.8)

        #     # Eye slope normalization
        norm_slope = normalize(eye_slope, 0, 40)

             # Horizontal distraction normalization
        norm_horizontal = normalize(horizontal_diff, 0, 80)

        #     # =================================================
             # FATIGUE SCORE
             
        #     # =================================================

        fatigue_score = (
                 0.4471 * eye_closure_score +
                 0.0078 * norm_blink +          # WEIGHTS FROM LOGISTIC REGRESSION MODEL
                 0.0685 * norm_mar +
                 0.2469 * norm_yawn +
                 0.2111 * norm_vertical +
                 0.0153 * norm_slope +
                 0.0033 * norm_horizontal
             )

            # =================================================
            # DISPLAY VALUES
            # =================================================

        cv2.putText(frame,
                        f'EAR: {ear:.2f}',
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2)

        cv2.putText(frame,
                        f'MAR: {mar:.2f}',
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2)

        cv2.putText(frame,
                        f'Blink Rate: {blink_rate:.1f}/min',
                        (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2)

        cv2.putText(frame,
                        f'Yawn Rate: {yawn_rate:.1f}/min',
                        (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2)

        cv2.putText(frame,
                        f'Fatigue Score: {fatigue_score:.2f}',
                        (30, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 0),
                        3)
        cv2.putText(frame,
            f'Head Tilt: {eye_slope:.2f}',
            (30, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2)
            # =================================================
            # ALERT LOGIC
            # =================================================

            # =================================================
        # FATIGUE ALERT LOGIC
        if fatigue_score > FATIGUE_THRESHOLD:

            fatigue_frames += 1

            cv2.putText(frame,
                        'DRIVER FATIGUE DETECTED!',
                         (30, 320),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         1.0,
                         (0, 0, 255),
                         4)

            # Alarm after sustained fatigue
            if fatigue_frames >= ALARM_FRAMES:

                 cv2.putText(frame,
                             'WAKE UP ALARM!',
                             (30, 380),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             1.0,
                             (0, 0, 255),
                             4)

                 # Beep Sound
                 winsound.Beep(1000, 500)

        else:

             fatigue_frames = 0

             cv2.putText(frame,
                         'Driver Attentive',
                         (50, 320),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         1,
                         (0, 255, 0),
                         3)
        current_time = time.time()
        if current_time - last_save_time >= 1:  # Save every 1 seconds
            writer.writerow([
                ear,
                blink_rate,
                mar,
                yawn_rate,
                vertical_ratio,
                eye_slope,
                horizontal_diff,
                fatigue_label
            ])
            last_save_time = current_time

        # =====================================================
        # SHOW OUTPUT
        # =====================================================

        cv2.imshow("Driver Fatigue Monitoring System", frame)

# Read keyboard input
        key = cv2.waitKey(1) & 0xFF

# Label data manually
        if key == ord('a'):

          fatigue_label = 0      # Alert

        elif key == ord('d'):
          fatigue_label = 1      # Drowsy

# Quit application
        elif key == ord('q'):
          break

# =====================================================
# RELEASE RESOURCES
# =====================================================

cap.release()
cv2.destroyAllWindows()
file.close()