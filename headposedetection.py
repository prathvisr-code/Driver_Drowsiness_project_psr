import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Important Face Landmarks
# -----------------------------
# Nose tip
NOSE_TIP = 1

# Chin
CHIN = 152

# Left eye corner
LEFT_EYE = 33

# Right eye corner
RIGHT_EYE = 263

# Forehead
FOREHEAD = 10

# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # -----------------------------
            # Get Landmark Coordinates
            # -----------------------------
            landmarks = {}

            important_points = {
                "nose": NOSE_TIP,
                "chin": CHIN,
                "left_eye": LEFT_EYE,
                "right_eye": RIGHT_EYE,
                "forehead": FOREHEAD
            }

            for name, idx in important_points.items():

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                landmarks[name] = (x, y)

                # Draw landmark
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # -----------------------------
            # Head Pose Calculations
            # -----------------------------

            nose_x, nose_y = landmarks["nose"]

            left_eye_x, left_eye_y = landmarks["left_eye"]
            right_eye_x, right_eye_y = landmarks["right_eye"]

            chin_x, chin_y = landmarks["chin"]

            forehead_x, forehead_y = landmarks["forehead"]

            # --------------------------------
            # LEFT / RIGHT Detection
            # --------------------------------
            eye_center_x = (left_eye_x + right_eye_x) // 2

            horizontal_diff = nose_x - eye_center_x

            # --------------------------------
            # UP / DOWN Detection
            # --------------------------------
            vertical_face_length = chin_y - forehead_y

            vertical_diff = nose_y - forehead_y

            # Normalize
            vertical_ratio = vertical_diff / vertical_face_length

            # --------------------------------
            # Head Tilt Detection
            # --------------------------------
            eye_slope = (right_eye_y - left_eye_y)

            # -----------------------------
            # Attention Status
            # -----------------------------
            status = "ATTENTIVE"

            # Looking left/right
            if horizontal_diff > 20:
                status = "LOOKING RIGHT"

            elif horizontal_diff < -20:
                status = "LOOKING LEFT"

            # Looking down
            if vertical_ratio > 0.65:
                status = "LOOKING DOWN"

            # Sleeping sideways
            if abs(eye_slope) > 15:
                status = "HEAD TILTED"

            # -----------------------------
            # Display Information
            # -----------------------------
            cv2.putText(frame,
                        f'Status: {status}',
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)

            cv2.putText(frame,
                        f'Horizontal Diff: {horizontal_diff}',
                        (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2)

            cv2.putText(frame,
                        f'Vertical Ratio: {vertical_ratio:.2f}',
                        (50, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2)

            cv2.putText(frame,
                        f'Eye Slope: {eye_slope}',
                        (50, 220),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2)

    # Show frame
    cv2.imshow("Head Pose Detection", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()