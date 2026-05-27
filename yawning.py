

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
# Mouth Landmark Indices
# -----------------------------

# Mouth landmarks
MOUTH = [61, 81, 13, 311, 291, 402, 14, 178]

# -----------------------------
# MAR Function
# -----------------------------
def calculate_mar(mouth_points):

    # Vertical distances
    v1 = np.linalg.norm(mouth_points[1] - mouth_points[7])
    v2 = np.linalg.norm(mouth_points[2] - mouth_points[6])
    v3 = np.linalg.norm(mouth_points[3] - mouth_points[5])

    # Horizontal distance
    h = np.linalg.norm(mouth_points[0] - mouth_points[4])

    # Mouth Aspect Ratio
    mar = (v1 + v2 + v3) / (2.0 * h)

    return mar

# -----------------------------
# Thresholds
# -----------------------------
MAR_THRESHOLD = 0.60
YAWN_FRAMES = 15

frame_counter = 0
yawn_count = 0

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

            mouth_points = []

            # -----------------------------
            # Extract Mouth Landmarks
            # -----------------------------
            for idx in MOUTH:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                mouth_points.append([x, y])

                # Draw mouth points
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            mouth_points = np.array(mouth_points)

            # -----------------------------
            # Calculate MAR
            # -----------------------------
            mar = calculate_mar(mouth_points)

            # Display MAR
            cv2.putText(frame,
                        f'MAR: {mar:.2f}',
                        (400, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2)

            # -----------------------------
            # Yawning Detection
            # -----------------------------
            if mar > MAR_THRESHOLD:

                frame_counter += 1

                cv2.putText(frame,
                            'Yawning...',
                            (400, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3)

                # Count yawn
                if frame_counter == YAWN_FRAMES:
                    yawn_count += 1

            else:
                frame_counter = 0

            # Display Yawn Count
            cv2.putText(frame,
                        f'Yawns: {yawn_count}',
                        (400, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2)

    # Show output
    cv2.imshow("Yawning Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()