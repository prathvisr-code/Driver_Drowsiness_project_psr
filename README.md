## Driver Fatigue Monitoring System using OpenCV and MediaPipe

A real-time AI-based Driver Fatigue Monitoring System developed using Python, OpenCV, and MediaPipe.  
The system detects driver drowsiness and inattentiveness using multiple facial behavior indicators such as eye closure, blinking rate, yawning rate, and head pose estimation.


Features

- Real-time facial landmark detection
- Eye Aspect Ratio (EAR) based eye closure detection
- Blink rate monitoring
- Mouth Aspect Ratio (MAR) based yawn detection
- Yawn rate monitoring
- Head pose estimation
- Detection of:
  - looking down
  - head tilt
  - driver distraction
- Fatigue score calculation using weighted normalized features
- Real-time fatigue alert system
- Alarm sound for prolonged fatigue detection
- Live webcam visualization with landmarks and monitoring parameters


Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Winsound


 System Workflow

Webcam Feed  
→ Face Mesh Detection  
→ Facial Landmark Extraction  
→ Feature Engineering  
→ Feature Normalization  
→ Weighted Fatigue Score Calculation  
→ Alert Generation


 Facial Features Used

 Eye Features
- Eye Aspect Ratio (EAR)
- Blink Rate

 Mouth Features
- Mouth Aspect Ratio (MAR)
- Yawn Rate

 Head Pose Features
- Horizontal Head Deviation
- Vertical Face Ratio
- Eye Slope / Head Tilt


 Fatigue Score Formula

The fatigue score is computed using weighted normalized features:

FatigueScore =
0.39 × EyeClosureScore +
0.10 × BlinkRate +
0.25 × MAR +
0.10 × YawnRate +
0.03 × VerticalRatio +
0.10 × EyeSlope +
0.03 × HorizontalDifference


 Fatigue Detection Logic

The system:
- normalizes all extracted features
- computes a weighted fatigue score
- triggers an alert if fatigue persists for multiple frames

Alarm activates when:
- fatigue score exceeds threshold
- fatigue condition remains sustained


Installation

Clone Repository

```bash
git clone https://github.com/your-username/driver-fatigue-monitoring-system.git
cd driver-fatigue-monitoring-system
