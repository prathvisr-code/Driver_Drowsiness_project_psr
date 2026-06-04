# Real-Time Driver Fatigue and Attention Monitoring System Using openCV , Mediapipe and Machine Learning

## Overview

This project is a real-time Driver Fatigue and Attention Monitoring System developed using Python, OpenCV, MediaPipe, and Machine Learning.

The system monitors a driver's facial behavior through a webcam and detects signs of fatigue and inattention using multiple physiological indicators including eye closure, blinking behavior, yawning behavior, and head pose changes.

A custom dataset was collected from webcam recordings and used to train a machine learning model for fatigue classification.

---

## Key Features

- Real-time facial landmark detection using MediaPipe Face Mesh
- Eye Aspect Ratio (EAR) based eye closure monitoring
- Blink rate estimation
- Mouth Aspect Ratio (MAR) based yawn detection
- Yawn rate estimation
- Head pose analysis
- Driver distraction monitoring
- Machine Learning based fatigue classification
- Real-time fatigue alerts
- Audible alarm for sustained fatigue detection
- Live visualization of facial landmarks and monitoring parameters

---

## Motivation

Driver fatigue is a major contributor to road accidents worldwide. Traditional monitoring methods often require specialized hardware and are expensive to deploy.

This project demonstrates a low-cost computer vision solution that can monitor driver alertness using only a standard webcam.

---

## System Architecture

```text
Webcam Feed
      │
      ▼
MediaPipe Face Mesh
      │
      ▼
Facial Landmark Extraction
      │
      ▼
Feature Engineering
      │
      ▼
Machine Learning Model
      │
      ▼
Fatigue Classification
      │
      ▼
Alert & Alarm Generation
```

---

## Extracted Features

The following features are extracted in real time:

### Eye Features

- Eye Aspect Ratio (EAR)
- Blink Rate

### Mouth Features

- Mouth Aspect Ratio (MAR)
- Yawn Rate

### Head Pose Features

- Horizontal Head Deviation
- Vertical Face Ratio
- Eye Slope (Head Tilt)

---

## Dataset Collection

A custom dataset was created specifically for this project.

### Data Collection Procedure

Data was collected using the developed webcam-based monitoring system.

Two driving states were recorded:

#### Alert State (Label = 0)

- Eyes open
- Normal blinking
- Looking straight
- No yawning

#### Fatigued State (Label = 1)

- Simulated drowsiness
- Frequent yawning
- Extended eye closure
- Head tilting
- Looking down

During recording, extracted feature values were automatically stored in a CSV file.

Example dataset format:

```csv
EAR,BlinkRate,MAR,YawnRate,VerticalRatio,EyeSlope,HorizontalDiff,Fatigue
0.31,14.2,0.24,0.0,0.44,2.1,4.3,0
0.17,31.5,0.71,4.8,0.63,15.7,12.4,1
```

---

## Machine Learning Model

A Logistic Regression classifier was trained using the collected dataset.

### Input Features

- EAR
- Blink Rate
- MAR
- Yawn Rate
- Vertical Ratio
- Eye Slope
- Horizontal Difference

### Target Label

```text
0 = Alert
1 = Fatigued
```

### Training Workflow

```text
CSV Dataset
      │
      ▼
Train-Test Split
      │
      ▼
Logistic Regression Training
      │
      ▼
Model Evaluation
      │
      ▼
Real-Time Deployment
```

---

## Feature Importance Analysis

The Logistic Regression model automatically learned the contribution of each feature toward fatigue prediction.

The learned coefficients were analyzed to determine feature importance and validate the effectiveness of the selected fatigue indicators.

This data-driven approach replaced manually assigned feature weights and allowed the system to learn fatigue patterns directly from collected data.

---

## Real-Time Prediction

During runtime:

1. Facial landmarks are detected.
2. Features are extracted.
3. Features are passed to the trained Logistic Regression model.
4. The model predicts:

```text
0 = Alert
1 = Fatigued
```

5. If fatigue is detected continuously for a predefined number of frames, an alarm is triggered.

---

## Alert Mechanism

### Visual Alerts

- ALERT
- DROWSINESS DETECTED
- WAKE UP!

### Audible Alerts

An alarm sound is generated when fatigue persists for multiple consecutive frames.

This reduces false alarms caused by temporary blinks or short-term distractions.

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-Learn
- Joblib

---




## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/driver-fatigue-monitoring-system.git
cd driver-fatigue-monitoring-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python fatigue_monitor.py
```

Press:

```text
q
```

to exit the application.

---

## Applications

- Driver Monitoring Systems
- Advanced Driver Assistance Systems (ADAS)
- Fleet Safety Monitoring
- Automotive Safety Research
- Human Attention Monitoring
- Industrial Operator Monitoring

---

## Future Work

- Random Forest and XGBoost based models
- Deep Learning based fatigue prediction
- Multi-person monitoring
- Infrared/night-time monitoring
- Mobile deployment
- Cloud-based monitoring dashboard
- Integration with vehicle telemetry data

---

## Author

Prathvi Singh Rajawat (24117105) , 
Tushar Kumar (24119056)

  
Indian Institute of Technology Roorkee

---

## License

This project is released under the MIT License.
