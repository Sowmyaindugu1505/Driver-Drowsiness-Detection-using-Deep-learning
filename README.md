<div align="center">
  <h1>🚗 Driver Drowsiness Detection System</h1>
  <p><i>A robust, real-time Computer Vision system to detect driver fatigue using Eye and Yawn analysis.</i></p>
  
  <br />
</div>

> **Note:** This project has been rebuilt from the ground up for simplicity, modularity, and error-free execution.

---

## 📋 Features
- **Real-Time Detection:** Uses OpenCV and Haar Cascades for lightning-fast face and feature tracking.
- **Dual Drowsiness Metrics:** 
  - Tracks **Eye Closure** duration using a custom CNN.
  - Tracks **Yawn Frequency** using a separate custom CNN.
- **Audio & Visual Alarms:** Triggers an alarm sound (`alarm.wav`) and on-screen warnings when thresholds are exceeded.
- **Unified & Flat Architecture:** Say goodbye to deeply nested folders. One config file rules them all.

---

## 🛠️ Prerequisites

Ensure you have Python `3.9` or higher installed. You also need a working webcam for live detection.

```bash
# Install required libraries
pip install tensorflow opencv-python pygame
```

---

## 🚀 Quick Start Guide

### 1. Data Preparation
Place the raw dataset `archive.zip` in the root folder of this project.

Run the elegant dataset preparation script to automatically extract and structure your training data:
```bash
python prepare_dataset.py
```
*This will create `data_eyes/` and `data_yawns/` with proper `train/valid/test` splits.*

### 2. Train the Models
We use a unified training script. You must train both models before running the detector.

**Train the Eye Model:**
```bash
python train.py --target eyes
```

**Train the Yawn Model:**
```bash
python train.py --target yawns
```
*Models are saved automatically inside the `models/` directory.*

### 3. Evaluate the Models (Optional)
Check how well your models perform on the unseen test split:
```bash
python evaluate.py --target eyes
python evaluate.py --target yawns
```

### 4. Run Live Detection
Start the real-time detector. Ensure your webcam is unobstructed and the room is well-lit.
```bash
python detect.py
```

- **Eyes Closed** too long? 🚨 ALARM!
- **Too Many Yawns**? 🚨 ALARM!

*Press **`q`** to quit the detection window.*

---

## ⚙️ Configuration (`config.py`)

All core settings are centralized in `config.py`. You never need to hunt for hardcoded values again! 

**Key Hyperparameters to Tune:**
* `EYE_DROWSY_SCORE_THRESHOLD`: How many consecutive "closed" frames trigger an alert (default: 15).
* `YAWN_PROBABILITY_THRESHOLD`: Confidence required to count a yawn (default: 0.7).
* `YAWN_EVENT_LIMIT`: Number of recent yawns that trigger an alert (default: 3).

---

## 📁 Project Structure

```text
.
├── config.py             # Centralized settings and hyperparameters
├── prepare_dataset.py    # Auto-extracts & splits archive.zip
├── train.py              # Unified CNN training script
├── evaluate.py           # Unified CNN evaluation script
├── detect.py             # Live webcam driver drowsiness detector
├── README.md             # This comprehensive guide
├── alarm.wav             # Alert sound triggered on drowsiness
├── archive.zip           # Raw dataset (user provided)
├── haarcascades/         # OpenCV face/eye detection cascades
└── models/               # Auto-generated directory for trained .h5 models
```

---

<div align="center">
  <p><i>Drive Safe. Stay Awake.</i></p>
</div>
