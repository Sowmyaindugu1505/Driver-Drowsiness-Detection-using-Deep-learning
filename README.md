<div align="center">
  <h1>🚗 Ultimate Driver Drowsiness System</h1>
  <p><i>A beautifully structured, enterprise-grade Machine Learning computer vision project.</i></p>
  <br />
</div>

> **Note:** This project has been rebuilt from the ground up for maximum simplicity, modularity, and error-free execution. It uses a single-entry-point architecture (`run.py`).

---

## 📋 Features
- **Single Command Interface:** You only ever need to execute `run.py`.
- **Real-Time Detection:** Uses OpenCV and Haar Cascades for lightning-fast face tracking.
- **Dual Neural Networks:** Tracks *Eye Closure* and *Yawn Frequency* with two custom CNNs.
- **Clean Architecture:** Strict separation of source code (`src/`), trained models (`models/`), datasets (`data/`), and static dependencies (`assets/`).

---

## 🛠️ Prerequisites

Ensure you have Python `3.9` or higher installed.

```bash
# Install the required libraries in 1 click!
pip install -r requirements.txt
```

---

## 🚀 Quick Start Guide

We have consolidated the entire project into a powerful, easy-to-use Command Line Interface.

### 1. Data Preparation
Place your raw dataset `archive.zip` inside `data/raw/`. Then simply run:

```bash
python run.py --mode prep
```

*This will extract and structure your training data nicely inside `data/processed/`.*

### 2. Train the Models
Train both models. They will automatically save inside the `models/` directory.

```bash
python run.py --mode train --target eyes
python run.py --mode train --target yawns
```

### 3. Evaluate the Models (Optional)
Check accuracy on the unseen test splits:

```bash
python run.py --mode eval --target eyes
python run.py --mode eval --target yawns
```

### 4. Run Live Detection
Start the real-time detector. Ensure your webcam is unobstructed and the room is well-lit.

```bash
python run.py --mode detect
```

- **Eyes Closed** too long? 🚨 ALARM!
- **Too Many Yawns**? 🚨 ALARM!

*Press **`q`** to quit the detection window.*

---

## 📁 Project Structure

```text
.
├── run.py                # 🔥 The ONLY script you need to execute 🔥
├── requirements.txt      # 1-click dependency installation
├── README.md             # This comprehensive guide
├── src/                  # The Python source code
│   ├── config.py         # Centralized hyperparameters & thresholds
│   ├── data_prep.py      # Zip extraction & splitting logic
│   ├── model.py          # Unified CNN training logic
│   ├── evaluate.py       # Metrics and accuracy
│   └── detector.py       # Live webcam tracking logic
├── data/                 
│   ├── raw/archive.zip   # Your raw data goes here!
│   └── processed/        # Auto-generated image datasets
├── assets/               
│   ├── haarcascades/     # OpenCV XML cascades
│   └── alarm.wav         # Alert sound
└── models/               # Auto-generated directory for .h5 models
```

---

## ⚙️ Configuration (`src/config.py`)

All core settings are centralized in `src/config.py`. 

**Key Hyperparameters to Tune:**
* `EYE_DROWSY_SCORE_THRESHOLD`: (Default: 15) Consecutive "closed" frames required for an alert.
* `YAWN_PROBABILITY_THRESHOLD`: (Default: 0.7) Confidence required to count a yawn.
* `YAWN_EVENT_LIMIT`: (Default: 3) Number of recent yawns that trigger an alarm.

<div align="center">
  <br />
  <p><i>Drive Safe. Stay Awake.</i></p>
</div>
