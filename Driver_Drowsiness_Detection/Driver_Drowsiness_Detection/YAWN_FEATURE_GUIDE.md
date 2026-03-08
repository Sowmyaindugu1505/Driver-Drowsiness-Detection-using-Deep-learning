# Driver Drowsiness Detection (Eye + Yawn) - Complete Run Guide

This guide gives exact steps to run and verify the full project.

## Project objective
Drowsiness alert is triggered when either condition is true:
- Eyes stay closed long enough (eye score threshold), OR
- Yawn events reach 3 or more.

---

## 1) Prerequisites

### Local machine (recommended for live webcam detection)
- Python 3.9+ 
- Webcam
- Working audio output (alarm)

Install dependencies:

```bash
pip install tensorflow opencv-python pygame
```

---

## 2) Dataset requirement (`archive.zip`)

Place `archive.zip` inside:

`Driver_Drowsiness_Detection/Driver_Drowsiness_Detection/`

The zip should contain images for:
- Eye classes: `open`, `closed` (or similar aliases)
- Yawn classes: `yawn`, `no_yawn` (or similar aliases)

---

## 3) Prepare dataset splits (automatic)

Run from inside `Driver_Drowsiness_Detection/Driver_Drowsiness_Detection/`:

```bash
python prepare_combined_dataset.py
```

What this script does:
- Extracts `archive.zip` to `datasets/archive_extracted`
- Detects matching class folders by aliases
- Creates 70/15/15 splits for both tasks:
  - Eye dataset: `data/train|valid|test/{open,closed}`
  - Yawn dataset: `data_yawn/train|valid|test/{yawn,no_yawn}`

---

## 4) Train eye model

```bash
python train.py
```

Output:
- `models/cnnCat2.h5`

---

## 5) Evaluate eye model

```bash
python evaluate_eye.py
```

Expected output:
- Eye test loss
- Eye test accuracy

---

## 6) Train yawn model

```bash
python train_yawn.py
```

Output:
- `models/yawn_cnn.h5`

---

## 7) Evaluate yawn model

```bash
python evaluate_yawn.py
```

Expected output:
- Yawn test loss
- Yawn test accuracy

---

## 8) Run live detection (combined eye + yawn)

```bash
python detect.py
```

In the window you should see:
- Eye status
- Eye score
- Yawn probability
- Yawn count

Alert condition:
- `eye_score > EYE_DROWSY_SCORE_THRESHOLD` OR
- `yawn_count >= YAWN_EVENT_LIMIT` (default 3)

Press `q` to quit.

---

## 9) Important tuning knobs (`detect.py`)

- `EYE_DROWSY_SCORE_THRESHOLD`
- `YAWN_PROBABILITY_THRESHOLD`
- `YAWN_CONSECUTIVE_FRAMES`
- `YAWN_EVENT_LIMIT`

If too many false alarms:
- Increase probability threshold / required frames.

If model misses yawns:
- Decrease probability threshold slightly.

---

## 10) Google Colab usage

Use notebook:

`Driver_Drowsiness_Detection_Colab.ipynb`

Notebook does:
- Dependency install
- Upload `archive.zip`
- Prepare dataset
- Train + evaluate eye model
- Train + evaluate yawn model
- Verify artifact files

Note: live webcam `detect.py` is best tested locally.
