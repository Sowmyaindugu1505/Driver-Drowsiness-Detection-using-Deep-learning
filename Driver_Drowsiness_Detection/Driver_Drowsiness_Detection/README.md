# Driver Drowsiness Detection (Simple Structure)

This project detects drowsiness using **two simple signals**:
1. Eye closure (open/closed model)
2. Yawn count (yawn/no-yawn model)

An alarm is triggered if either:
- eye score is high enough, or
- yawn events are 3 or more.

## Project structure

- `app_config.py` → single place for paths
- `prepare_combined_dataset.py` → prepares both eye + yawn datasets from `archive.zip`
- `train.py` → trains eye model (`models/cnnCat2.h5`)
- `evaluate_eye.py` → evaluates eye model on `data/test`
- `train_yawn.py` → trains yawn model (`models/yawn_cnn.h5`) and stores class mapping
- `evaluate_yawn.py` → evaluates yawn model on `data_yawn/test`
- `detect.py` → live webcam detection + alarm
- `Driver_Drowsiness_Detection_Colab.ipynb` → end-to-end Colab workflow

## Quick start (local)

1. Install dependencies
```bash
pip install tensorflow opencv-python pygame
```

2. Put `archive.zip` in this folder.

3. Prepare dataset
```bash
python prepare_combined_dataset.py
```

4. Train + evaluate eye model
```bash
python train.py
python evaluate_eye.py
```

5. Train + evaluate yawn model
```bash
python train_yawn.py
python evaluate_yawn.py
```

6. Run live detector
```bash
python detect.py
```
Press `q` to stop.

## Notes
- Dataset split is deterministic (seed=42).
- If yawn class order changes, `detect.py` reads `models/yawn_class_indices.json` automatically.

## Merge-readiness checklist

Before approving a PR, run:

```bash
git ls-files -u
rg -n "^(<<<<<<<|=======|>>>>>>>)" *.py *.md *.ipynb
```

Both commands should return no unresolved conflicts.

## Manual PR merge via command line

If GitHub shows conflicts, run:

```bash
git pull origin main
git checkout codex/add-yawn-detection-to-drowsiness-model-5mc8qx
git merge main
```

Then resolve conflicts in editor, add files, and commit:

```bash
git add .
git commit -m "Resolve merge conflicts with main"
git push -u origin codex/add-yawn-detection-to-drowsiness-model-5mc8qx
```

If your local repo has no `origin` remote configured, add it first with:

```bash
git remote add origin https://github.com/Sowmyaindugu1505/Driver-Drowsiness-Detection-using-Deep-learning.git
```
