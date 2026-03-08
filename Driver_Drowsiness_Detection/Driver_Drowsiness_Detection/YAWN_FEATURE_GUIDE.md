# Yawn Feature Guide (Simplified)

For complete setup and run steps, use `README.md` in this folder.

## Minimal commands

```bash
python prepare_combined_dataset.py
python train.py
python evaluate_eye.py
python train_yawn.py
python evaluate_yawn.py
python detect.py
```

## Main behavior
- Eye model predicts open/closed.
- Yawn model predicts yawn/no-yawn.
- Alarm triggers when eye score is high **or** yawn count >= 3.

## Tune in `detect.py`
- `EYE_DROWSY_SCORE_THRESHOLD`
- `YAWN_PROBABILITY_THRESHOLD`
- `YAWN_CONSECUTIVE_FRAMES`
- `YAWN_EVENT_LIMIT`
