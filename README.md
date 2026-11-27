# Real-time sign language recognition

Recognises individual signs from webcam video and shows the predicted word live
on screen.

Pipeline: webcam frame → MediaPipe pose + hand landmarks → sequence of 32 frames
→ GRU → top predictions with confidence

## Results (WLASL100 test split, mean of 3 seeds)

| Model | Top-1 | Top-5 |
|---|---|---|
| Logistic regression on time-averaged landmarks | 23.0% | 54.0% |
| GRU over landmark sequences | **29.7% ± 0.9%** | **71.0%** |

The GRU uses motion, which the baseline ignores by design.

## How it works

- **Landmarks, not pixels:** MediaPipe (Tasks API) extracts 33 pose and 2×21
  hand landmarks per frame, cached once as NumPy arrays.
- **Normalisation:** centred on the shoulder midpoint and scaled by shoulder
  width, so position and distance from the camera don't matter. Missing hands
  are zero-filled; every clip is resampled to 32 frames.
- **Augmentation:** mirroring (with left/right hand swap), small rotations,
  scaling.
- **Live demo:** sliding window of recent frames; a word is shown only when
  confidence is high and stable across several predictions.

## Run locally

    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python live_demo.py

Model files are not yet in this repository 
a download link is coming.

## Limitations

- **Isolated signs only:** recognises single words from a fixed vocabulary of
  100, not continuous signing or sentences.
- **Small dataset:** 1,013 WLASL100 videos could be processed.
- **Limited live testing:** so far one person, one background, one lighting
  condition. Live accuracy appears lower than test accuracy.

## Future work

- Larger vocabulary (e.g. WLASL300 or WLASL2000)
- Continuous sign language recognition
- Live evaluation with more signers and conditions

Dataset: WLASL. Check its licence terms before reuse.
