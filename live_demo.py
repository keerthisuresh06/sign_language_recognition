"""live_demo.py - webcam -> MediaPipe landmarks -> GRU -> predicted sign on screen."""
import os, sys, csv, time, datetime, urllib.request
from collections import deque, Counter
import numpy as np
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# ===================== settings you can tweak =====================
MODEL_DIR     = "models"
CKPT_PATH     = f"{MODEL_DIR}/sign_model.pt"
CAM_INDEX     = 0          # try 1 if the wrong/no camera opens
T_OUT         = 32         # must match training
WINDOW_SEC    = 1.8        # how much recent video counts as "one sign" (try 1.2-2.5)
MIN_FRAMES    = 12         # need at least this many frames in the window
MIN_HAND_FRAC = 0.25       # hands must be visible in this fraction of the window
PREDICT_EVERY = 3          # run the model every N frames
HISTORY       = 5          # how many recent predictions to look at
STABLE_MIN    = 3          # this many must agree before a word is shown
CONF_THRESH   = 0.30       # change live with the [ and ] keys
HOLD_SEC      = 1.0 
def arg(name, default=""):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
TARGET = arg("--target").lower()      # the word you are attempting       # keep a shown word on screen this long

# ===================== 1. landmark models =====================
URLS = {
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}
os.makedirs(MODEL_DIR, exist_ok=True)
for name, url in URLS.items():
    path = f"{MODEL_DIR}/{name}"
    if not os.path.exists(path):
        print("Downloading", name)
        urllib.request.urlretrieve(url, path)

def make_detectors():
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=f"{MODEL_DIR}/pose_landmarker_full.task"),
        running_mode=vision.RunningMode.VIDEO))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=f"{MODEL_DIR}/hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    return pose, hands

# ===================== 2. feature code: SAME as training =====================
def to_xyz(landmarks, aspect):
    a = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    a[:, 0] *= aspect
    return a

def frame_to_vector(p, h, aspect):
    """One frame -> 225 numbers: [pose 99 | left hand 63 | right hand 63]. NaN = missing."""
    vec = np.full(225, np.nan, dtype=np.float32)
    if not p.pose_landmarks:
        return vec
    pose = to_xyz(p.pose_landmarks[0], aspect)
    vec[:99] = pose.ravel()
    wrist_L, wrist_R = pose[15, :2], pose[16, :2]
    best = {}
    for hl in h.hand_landmarks:
        hp = to_xyz(hl, aspect)
        w = hp[0, :2]
        dL, dR = np.linalg.norm(w - wrist_L), np.linalg.norm(w - wrist_R)
        side, d = ("L", dL) if dL <= dR else ("R", dR)
        if side not in best or d < best[side][0]:
            best[side] = (d, hp)
    if "L" in best: vec[99:162] = best["L"][1].ravel()
    if "R" in best: vec[162:225] = best["R"][1].ravel()
    return vec

def normalize(raw, start, end, T=T_OUT):
    n = len(raw)
    s = max(start - 1, 0) if start > 0 else 0
    e = min(end, n) if end > 0 else n
    if e - s >= 8:
        raw = raw[s:e]
    n = len(raw)
    pts = raw.reshape(n, 75, 3)
    l_sh, r_sh = pts[:, 11, :], pts[:, 12, :]
    center = (l_sh + r_sh) / 2
    width = np.linalg.norm((l_sh - r_sh)[:, :2], axis=1)
    valid = ~np.isnan(width)
    if valid.sum() == 0:
        return None
    idx = np.arange(n)
    for c in range(3):
        center[:, c] = np.interp(idx, idx[valid], center[valid, c])
    scale = np.median(width[valid])
    if scale < 1e-3:
        return None
    pts = (pts - center[:, None, :]) / scale
    pts = np.nan_to_num(pts, nan=0.0)
    sel = np.linspace(0, n - 1, T).round().astype(int)
    return pts[sel].reshape(T, 225).astype(np.float32)

# ===================== 3. the classifier =====================
class SignRNN(nn.Module):
    def __init__(self, n_classes, rnn_type="gru", hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(225, hidden), nn.ReLU(), nn.Dropout(dropout))
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
        self.rnn = rnn(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))
    def forward(self, x):
        out, _ = self.rnn(self.inp(x))
        return self.head(out.mean(dim=1))

# weights_only=False because the file also stores the class-name list (it's your own file)
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
class_names = [str(c) for c in ckpt["class_names"]]
model = SignRNN(len(class_names), ckpt["rnn_type"], ckpt["hidden"], ckpt["layers"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
torch.set_num_threads(2)         # leave CPU for MediaPipe

if "--list" in sys.argv:         # python live_demo.py --list
    print(", ".join(class_names)); sys.exit()

@torch.no_grad()
def predict_probs(x):
    return torch.softmax(model(torch.from_numpy(x).unsqueeze(0)), dim=-1)[0].numpy()

# ===================== 4. drawing helpers =====================
def draw_landmarks(frame, p, hd):
    h, w = frame.shape[:2]
    if p.pose_landmarks:
        pl = p.pose_landmarks[0]
        pts = {i: (int(pl[i].x * w), int(pl[i].y * h)) for i in (11, 12, 13, 14, 15, 16)}
        for a, b in [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
            cv2.line(frame, pts[a], pts[b], (255, 200, 0), 2)
        for pt in pts.values():
            cv2.circle(frame, pt, 4, (255, 200, 0), -1)
    for hl in hd.hand_landmarks:
        for l in hl:
            cv2.circle(frame, (int(l.x * w), int(l.y * h)), 3, (0, 255, 0), -1)

def put(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

# ===================== 5. main loop =====================
def main():
    conf_thresh = CONF_THRESH
    mirror_view = True
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam. Try CAM_INDEX = 1."); return
    pose, hands = make_detectors()

    buf = deque()                       # (time, 225-vector) for the last WINDOW_SEC seconds
    hist = deque(maxlen=HISTORY)        # recent probability vectors
    t0, last_ts, frame_i = time.time(), -1, 0
    prev, fps = time.time(), 0.0
    top3, status = [], "warming up"
    shown_word, shown_conf, shown_until = "", 0.0, 0.0
    n_logged = 0    

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6); prev = now
        h_, w_ = frame.shape[:2]

        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = max(int((now - t0) * 1000), last_ts + 1); last_ts = ts   # must strictly increase
        p = pose.detect_for_video(img, ts)
        hd = hands.detect_for_video(img, ts)

        buf.append((now, frame_to_vector(p, hd, w_ / h_)))
        while buf and now - buf[0][0] > WINDOW_SEC:
            buf.popleft()

        frame_i += 1
        if frame_i % PREDICT_EVERY == 0:
            if len(buf) < MIN_FRAMES:
                status = "warming up"
            else:
                raw = np.stack([v for _, v in buf])
                hand_frac = (~np.isnan(raw[:, 99:])).any(axis=1).mean()
                if hand_frac < MIN_HAND_FRAC:
                    hist.clear(); top3 = []; status = "show your hands"
                else:
                    x = normalize(raw, 0, -1)
                    if x is None:
                        status = "body not visible"
                    else:
                        hist.append(predict_probs(x)); status = ""
            if hist:
                avg = np.mean(hist, axis=0)
                top3 = [(class_names[i], float(avg[i])) for i in np.argsort(-avg)[:3]]
                votes = Counter(int(np.argmax(pr)) for pr in hist)
                idx, n = votes.most_common(1)[0]
                if n >= STABLE_MIN and avg[idx] >= conf_thresh:
                    shown_word, shown_conf, shown_until = class_names[idx], float(avg[idx]), now + HOLD_SEC

        draw_landmarks(frame, p, hd)
        disp = cv2.flip(frame, 1) if mirror_view else frame     # flip only the picture we SHOW
        put(disp, f"FPS: {fps:.0f}   threshold: {conf_thresh:.2f}", (10, 28), 0.6, (0, 255, 255))
        if status:
            put(disp, status, (10, 58), 0.7, (0, 165, 255))
        for k, (name, pr) in enumerate(top3):
            put(disp, f"{k + 1}. {name} {pr:.0%}", (10, 90 + 28 * k), 0.65, (200, 200, 200), 1)
        if now < shown_until:
            put(disp, f"{shown_word.upper()}  ({shown_conf:.0%})", (10, h_ - 25), 1.5, (0, 255, 0), 3)
        cv2.imshow("Sign recognition  (q quit | [ ] threshold | m mirror)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        elif key == ord("["): conf_thresh = max(0.05, conf_thresh - 0.05)
        elif key == ord("]"): conf_thresh = min(0.95, conf_thresh + 0.05)
        elif key == ord("m"): mirror_view = not mirror_view
        elif key == ord(" ") and TARGET:
            top_names = [n for n, _ in top3]
            shown = shown_word if now < shown_until else ""
            new = not os.path.exists("results.csv")
            with open("results.csv", "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "who", "condition", "target", "shown", "top3",
                                "correct_shown", "correct_in_top3"])
                w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                            arg("--who"), arg("--cond"), TARGET, shown, "|".join(top_names),
                            int(shown == TARGET), int(TARGET in top_names)])
            n_logged += 1

    pose.close(); hands.close(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()