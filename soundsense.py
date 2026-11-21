"""
SoundSense: edge audio event classification (CPU only, fully local).
 
Listens to the microphone (or a .wav file), runs Google's YAMNet on 1-second
windows, maps its 521 AudioSet classes to a few target sounds, smooths the
predictions over time, and triggers a local alert.
 
Usage:
    python soundsense.py                       # live microphone, text output
    python soundsense.py --gui                 # live microphone, small window
    python soundsense.py --log detections.csv  # also save every detection to a CSV file
    python soundsense.py --file clip.wav       # test on a recorded clip
    python soundsense.py --list-devices        # show audio input devices
    python soundsense.py --device 4 --verbose  # pick a mic, show top-3 raw classes
"""
 
import argparse
import collections
import csv
import datetime
import os
import queue
import sys
import threading
import time
 
import numpy as np
 
SAMPLE_RATE = 16000      # YAMNet expects 16 kHz mono
WINDOW_SEC = 1.0         # audio length per prediction
HOP_SEC = 0.5            # how often we predict
SILENCE_PEAK = 0.01      # if the loudest sample in a window is below this, treat as silence
TARGET_PEAK = 0.5        # quiet audio is amplified so its peak reaches about this level
MAX_GAIN = 25.0          # never amplify more than this (avoids boosting pure noise)
HISTORY = 5              # smoothing: look at the last N predictions...
MIN_VOTES = 3            # ...and require this many to agree
ALERT_COOLDOWN = 3.0     # seconds between repeated alerts/log lines for the same sound
 
# target label -> (AudioSet class names, score threshold)
# Names must match YAMNet's class map exactly; missing ones are reported at startup.
TARGETS = {
    "alarm":    (["Alarm", "Smoke detector, smoke alarm", "Fire alarm",
                  "Alarm clock", "Buzzer", "Car alarm"], 0.30),
    "clap":     (["Clapping", "Hands", "Finger snapping", "Slap, smack", "Applause"], 0.10),
    "horn":     (["Vehicle horn, car horn, honking", "Air horn, truck horn"], 0.15),
    "speech":   (["Speech"], 0.50),
    "dog bark": (["Bark", "Dog"], 0.25),
    "whistle":  (["Whistle", "Whistling"], 0.20),
    "siren":    (["Siren", "Emergency vehicle", "Police car (siren)",
                  "Ambulance (siren)", "Fire engine, fire truck (siren)"], 0.30),
}
 
# Short, impulsive sounds: confirm on a single window instead of waiting for several votes
FAST_LABELS = {"clap"}
 
# Short, bursty sounds (honks, beeps) need fewer agreeing windows than the default
MIN_VOTES_BY_LABEL = {"horn": 2, "alarm": 2}
 
# A very confident single window confirms immediately (catches one short honk or bark).
# Speech is excluded because it needs smoothing to avoid false triggers.
STRONG_SCORE = 0.40
 
# Which detections should raise a loud alert (the rest are just logged)
ALERT_ON = {"alarm", "horn", "dog bark", "whistle", "siren"}
 
 
def load_model():
    import tensorflow_hub as hub
 
    print("Loading YAMNet (first run downloads ~15 MB, then it is cached)...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    path = model.class_map_path().numpy().decode("utf-8")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    class_names = [r["display_name"] for r in rows]
    return model, class_names
 
 
def build_target_indices(class_names):
    index_of = {name: i for i, name in enumerate(class_names)}
    targets = {}
    for label, (names, thr) in TARGETS.items():
        idx = [index_of[n] for n in names if n in index_of]
        missing = [n for n in names if n not in index_of]
        if missing:
            print(f"  warning: class not found for '{label}': {missing}")
        if idx:
            targets[label] = (idx, thr)
    return targets
 
 
def prepare_window(window, user_gain=1.0):
    """Silence gate + auto-gain. Returns (scaled_window, peak, gain) or None if silent."""
    peak = float(np.max(np.abs(window)))
    if peak < SILENCE_PEAK:
        return None
    gain = min(TARGET_PEAK / peak, MAX_GAIN) * user_gain
    return np.clip(window * gain, -1.0, 1.0).astype(np.float32), peak, gain
 
 
def best_target(mean_scores, targets):
    """Map YAMNet's 521 class scores to our target sounds. Returns (label, score)."""
    best_label, best_score = "other", 0.0
    for label, (idx, thr) in targets.items():
        s = float(mean_scores[idx].max())
        if s >= thr and s > best_score:
            best_label, best_score = label, s
    return best_label, best_score
 
 
def confirm_label(history, label, score, targets):
    """Smoothing rule: decide whether the latest window confirms a sound."""
    strong = score >= STRONG_SCORE and label in targets and label != "speech"
    if label in FAST_LABELS or strong:
        return label
    counts = collections.Counter(l for l in history if l not in ("silence", "other"))
    if counts:
        cand, votes = counts.most_common(1)[0]
        if votes >= MIN_VOTES_BY_LABEL.get(cand, MIN_VOTES) and cand not in FAST_LABELS:
            return cand
    return None
 
 
class Detector:
    def __init__(self, model, class_names, verbose=False, gain=1.0,
                 log_path=None, quiet=False):
        self.gain = gain
        self.model = model
        self.class_names = class_names
        self.targets = build_target_indices(class_names)
        self.history = collections.deque(maxlen=HISTORY)
        self.scores = collections.deque(maxlen=HISTORY)
        self.last_alert = {}
        self.last_logged = {}
        self.verbose = verbose
        self.quiet = quiet
        self.on_result = None        # optional callback(raw_label, score, confirmed)
        self.log_file = None
        self.log_writer = None
        if log_path:
            is_new = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
            self.log_file = open(log_path, "a", newline="", encoding="utf-8")
            self.log_writer = csv.writer(self.log_file)
            if is_new:
                self.log_writer.writerow(["time", "audio_time", "sound", "score"])
                self.log_file.flush()
 
    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None
 
    def predict(self, window):
        """Return (label, score, info) for one audio window; label may be 'silence'/'other'."""
        prep = prepare_window(window, self.gain)
        if prep is None:
            peak = float(np.max(np.abs(window)))
            info = f"peak {peak:.4f} (below gate {SILENCE_PEAK})" if self.verbose else None
            return "silence", 0.0, info
        window, peak, gain = prep
 
        scores, _, _ = self.model(window)
        mean_scores = scores.numpy().mean(axis=0)   # average over frames -> (521,)
        best_label, best_score = best_target(mean_scores, self.targets)
 
        top3 = None
        if self.verbose:
            top = np.argsort(mean_scores)[::-1][:3]
            top3 = f"peak {peak:.3f} gain x{gain:.1f} | " + ", ".join(
                f"{self.class_names[i]} {mean_scores[i]:.2f}" for i in top)
        return best_label, best_score, top3
 
    def step(self, window, t_label, t_sec=None):
        label, score, top3 = self.predict(window)
        self.history.append(label)
        self.scores.append(score)
 
        # smoothing: a sound counts only if it wins enough recent windows
        confirmed = confirm_label(self.history, label, score, self.targets)
 
        if not self.quiet:
            line = f"[{t_label}] raw={label:<9} score={score:.2f}"
            if top3:
                line += f" | top3: {top3}"
            if confirmed:
                line += f"  --> DETECTED: {confirmed.upper()}"
            print(line)
 
        if confirmed:
            self.log_detection(confirmed, t_label, t_sec)
            self.maybe_alert(confirmed)
        if self.on_result:
            self.on_result(label, score, confirmed)
        return confirmed
 
    def log_detection(self, label, t_label, t_sec):
        if not self.log_writer:
            return
        now = t_sec if t_sec is not None else time.time()
        if now - self.last_logged.get(label, -1e9) < ALERT_COOLDOWN:
            return
        self.last_logged[label] = now
        best = max((s for l, s in zip(self.history, self.scores) if l == label), default=0.0)
        self.log_writer.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                                  t_label.strip(), label, f"{best:.2f}"])
        self.log_file.flush()
 
    def maybe_alert(self, label):
        now = time.time()
        if now - self.last_alert.get(label, 0) < ALERT_COOLDOWN:
            return
        self.last_alert[label] = now
        if label in ALERT_ON and not self.quiet:
            print(f"\n  *** ALERT: {label.upper()} detected! ***\n")
        if label in ALERT_ON:
            beep()
 
 
def beep():
    try:
        import winsound
        winsound.Beep(1000, 300)
    except ImportError:
        sys.stdout.write("\a")
        sys.stdout.flush()
 
 
def run_file(detector, path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd
 
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
 
    win, hop = int(WINDOW_SEC * SAMPLE_RATE), int(HOP_SEC * SAMPLE_RATE)
    if len(data) < win:
        data = np.pad(data, (0, win - len(data)))
    print(f"Analysing {path} ({len(data) / SAMPLE_RATE:.1f}s)\n")
    for start in range(0, len(data) - win + 1, hop):
        t = start / SAMPLE_RATE
        detector.step(data[start:start + win], f"{t:5.1f}s", t)
 
 
def run_mic(detector, device, stop=None):
    import sounddevice as sd
 
    win, hop = int(WINDOW_SEC * SAMPLE_RATE), int(HOP_SEC * SAMPLE_RATE)
    q = queue.Queue()
 
    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata[:, 0].copy())
 
    buf = np.zeros(win, dtype=np.float32)
    if not detector.quiet:
        print("Listening... press Ctrl+C to stop.\n")
    start = time.time()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=hop, device=device, callback=callback):
        while stop is None or not stop.is_set():
            try:
                block = q.get(timeout=0.5)
            except queue.Empty:
                continue
            buf = np.concatenate([buf[len(block):], block])[-win:]
            t = time.time() - start
            detector.step(buf, f"{t:6.1f}s", t)
 
 
class DisplayState:
    """What the live window shows (kept separate from the GUI code so it is easy to test)."""
    HOLD_SEC = 2.0
 
    def __init__(self):
        self.current = None
        self.until = 0.0
        self.raw = "silence"
        self.score = 0.0
        self.history = collections.deque(maxlen=12)
        self.version = 0
 
    def update(self, now, raw, score, confirmed):
        self.raw, self.score = raw, score
        if confirmed:
            if confirmed != self.current or now >= self.until:
                self.history.appendleft(f"{time.strftime('%H:%M:%S')}   {confirmed.upper()}")
                self.version += 1
            self.current = confirmed
            self.until = now + self.HOLD_SEC
 
    def shown(self, now):
        return self.current if now < self.until else None
 
 
def run_gui(detector, device):
    try:
        import tkinter as tk
    except ImportError:
        raise SystemExit("tkinter is not available in this Python install. "
                         "Reinstall Python with 'tcl/tk and IDLE' ticked, or run without --gui.")
 
    BG, CARD, TEXT, DIM = "#1e1e2e", "#313244", "#cdd6f4", "#a6adc8"
    detector.quiet = True
    q = queue.Queue()
    detector.on_result = lambda raw, score, conf: q.put((raw, score, conf))
    stop = threading.Event()
 
    def worker():
        try:
            run_mic(detector, device, stop)
        except Exception as e:                      # e.g. microphone not available
            q.put(("__error__", str(e), None))
 
    threading.Thread(target=worker, daemon=True).start()
 
    state = DisplayState()
    root = tk.Tk()
    root.title("SoundSense")
    root.geometry("520x520")
    root.configure(bg=BG)
 
    status = tk.Label(root, text="Listening...", font=("Segoe UI", 34, "bold"),
                      fg=TEXT, bg=BG, pady=20)
    status.pack(fill="x")
    heard = tk.Label(root, text="", font=("Segoe UI", 12), fg=DIM, bg=BG)
    heard.pack()
    bar = tk.Canvas(root, width=440, height=22, bg=CARD, highlightthickness=0)
    bar.pack(pady=8)
    fill = bar.create_rectangle(0, 0, 0, 22, fill="#89b4fa", width=0)
    tk.Label(root, text="Recent detections", font=("Segoe UI", 11),
             fg=DIM, bg=BG).pack(pady=(16, 4))
    box = tk.Listbox(root, height=12, font=("Consolas", 12), bg=CARD, fg=TEXT,
                     highlightthickness=0, borderwidth=0)
    box.pack(fill="both", expand=True, padx=20, pady=(0, 20))
 
    shown_version = {"v": -1}
    error = {"msg": None}
 
    def poll():
        now = time.time()
        try:
            while True:
                raw, score, conf = q.get_nowait()
                if raw == "__error__":
                    error["msg"] = score
                else:
                    state.update(now, raw, score, conf)
        except queue.Empty:
            pass
 
        if error["msg"]:
            status.config(text="Microphone error", fg="#f38ba8", font=("Segoe UI", 26, "bold"))
            heard.config(text=error["msg"][:70])
        else:
            current = state.shown(now)
            if current:
                status.config(text=current.upper(),
                              fg="#f38ba8" if current in ALERT_ON else "#a6e3a1")
            else:
                status.config(text="Listening...", fg=TEXT)
            heard.config(text=f"heard: {state.raw}")
            bar.coords(fill, 0, 0, int(440 * min(1.0, state.score)), 22)
        if shown_version["v"] != state.version:
            shown_version["v"] = state.version
            box.delete(0, "end")
            for item in state.history:
                box.insert("end", item)
        root.after(100, poll)
 
    def on_close():
        stop.set()
        root.destroy()
 
    root.protocol("WM_DELETE_WINDOW", on_close)
    poll()
    root.mainloop()
 
 
def main():
    ap = argparse.ArgumentParser(description="SoundSense: local audio event classifier")
    ap.add_argument("--file", help="analyse a .wav file instead of the microphone")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--verbose", action="store_true", help="show YAMNet's top-3 raw classes")
    ap.add_argument("--gui", action="store_true", help="show a small live window (microphone only)")
    ap.add_argument("--log", metavar="CSV", help="append every detection to this CSV file")
    ap.add_argument("--gain", type=float, default=1.0, help="extra input gain multiplier (e.g. 2.0)")
    args = ap.parse_args()
 
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.gui and args.file:
        raise SystemExit("--gui works with the microphone only; drop --file or --gui.")
 
    model, class_names = load_model()
    detector = Detector(model, class_names, verbose=args.verbose, gain=args.gain,
                        log_path=args.log)
    if args.log:
        print(f"Logging detections to {os.path.abspath(args.log)}")
    try:
        if args.file:
            run_file(detector, args.file)
        elif args.gui:
            run_gui(detector, args.device)
        else:
            run_mic(detector, args.device)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        detector.close()
 
 
if __name__ == "__main__":
    main()
 