"""
Evaluate SoundSense on ESC-50 clips (and optionally your own recordings).
 
It reuses the exact detection logic from soundsense.py (windowing, auto-gain,
class mapping, smoothing), so the numbers describe the real system.
 
ESC-50 has no whistle or speech classes, so those two are evaluated from your
own recordings via --extra (see below).
 
Usage:
    python evaluate.py --esc50 "D:\\datasets\\ESC-50-master"
    python evaluate.py --esc50 "D:\\datasets\\ESC-50-master" --per-class 30 --neg 60
    python evaluate.py --esc50 "..." --extra my_clips
 
--extra folder layout (wav files only, folder name = true label):
    my_clips/whistle/w1.wav, w2.wav, ...
    my_clips/speech/s1.wav, ...
    my_clips/clap/c1.wav, ...      (any target label works)
 
Outputs (in --out, default "eval_results"):
    results.csv, confusion_matrix.csv, confusion_matrix.png (if matplotlib is installed)
"""
 
import argparse
import collections
import contextlib
import csv
import io
import os
import random
import time
from math import gcd
 
import numpy as np
 
import soundsense as ss
 
# our target label -> ESC-50 category
ESC50_MAP = {
    "clap": "clapping",
    "dog bark": "dog",
    "horn": "car_horn",
    "siren": "siren",
    "alarm": "clock_alarm",
}
 
# ESC-50 categories that should NOT trigger any target (false-alarm test)
NEGATIVE_CATEGORIES = [
    "rain", "wind", "keyboard_typing", "footsteps", "vacuum_cleaner",
    "sea_waves", "clock_tick", "washing_machine", "mouse_click", "water_drops",
]
 
NEG = "negative"
 
 
def load_audio(path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
 
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
    if sr != ss.SAMPLE_RATE:
        g = gcd(sr, ss.SAMPLE_RATE)
        data = resample_poly(data, ss.SAMPLE_RATE // g, sr // g).astype(np.float32)
    return data
 
 
def classify_clip(det, data):
    """Run the real detector over a clip. Returns (predicted_label, n_windows, seconds)."""
    win = int(ss.WINDOW_SEC * ss.SAMPLE_RATE)
    hop = int(ss.HOP_SEC * ss.SAMPLE_RATE)
    if len(data) < win:
        data = np.pad(data, (0, win - len(data)))
 
    det.history.clear()
    confirmed = []
    n = 0
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):      # silence per-window printing
        for start in range(0, len(data) - win + 1, hop):
            c = det.step(data[start:start + win], f"{start / ss.SAMPLE_RATE:5.1f}s")
            n += 1
            if c:
                confirmed.append(c)
    elapsed = time.perf_counter() - t0
 
    pred = collections.Counter(confirmed).most_common(1)[0][0] if confirmed else "none"
    return pred, n, elapsed
 
 
def collect_items(args):
    items = []   # (path, true_label)
    rng = random.Random(args.seed)
 
    meta_path = os.path.join(args.esc50, "meta", "esc50.csv")
    audio_dir = os.path.join(args.esc50, "audio")
    if not os.path.isfile(meta_path) or not os.path.isdir(audio_dir):
        raise SystemExit(f"ESC-50 folder not found or incomplete: {args.esc50}\n"
                         "Expected <folder>/meta/esc50.csv and <folder>/audio/*.wav")
 
    by_cat = collections.defaultdict(list)
    with open(meta_path, newline="") as f:
        for row in csv.DictReader(f):
            by_cat[row["category"]].append(row["filename"])
 
    for label, cat in ESC50_MAP.items():
        files = sorted(by_cat.get(cat, []))
        if not files:
            print(f"warning: category '{cat}' not found in ESC-50 metadata")
            continue
        for fn in rng.sample(files, min(args.per_class, len(files))):
            items.append((os.path.join(audio_dir, fn), label))
 
    neg_pool = []
    for cat in NEGATIVE_CATEGORIES:
        neg_pool += [os.path.join(audio_dir, fn) for fn in by_cat.get(cat, [])]
    for p in rng.sample(neg_pool, min(args.neg, len(neg_pool))):
        items.append((p, NEG))
 
    if args.extra:
        for label in sorted(os.listdir(args.extra)):
            folder = os.path.join(args.extra, label)
            if not os.path.isdir(folder):
                continue
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(".wav"):
                    items.append((os.path.join(folder, fn), label))
    return items
 
 
def main():
    ap = argparse.ArgumentParser(description="Evaluate SoundSense on ESC-50")
    ap.add_argument("--esc50", required=True, help="path to the ESC-50-master folder")
    ap.add_argument("--per-class", type=int, default=20, help="clips per target class")
    ap.add_argument("--neg", type=int, default=40, help="number of negative (non-target) clips")
    ap.add_argument("--extra", help="folder of your own wavs: <extra>/<label>/*.wav")
    ap.add_argument("--out", default="eval_results", help="output folder")
    ap.add_argument("--seed", type=int, default=0, help="random seed for clip sampling")
    args = ap.parse_args()
 
    items = collect_items(args)
    print(f"Evaluating {len(items)} clips...\n")
 
    model, names = ss.load_model()
    det = ss.Detector(model, names)
    det.maybe_alert = lambda label: None     # no beeping during evaluation
 
    rows = []
    total_windows, total_time = 0, 0.0
    for i, (path, true) in enumerate(items, 1):
        try:
            pred, n, sec = classify_clip(det, load_audio(path))
        except Exception as e:
            print(f"  skipped {os.path.basename(path)}: {e}")
            continue
        total_windows += n
        total_time += sec
        correct = (pred == "none") if true == NEG else (pred == true)
        rows.append((os.path.basename(path), true, pred, correct))
        print(f"[{i:3d}/{len(items)}] {os.path.basename(path):<28} true={true:<9} "
              f"pred={pred:<9} {'OK' if correct else 'WRONG'}")
 
    if not rows:
        raise SystemExit("No clips were evaluated.")
 
    # ---- confusion matrix ----
    true_labels = [l for l in list(ESC50_MAP) + sorted({r[1] for r in rows} - set(ESC50_MAP) - {NEG})
                   if any(r[1] == l for r in rows)]
    if any(r[1] == NEG for r in rows):
        true_labels.append(NEG)
    pred_labels = list(ss.TARGETS) + ["none"]
 
    matrix = {t: collections.Counter() for t in true_labels}
    for _, true, pred, _ in rows:
        matrix[true][pred] += 1
 
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", "predicted", "correct"])
        w.writerows(rows)
    with open(os.path.join(args.out, "confusion_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + pred_labels)
        for t in true_labels:
            w.writerow([t] + [matrix[t][p] for p in pred_labels])
 
    # ---- report ----
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (rows = true, columns = predicted)")
    print("=" * 70)
    col_w = 10
    print(f"{'':<10}" + "".join(f"{p[:col_w - 1]:>{col_w}}" for p in pred_labels))
    for t in true_labels:
        print(f"{t:<10}" + "".join(f"{matrix[t][p]:>{col_w}}" for p in pred_labels))
 
    print("\nPER-CLASS RESULTS")
    print(f"{'class':<10}{'clips':>7}{'recall':>10}{'precision':>12}")
    for t in true_labels:
        n_true = sum(matrix[t].values())
        if t == NEG:
            ok = matrix[t]["none"]
            print(f"{t:<10}{n_true:>7}{ok / n_true:>10.0%}{'-':>12}   (no false alarm)")
            continue
        recall = matrix[t][t] / n_true
        n_pred = sum(matrix[r][t] for r in true_labels)
        prec = f"{matrix[t][t] / n_pred:.0%}" if n_pred else "-"
        print(f"{t:<10}{n_true:>7}{recall:>10.0%}{prec:>12}")
 
    n_correct = sum(r[3] for r in rows)
    print(f"\nOVERALL ACCURACY: {n_correct}/{len(rows)} = {n_correct / len(rows):.1%}")
    neg_rows = [r for r in rows if r[1] == NEG]
    if neg_rows:
        fa = sum(1 for r in neg_rows if r[2] != "none")
        print(f"FALSE ALARM RATE on non-target sounds: {fa}/{len(neg_rows)} = {fa / len(neg_rows):.1%}")
    if total_windows:
        print(f"INFERENCE: {1000 * total_time / total_windows:.1f} ms per 1-second window "
              f"(CPU, includes smoothing logic; {total_windows} windows)")
    print(f"\nSaved results to: {os.path.abspath(args.out)}")
 
    # ---- optional picture ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
 
        arr = np.array([[matrix[t][p] for p in pred_labels] for t in true_labels])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(arr, cmap="Blues")
        ax.set_xticks(range(len(pred_labels)))
        ax.set_xticklabels(pred_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(true_labels)))
        ax.set_yticklabels(true_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("SoundSense confusion matrix")
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                if arr[r, c]:
                    ax.text(c, r, str(arr[r, c]), ha="center", va="center",
                            color="white" if arr[r, c] > arr.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "confusion_matrix.png"), dpi=150)
        print("Saved confusion_matrix.png")
    except ImportError:
        print("(install matplotlib to also get a confusion_matrix.png picture)")
 
 
if __name__ == "__main__":
    main()
 