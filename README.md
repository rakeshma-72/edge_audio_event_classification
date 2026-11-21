SoundSense: Edge Audio Event Classification

Detects everyday sounds (clap, dog bark, car horn, siren, alarm, whistle, speech) from a microphone and raises a local alert. Everything runs on the laptop CPU, offline, with no cloud and no cost. The only download is the pretrained YAMNet model (about 15 MB, once).

How it works
The microphone is read at 16 kHz in 1-second windows, one every 0.5 s.
Quiet windows are skipped (silence gate) and the rest are amplified automatically, because laptop microphones are weak.
Google's YAMNet (MobileNet-based, Apache 2.0) scores 521 AudioSet classes per window.
Related classes are grouped into our target sounds (for example "Vehicle horn, car horn, honking" and "Air horn, truck horn" become horn), each with its own score threshold.
Smoothing: a sound only counts as detected if it wins enough recent windows. Short sounds are handled specially (claps confirm on one window; horns and alarms need two; any single window scoring 0.40 or higher confirms immediately).
A detection prints a line, optionally writes a row to a CSV file, and for alarm, horn, dog bark, whistle and siren also beeps (3-second cooldown per sound).
Setup (Windows)

Use Python 3.12 (TensorFlow does not support the newest Python versions yet).

py -3.12 -m venv venv312
venv312\Scripts\activate
pip install -r requirements.txt

If PowerShell blocks activation, either run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass first, or skip activation and call venv312\Scripts\python.exe directly.

Run
python soundsense.py                         # live microphone, text output
python soundsense.py --gui                   # live microphone, small status window
python soundsense.py --log detections.csv    # also save every detection to a CSV file
python soundsense.py --file clip.wav         # analyse a recorded clip
python soundsense.py --verbose               # show YAMNet's top-3 classes per window
python soundsense.py --list-devices          # list microphones
python soundsense.py --device 4 --gain 2     # choose a microphone, boost the input

The first run downloads YAMNet and needs internet once. After that it works offline.

CSV log columns: time (wall clock), audio_time (seconds since start), sound, score (best confidence in the recent windows). Repeated detections of the same sound within 3 seconds are logged once.

The --gui window shows the current sound in large text (red for alert sounds, green for others), a confidence bar for the latest window, and a list of recent detections. It needs tkinter, which is included with the standard python.org Windows installer.

Microphone troubleshooting
If everything shows as silence, check the peak value with --verbose. A clap should reach 0.05 or more. If it stays near 0.001, raise the input volume in Windows (Settings, System, Sound, Input) and allow desktop apps to use the microphone (Settings, Privacy and security, Microphone).
Try each input with --list-devices and --device N. On the test laptop the "Primary Sound Capture Driver" worked best; the Intel and Realtek entries with strange huge values are drivers that do not accept 16 kHz and should be ignored.
Turn off "audio enhancements" / noise suppression for the microphone. They are built for voice calls and remove claps, whistles and sirens.
Tuning

Everything is at the top of soundsense.py:

Setting	Effect
TARGETS	Which YAMNet classes make up each sound, and each score threshold
SILENCE_PEAK	Raise if fan noise causes detections; lower if real sounds show as silence
MIN_VOTES, MIN_VOTES_BY_LABEL	How many recent windows must agree (higher means fewer false alarms, slower reaction)
STRONG_SCORE	Score at which one window confirms a sound immediately
ALERT_ON	Which sounds beep
Evaluation

evaluate.py runs the real detection logic on ESC-50 clips and reports a confusion matrix, per-class recall and precision, overall accuracy, false alarm rate and inference time.

python evaluate.py --esc50 "C:\path\to\ESC-50-master" --out eval_results
python evaluate.py --esc50 "C:\path\to\ESC-50-master" --seed 1 --per-class 30 --out eval_results_fair
python evaluate.py --esc50 "..." --extra my_clips      # your own wavs in my_clips\<label>\*.wav

ESC-50 has no whistle or speech class, so those two are not part of the benchmark (use --extra with your own recordings to test them).

Results
Run	Clips	Accuracy	False alarm rate
v1: first version	140	80.7%	5.0%
v2: horn/alarm voting and threshold tuning	140	85.0%	5.0%
v3: single strong window confirms	140	87.1%	5.0%
Fair test (different random sample, --seed 1)	190	86.3%	2.5%

Per-class results on the 190-clip test:

Sound	Clips	Recall	Precision
clap	30	97%	97%
dog bark	30	100%	100%
siren	30	90%	90%
alarm	30	87%	84%
horn	30	43%	100%
non-target sounds (no alarm expected)	40	98% correct	n/a

Speed: about 20 to 28 ms per 1-second window on a laptop CPU, so it runs in real time using a small fraction of the available time.

Limitations
Horn recall is low (43%). For some ESC-50 horn clips YAMNet describes the sound as generic "Vehicle" or "Car", so no threshold can catch them without also reacting to engine and traffic noise. Fine-tuning a small classifier on YAMNet embeddings is the natural next step.
Siren and alarm are sometimes confused with each other.
The thresholds were tuned while looking at ESC-50 clips, and the "fair test" sample overlaps with the first one (ESC-50 has only 40 clips per class), so the numbers are a consistency check and not a fully held-out test.
ESC-50 results are on clean recorded clips. Live accuracy depends on the microphone, room noise, and how loud and close the sound is. Sounds played through laptop speakers are distorted and detect worse than real ones.
Speech and whistle were tested live only, not on a benchmark.
Licenses
YAMNet: Apache License 2.0. AudioSet ontology labels: CC BY 4.0.
ESC-50 dataset: CC BY-NC (non-commercial use). Check this before using it in a product.
Python libraries used are open source.
Files
soundsense.py: the detector (live microphone, --gui window, --file, --log)
evaluate.py: ESC-50 benchmark script
requirements.txt: Python packages
## Licenses and Acknowledgements

- **YAMNet**: pretrained audio event classifier by Google, loaded through
  TensorFlow Hub (https://tfhub.dev/google/yamnet/1). Apache License 2.0.
  Source: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- **AudioSet** (the data YAMNet was trained on): dataset CC BY 4.0, ontology
  (class labels) CC BY-SA 4.0.
- **ESC-50** (used for evaluation): CC BY-NC, non-commercial use only.
  It is not included in this repository; download it from its official source.
- Python libraries used are open source under their own licenses.

The YAMNet model is downloaded at runtime and is not redistributed here.