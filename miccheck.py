import sounddevice as sd, numpy as np
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        try:
            x = sd.rec(int(2 * 16000), samplerate=16000, channels=1, device=i, dtype="float32")
            print(f"Device {i} ({d['name']}): clap now...", end=" ", flush=True)
            sd.wait()
            print(f"peak = {np.abs(x).max():.4f}")
        except Exception as e:
            print(f"Device {i} failed: {e}")