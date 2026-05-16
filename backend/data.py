import numpy as np
import wfdb

# ── Real PhysioNet data (MIT-BIH Arrhythmia Database) ──
# Record 100 is the most commonly cited ECG record in literature.
# wfdb downloads it automatically from PhysioNet on first run,
# then caches it locally — no manual download needed.

CHUNK_SIZE = 256  # samples per chunk

def load_physionet_ecg():
    """
    Downloads and loads MIT-BIH record 100 from PhysioNet.
    Returns the first channel (MLII lead) as a float32 array.
    Normalised to [0, 1] range for consistent compression behaviour.
    """
    print("Loading PhysioNet MIT-BIH record 100...")
    try:
        record = wfdb.rdrecord("100", pn_dir="mitdb")
        signal = record.p_signal[:, 0].astype(np.float32)  # MLII lead
        # Normalise
        signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
        print(f"ECG loaded: {len(signal)} samples at {record.fs} Hz")
        return signal
    except Exception as e:
        print(f"PhysioNet download failed: {e}")
        print("Falling back to simulated ECG...")
        return generate_ecg_simulated()


def generate_ecg_simulated(n_samples=10000):
    """Fallback simulated ECG if PhysioNet is unavailable."""
    t = np.linspace(0, 1, n_samples)
    base = 0.1 * np.sin(2 * np.pi * 1.2 * t)
    peaks = np.zeros(n_samples)
    for i in range(0, n_samples, 100):
        if i + 10 < n_samples:
            peaks[i:i+10] += np.exp(-0.5 * ((np.arange(10) - 5) / 1.5) ** 2)
    noise = 0.02 * np.random.randn(n_samples)
    signal = base + peaks + noise
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    return signal.astype(np.float32)


def generate_accelerometer(n_samples=10000):
    """Simulated accelerometer — no public equivalent needed for this project."""
    t = np.linspace(0, 10, n_samples)
    signal = (
        np.sin(2 * np.pi * 0.5 * t) +
        0.3 * np.sin(2 * np.pi * 2.0 * t) +
        0.05 * np.random.randn(n_samples)
    )
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    return signal.astype(np.float32)


def generate_temperature(n_samples=10000):
    """Simulated temperature sensor signal."""
    t = np.linspace(0, 1, n_samples)
    signal = (
        25 + 5 * np.sin(2 * np.pi * 0.1 * t) +
        0.1 * np.random.randn(n_samples)
    )
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    return signal.astype(np.float32)


# ── Load all signals once at startup ──
print("Initialising signal sources...")
SIGNALS = {
    "ecg":           load_physionet_ecg(),   # real PhysioNet data
    "accelerometer": generate_accelerometer(),
    "temperature":   generate_temperature(),
}

_pointers = {k: 0 for k in SIGNALS}


def get_signal_type(signal_name):
    mapping = {"ecg": 0, "accelerometer": 1, "temperature": 2}
    return mapping.get(signal_name, 0)


def stream_chunk(signal_name="ecg"):
    """Returns next CHUNK_SIZE samples, looping at end of signal."""
    signal = SIGNALS[signal_name]
    ptr    = _pointers[signal_name]
    chunk  = signal[ptr: ptr + CHUNK_SIZE]

    if len(chunk) < CHUNK_SIZE:
        leftover = CHUNK_SIZE - len(chunk)
        chunk = np.concatenate([chunk, signal[:leftover]])
        _pointers[signal_name] = leftover
    else:
        _pointers[signal_name] = ptr + CHUNK_SIZE

    return chunk


def reset_stream(signal_name=None):
    if signal_name:
        _pointers[signal_name] = 0
    else:
        for k in _pointers:
            _pointers[k] = 0


# ── Quick test ──
if __name__ == "__main__":
    chunk = stream_chunk("ecg")
    print(f"ECG chunk shape : {chunk.shape}")
    print(f"ECG min/max     : {chunk.min():.4f} / {chunk.max():.4f}")
    print(f"Source          : PhysioNet MIT-BIH record 100")
    chunk2 = stream_chunk("accelerometer")
    print(f"Accel chunk     : {chunk2.shape}")
    print("data.py OK")