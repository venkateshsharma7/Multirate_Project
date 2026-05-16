import numpy as np
import pywt

# ── Action space ──
# The RL agent picks one action = one combination of (wavelet, level, threshold_mode)
# We define all valid combinations here as a list.
# Action 0 = first combo, Action 1 = second combo, etc.

WAVELET_FAMILIES = ["haar", "db4", "sym4", "coif2"]
DECOMP_LEVELS    = [2, 3, 4]
THRESHOLD_MODES  = ["soft", "hard"]

# Build the full action space: all combinations
ACTION_SPACE = [
    (w, l, t)
    for w in WAVELET_FAMILIES
    for l in DECOMP_LEVELS
    for t in THRESHOLD_MODES
]
# Total actions = 4 wavelet × 3 levels × 2 threshold = 24 actions

N_ACTIONS = len(ACTION_SPACE)  # 24


def action_to_params(action_idx):
    """Converts an integer action index to (wavelet, level, threshold_mode)."""
    return ACTION_SPACE[action_idx]


def compress(signal, action_idx, threshold_fraction=0.3):
    """
    Compresses a signal chunk using the wavelet params chosen by the RL agent.

    Steps:
      1. Decompose signal into wavelet coefficients
      2. Zero out small coefficients (thresholding — this is the actual compression)
      3. Count how many coefficients survived (compression ratio)
      4. Reconstruct signal from thresholded coefficients

    Returns:
        compressed_coeffs : thresholded coefficient list (what you'd store/transmit)
        reconstructed     : signal rebuilt from those coefficients
        compression_ratio : float — higher is better (more zeros = smaller file)
        reconstruction_error : float — lower is better (closer to original)
    """
    wavelet, level, threshold_mode = action_to_params(action_idx)

    # Step 1 — Wavelet decomposition
    # coeffs is a list: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    # cA = approximation (low-freq), cD = details (high-freq)
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Step 2 — Thresholding
    # We only threshold the detail coefficients (cD), not the approximation (cA)
    # threshold = fraction of max absolute value in all detail coeffs
    all_details = np.concatenate([c for c in coeffs[1:]])
    threshold_val = threshold_fraction * np.max(np.abs(all_details))

    compressed_coeffs = [coeffs[0]]  # keep approximation as-is
    for detail in coeffs[1:]:
        thresholded = pywt.threshold(detail, threshold_val, mode=threshold_mode)
        compressed_coeffs.append(thresholded)

    # Step 3 — Compression ratio
    # How many coefficients are non-zero after thresholding?
    total_coeffs   = sum(len(c) for c in coeffs)
    nonzero_coeffs = sum(np.count_nonzero(c) for c in compressed_coeffs)
    zero_fraction  = 1.0 - (nonzero_coeffs / total_coeffs)
    compression_ratio = 1.0 / (1.0 - zero_fraction + 1e-6)  # higher = better

    # Step 4 — Reconstruct
    reconstructed = pywt.waverec(compressed_coeffs, wavelet)
    reconstructed = reconstructed[:len(signal)]  # trim to original length

    # Reconstruction error — Mean Squared Error
    reconstruction_error = float(np.mean((signal - reconstructed) ** 2))

    return compressed_coeffs, reconstructed, float(compression_ratio), reconstruction_error


def compute_reward(compression_ratio, reconstruction_error, latency=0.0,
                   alpha=0.6, beta=0.35, gamma=0.05):
    """
    The reward signal the RL agent uses to learn.

    reward = alpha * compression_ratio
           - beta  * reconstruction_error (scaled)
           - gamma * latency

    We scale reconstruction_error by 100 so it's on a similar magnitude
    to compression_ratio (which is typically 1.5 to 5.0).

    alpha + beta + gamma should sum to 1.0 (they're weights).
    You can tune these based on what matters more for your use case.
    """
    reward = (
        alpha * compression_ratio
        - beta  * (reconstruction_error * 100)
        - gamma * latency
    )
    return float(reward)


# ── Quick test ──
if __name__ == "__main__":
    print(f"Total actions in action space: {N_ACTIONS}")
    print(f"Action 0: {action_to_params(0)}")
    print(f"Action 12: {action_to_params(12)}")

    # Fake signal
    signal = np.sin(np.linspace(0, 4 * np.pi, 256)).astype(np.float32)

    _, recon, cr, err = compress(signal, action_idx=5)
    print(f"\nCompression ratio: {cr:.3f}")
    print(f"Reconstruction error (MSE): {err:.6f}")
    reward = compute_reward(cr, err)
    print(f"Reward: {reward:.4f}")
    print("compress.py OK")