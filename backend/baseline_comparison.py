import numpy as np
import json
from data import stream_chunk, get_signal_type, reset_stream, SIGNALS
from compress import compress, compute_reward, action_to_params, ACTION_SPACE, N_ACTIONS
from agent import QLearningAgent

# ── Fixed baselines ──
# These represent "existing approaches" — static param choices that don't adapt
BASELINES = {
    "haar-L2": ACTION_SPACE.index(("haar", 2, "soft")),
    "db4-L3":  ACTION_SPACE.index(("db4",  3, "soft")),
    "sym4-L4": ACTION_SPACE.index(("sym4", 4, "soft")),
}

N_CHUNKS = 200  # how many chunks to evaluate per signal


def evaluate(action_idx, signal_name, n_chunks=N_CHUNKS):
    """Runs n_chunks compression steps with a fixed action and returns avg metrics."""
    reset_stream(signal_name)
    ratios, errors, rewards = [], [], []
    for _ in range(n_chunks):
        chunk = stream_chunk(signal_name)
        _, _, cr, err = compress(chunk, action_idx)
        r = compute_reward(cr, err)
        ratios.append(cr)
        errors.append(err)
        rewards.append(r)
    return {
        "compression_ratio": round(float(np.mean(ratios)), 4),
        "mse":               round(float(np.mean(errors)), 6),
        "avg_reward":        round(float(np.mean(rewards)), 4),
    }


def get_agent_action(signal_name):
    """Loads the saved Q-table and returns the best converged action for this signal."""
    agent = QLearningAgent()
    state = get_signal_type(signal_name)
    action = agent.best_action_for(state)
    wavelet, level, mode = action_to_params(action)
    return action, f"{wavelet}-L{level}-{mode}"


def run():
    print("\n" + "=" * 65)
    print("  BASELINE vs RL AGENT — COMPRESSION BENCHMARK")
    print("=" * 65)

    results = {}

    for signal_name in ["ecg", "accelerometer", "temperature"]:
        print(f"\n── {signal_name.upper()} ──")
        results[signal_name] = {}

        # Evaluate each fixed baseline
        for label, action_idx in BASELINES.items():
            m = evaluate(action_idx, signal_name)
            results[signal_name][label] = m
            print(f"  {label:<12}  CR: {m['compression_ratio']:>7.4f}×  "
                  f"MSE: {m['mse']:.6f}  Reward: {m['avg_reward']:>7.4f}")

        # Evaluate RL agent's converged action
        agent_action, agent_label = get_agent_action(signal_name)
        m = evaluate(agent_action, signal_name)
        results[signal_name]["RL agent"] = {**m, "action": agent_label}
        print(f"  {'RL ('+agent_label+')':<12}  CR: {m['compression_ratio']:>7.4f}×  "
              f"MSE: {m['mse']:.6f}  Reward: {m['avg_reward']:>7.4f}  ← agent")

    # ── Summary table ──
    print("\n" + "=" * 65)
    print("  SUMMARY TABLE (copy into your paper)")
    print("=" * 65)
    print(f"  {'Method':<18} {'Signal':<16} {'CR':>8} {'MSE':>12} {'Reward':>10}")
    print("  " + "-" * 60)
    for signal_name, methods in results.items():
        for method, m in methods.items():
            print(f"  {method:<18} {signal_name:<16} "
                  f"{m['compression_ratio']:>8.4f} "
                  f"{m['mse']:>12.6f} "
                  f"{m['avg_reward']:>10.4f}")
    print("=" * 65)

    # Save to JSON for any further analysis
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved to baseline_results.json")


if __name__ == "__main__":
    run()