from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

app = Flask(__name__, static_folder=str(FRONTEND_DIST), static_url_path="")
CORS(app)  # allows the React frontend to call this API

agent = QLearningAgent()

SIGNAL_NAMES = ["ecg", "accelerometer", "temperature"]


@app.route("/api/step/<signal_name>", methods=["GET"])
def step(signal_name):
    """
    One full pipeline step for a given signal type.
    Called by the dashboard every second to get fresh data.

    Returns everything the frontend needs:
      - original signal chunk
      - reconstructed signal
      - compression ratio
      - reconstruction error
      - reward
      - which action the agent chose
      - agent stats (epsilon, avg reward, etc.)
    """
    if signal_name not in SIGNAL_NAMES:
        return jsonify({"error": f"Unknown signal: {signal_name}"}), 400

    # 1. Get next chunk from the signal stream
    chunk = stream_chunk(signal_name)
    state = get_signal_type(signal_name)

    # 2. Agent picks an action
    t0 = time.time()
    action = agent.choose_action(state)
    wavelet, level, threshold_mode = action_to_params(action)

    # 3. Compress using chosen params
    _, reconstructed, compression_ratio, reconstruction_error = compress(chunk, action)
    latency = time.time() - t0

    # 4. Compute reward and update agent
    reward = compute_reward(compression_ratio, reconstruction_error, latency)
    next_state = state  # signal type doesn't change within a session
    agent.update(state, action, reward, next_state)

    # 5. Save Q-table every 50 steps
    if agent.episode % 50 == 0:
        agent.save()

    return jsonify({
        "signal_name":         signal_name,
        "original":            chunk.tolist(),
        "reconstructed":       reconstructed.tolist(),
        "compression_ratio":   round(compression_ratio, 4),
        "reconstruction_error": round(reconstruction_error, 6),
        "reward":              round(reward, 4),
        "action":              action,
        "wavelet":             wavelet,
        "level":               level,
        "threshold_mode":      threshold_mode,
        "agent":               agent.get_stats(),
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    """Returns just the agent stats — for the reward curve chart."""
    return jsonify(agent.get_stats())


@app.route("/api/reset", methods=["POST"])
def reset():
    """Resets the agent — useful for demo restarts."""
    global agent
    agent = QLearningAgent()
    return jsonify({"status": "reset"})


@app.route("/api/signals", methods=["GET"])
def signals():
    """Lists available signal types."""
    return jsonify({"signals": SIGNAL_NAMES})


if __name__ == "__main__":
    print("Starting Flask server on http://localhost:5000")
    print("Endpoints:")
    print("  GET  /api/step/<signal_name>   — run one compression step")
    print("  GET  /api/stats                — agent stats")
    print("  POST /api/reset                — reset agent")
    app.run(debug=True, port=5000)