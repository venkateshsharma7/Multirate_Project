import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const API = "http://localhost:5000/api";

const SIGNALS = ["ecg", "accelerometer", "temperature"];

const SIGNAL_COLOR = {
  ecg: "#e05c4b",
  accelerometer: "#4b9fe0",
  temperature: "#e0b04b",
};

const SIGNAL_LABEL = {
  ecg: "ECG",
  accelerometer: "Accelerometer",
  temperature: "Temperature",
};

function StatCard({ label, value, unit = "", accent }) {
  return (
    <div style={{
      background: "#0d0d0d",
      border: `1px solid ${accent}33`,
      borderLeft: `3px solid ${accent}`,
      borderRadius: 8,
      padding: "14px 18px",
      minWidth: 140,
    }}>
      <div style={{ color: "#666", fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ color: accent, fontSize: 26, fontFamily: "'DM Mono', monospace", fontWeight: 500 }}>
        {value}<span style={{ fontSize: 13, color: "#555", marginLeft: 4 }}>{unit}</span>
      </div>
    </div>
  );
}

function RewardChart({ data, height = 200 }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ color: "#555", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
        RL reward over episodes ({data.length} pts)
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}>
          <XAxis dataKey="ep" hide />
          <YAxis tick={{ fill: "#444", fontSize: 10 }} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ background: "#111", border: "1px solid #222", borderRadius: 6, fontSize: 12 }}
            formatter={(v) => [v.toFixed(4), "reward"]}
          />
          <Line
            type="monotone"
            dataKey="reward"
            stroke="#4caf50"
            dot={false}
            strokeWidth={1.5}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function SignalChart({ data, accent, height = 200 }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ color: "#555", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
        Original vs Reconstructed
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}>
          <XAxis dataKey="i" hide />
          <YAxis tick={{ fill: "#444", fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: "#111", border: "1px solid #222", borderRadius: 6, fontSize: 12 }}
            labelStyle={{ color: "#555" }}
          />
          <Line type="monotone" dataKey="original" stroke={accent} dot={false} strokeWidth={1.5} isAnimationActive={false} />
          <Line type="monotone" dataKey="reconstructed" stroke="#ffffff44" dot={false} strokeWidth={1.5} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function App() {
  const [activeSignal, setActiveSignal] = useState("ecg");
  const [running, setRunning] = useState(false);
  const [signalData, setSignalData] = useState([]);
  const [rewardData, setRewardData] = useState([]);
  const [stats, setStats] = useState({
    compression_ratio: "—",
    reconstruction_error: "—",
    reward: "—",
    epsilon: "—",
    wavelet: "—",
    level: "—",
    threshold_mode: "—",
    episode: 0,
    avg_reward: "—",
  });
  const intervalRef = useRef(null);
  const rewardRef = useRef([]);

  const runStep = useCallback(async () => {
    try {
      const res = await fetch(`${API}/step/${activeSignal}`);
      const d = await res.json();

      // Signal chart
      const step = Math.max(1, Math.floor(d.original.length / 64));
      const pts = [];
      for (let i = 0; i < d.original.length; i += step) {
        pts.push({ i, original: +d.original[i].toFixed(4), reconstructed: +d.reconstructed[i].toFixed(4) });
      }
      setSignalData(pts);

      // Reward — use ref to avoid stale closure, then sync to state
      const newPoint = { ep: d.agent.episode, reward: +d.reward };
      rewardRef.current = [...rewardRef.current, newPoint].slice(-80);
      setRewardData([...rewardRef.current]);

      setStats({
        compression_ratio: d.compression_ratio,
        reconstruction_error: d.reconstruction_error,
        reward: d.reward,
        epsilon: d.agent.epsilon,
        wavelet: d.wavelet,
        level: d.level,
        threshold_mode: d.threshold_mode,
        episode: d.agent.episode,
        avg_reward: d.agent.avg_reward,
      });
    } catch (e) {
      console.error("API error:", e);
    }
  }, [activeSignal]);

  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(runStep, 800);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [running, runStep]);

  const accent = SIGNAL_COLOR[activeSignal];

  return (
    <div style={{
      minHeight: "100vh",
      background: "#080808",
      color: "#e0e0e0",
      fontFamily: "'DM Sans', sans-serif",
      padding: "32px 40px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #111; } ::-webkit-scrollbar-thumb { background: #333; }
        button { cursor: pointer; font-family: inherit; }
      `}</style>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 36 }}>
        <div>
          <div style={{ color: "#444", fontSize: 11, letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: 6 }}>
            IoT Adaptive Compression
          </div>
          <h1 style={{ margin: 0, fontSize: 28, fontWeight: 300, letterSpacing: "-0.02em", color: "#fff" }}>
            RL Wavelet <span style={{ color: accent }}>Dashboard</span>
          </h1>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: running ? "#4caf50" : "#444",
            boxShadow: running ? "0 0 8px #4caf5088" : "none",
            transition: "all 0.3s",
          }} />
          <span style={{ color: "#555", fontSize: 13 }}>{running ? "Live" : "Paused"}</span>
          <span style={{ color: "#333", margin: "0 4px" }}>·</span>
          <span style={{ color: "#555", fontSize: 13 }}>Episode {stats.episode}</span>
        </div>
      </div>

      {/* Signal selector + controls */}
      <div style={{ display: "flex", gap: 10, marginBottom: 28, alignItems: "center" }}>
        {SIGNALS.map(s => (
          <button
            key={s}
            onClick={() => {
              setActiveSignal(s);
              setSignalData([]);
              setRewardData([]);
              rewardRef.current = [];
            }}
            style={{
              padding: "8px 18px",
              borderRadius: 6,
              border: `1px solid ${activeSignal === s ? SIGNAL_COLOR[s] : "#222"}`,
              background: activeSignal === s ? `${SIGNAL_COLOR[s]}18` : "transparent",
              color: activeSignal === s ? SIGNAL_COLOR[s] : "#555",
              fontSize: 13,
              transition: "all 0.2s",
            }}
          >
            {SIGNAL_LABEL[s]}
          </button>
        ))}
        <div style={{ flex: 1 }} />
        <button
          onClick={() => setRunning(r => !r)}
          style={{
            padding: "8px 24px",
            borderRadius: 6,
            border: `1px solid ${running ? "#555" : accent}`,
            background: running ? "transparent" : `${accent}22`,
            color: running ? "#888" : accent,
            fontSize: 13,
            fontWeight: 500,
            transition: "all 0.2s",
          }}
        >
          {running ? "⏸ Pause" : "▶ Run"}
        </button>
        <button
          onClick={async () => {
            await fetch(`${API}/reset`, { method: "POST" });
            setRewardData([]);
            setSignalData([]);
            rewardRef.current = [];
            setStats(s => ({ ...s, episode: 0 }));
          }}
          style={{
            padding: "8px 18px",
            borderRadius: 6,
            border: "1px solid #222",
            background: "transparent",
            color: "#444",
            fontSize: 13,
          }}
        >
          Reset
        </button>
      </div>

      {/* Stats row */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 32 }}>
        <StatCard label="Compression ratio" value={stats.compression_ratio} unit="×" accent={accent} />
        <StatCard label="Recon. error (MSE)" value={stats.reconstruction_error} accent="#888" />
        <StatCard label="Reward" value={stats.reward} accent={stats.reward > 0 ? "#4caf50" : "#e05c4b"} />
        <StatCard label="Epsilon ε" value={stats.epsilon} accent="#888" />
        <StatCard label="Avg reward" value={stats.avg_reward ?? "—"} accent="#888" />
        <div style={{
          background: "#0d0d0d",
          border: "1px solid #1a1a1a",
          borderLeft: `3px solid ${accent}55`,
          borderRadius: 8,
          padding: "14px 18px",
          minWidth: 180,
        }}>
          <div style={{ color: "#666", fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 8 }}>
            Agent decision
          </div>
          <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 13, color: "#ccc", lineHeight: 1.8 }}>
            <span style={{ color: accent }}>{stats.wavelet || "—"}</span>{" "}
            <span style={{ color: "#555" }}>·</span>{" "}
            level <span style={{ color: accent }}>{stats.level || "—"}</span>{" "}
            <span style={{ color: "#555" }}>·</span>{" "}
            <span style={{ color: "#888" }}>{stats.threshold_mode || "—"}</span>
          </div>
        </div>
      </div>

      {/* Charts grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        <div style={{ background: "#0d0d0d", border: "1px solid #181818", borderRadius: 10, padding: 20 }}>
          <SignalChart data={signalData} accent={accent} height={200} />
          {signalData.length === 0 && (
            <div style={{ textAlign: "center", color: "#333", fontSize: 13, paddingBottom: 60 }}>
              Press Run to start streaming
            </div>
          )}
        </div>

        <div style={{ background: "#0d0d0d", border: "1px solid #181818", borderRadius: 10, padding: 20 }}>
          <RewardChart data={rewardData} height={200} />
          {rewardData.length === 0 && (
            <div style={{ textAlign: "center", color: "#333", fontSize: 13, paddingBottom: 60 }}>
              Reward curve appears here
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div style={{ marginTop: 28, color: "#2a2a2a", fontSize: 11, letterSpacing: "0.06em" }}>
        BACKEND → python app.py &nbsp;·&nbsp; FRONTEND → npm run dev &nbsp;·&nbsp; AM.EN.U4ECE23161
      </div>
    </div>
  );
}