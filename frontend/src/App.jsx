import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

const API = "/api";

const SIGNALS = ["ecg", "accelerometer", "temperature"];

const SIGNAL_COLOR = {
  ecg: "#ff4d4d",        // Brighter neon red
  accelerometer: "#3399ff", // Brighter neon blue
  temperature: "#ffcc00",   // Brighter neon yellow
};

const SIGNAL_LABEL = {
  ecg: "ECG",
  accelerometer: "Accelerometer",
  temperature: "Temperature",
};

// Sleeker Stat Card
function StatCard({ label, value, unit = "", accent, bg = "#111" }) {
  return (
    <div style={{
      background: bg,
      border: `1px solid ${accent}22`,
      borderRadius: 12,
      padding: "16px 20px",
      flex: 1,
      display: "flex",
      flexDirection: "column",
      justifyContent: "space-between"
    }}>
      <div style={{ color: "#888", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
        {label}
      </div>
      <div style={{ color: accent, fontSize: 28, fontFamily: "'DM Mono', monospace", fontWeight: 500 }}>
        {value}<span style={{ fontSize: 14, color: "#666", marginLeft: 4 }}>{unit}</span>
      </div>
    </div>
  );
}

// Visual Progress Bar for Epsilon
function EpsilonGauge({ value }) {
  const percent = typeof value === 'number' ? value * 100 : 0;
  return (
    <div style={{ background: "#111", border: "1px solid #222", borderRadius: 12, padding: "16px 20px", marginBottom: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <span style={{ color: "#888", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase" }}>Epsilon (Exploration)</span>
        <span style={{ color: "#ccc", fontSize: 12, fontFamily: "'DM Mono', monospace" }}>{typeof value === 'number' ? value.toFixed(3) : "—"}</span>
      </div>
      <div style={{ width: "100%", height: 6, background: "#222", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${percent}%`, height: "100%", background: "linear-gradient(90deg, #9c27b0, #00bcd4)", transition: "width 0.3s" }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, color: "#555", fontSize: 10 }}>
        <span>Exploiting</span>
        <span>Exploring</span>
      </div>
    </div>
  );
}

export default function App() {
  const [activeSignal, setActiveSignal] = useState("ecg");
  const [running, setRunning] = useState(false);
  const [signalData, setSignalData] = useState([]);
  const [rewardData, setRewardData] = useState([]);
  const [actionLog, setActionLog] = useState([]); // New feature: keep track of recent actions
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

      // Signal chart downsampling
      const step = Math.max(1, Math.floor(d.original.length / 128)); // Smoother render
      const pts = [];
      for (let i = 0; i < d.original.length; i += step) {
        pts.push({ i, original: +d.original[i].toFixed(4), reconstructed: +d.reconstructed[i].toFixed(4) });
      }
      setSignalData(pts);

      // Reward & Log
      const newPoint = { ep: d.agent.episode, reward: +d.reward };
      rewardRef.current = [...rewardRef.current, newPoint].slice(-100); // keep last 100
      setRewardData([...rewardRef.current]);

      // Update Action Log
      setActionLog(prev => [
        { ep: d.agent.episode, wave: d.wavelet, lvl: d.level, th: d.threshold_mode },
        ...prev
      ].slice(0, 4));

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
      height: "100vh",
      width: "100vw",
      background: "#050505",
      color: "#e0e0e0",
      fontFamily: "'DM Sans', sans-serif",
      padding: "24px",
      display: "flex",
      flexDirection: "column",
      overflow: "hidden"
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; } 
        ::-webkit-scrollbar-track { background: transparent; } 
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
        button { cursor: pointer; font-family: inherit; outline: none; }
        button:hover { filter: brightness(1.2); }
        button:active { transform: scale(0.98); }
      `}</style>

      {/* Header */}
      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: "60px", flexShrink: 0, marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <h1 style={{ fontSize: 24, fontWeight: 500, letterSpacing: "-0.02em", color: "#fff" }}>
            RL-WaveComp
          </h1>
          <span style={{ color: "#666", fontSize: 12, letterSpacing: "0.1em", textTransform: "uppercase" }}>Adaptive Telemetry</span>
        </div>
        
        <div style={{ display: "flex", alignItems: "center", gap: 12, background: "#111", padding: "8px 16px", borderRadius: 20, border: "1px solid #222" }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: running ? "#4caf50" : "#f44336", boxShadow: running ? "0 0 10px #4caf50" : "none", transition: "all 0.3s" }} />
          <span style={{ color: running ? "#fff" : "#888", fontSize: 13, fontWeight: 500 }}>{running ? "SYSTEM LIVE" : "SYSTEM PAUSED"}</span>
          <div style={{ width: 1, height: 16, background: "#333", margin: "0 8px" }} />
          <span style={{ color: "#aaa", fontSize: 13, fontFamily: "'DM Mono', monospace" }}>EPISODE {stats.episode}</span>
        </div>
      </header>

      {/* Main App Grid */}
      <main style={{ display: "flex", gap: 24, flex: 1, minHeight: 0 }}>
        
        {/* Left Sidebar (Controls & Context) */}
        <aside style={{ width: 340, display: "flex", flexDirection: "column", flexShrink: 0 }}>
          
          {/* Signal Selector */}
          <div style={{ background: "#111", border: "1px solid #222", borderRadius: 12, padding: 20, marginBottom: 20 }}>
            <div style={{ color: "#888", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12 }}>Data Modality</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {SIGNALS.map(s => (
                <button key={s} onClick={() => {
                  setActiveSignal(s); setSignalData([]); setRewardData([]); rewardRef.current = []; setActionLog([]);
                }} style={{
                  padding: "10px 16px", borderRadius: 8, textAlign: "left", fontSize: 14, fontWeight: 500,
                  border: `1px solid ${activeSignal === s ? SIGNAL_COLOR[s] : "#222"}`,
                  background: activeSignal === s ? `${SIGNAL_COLOR[s]}15` : "transparent",
                  color: activeSignal === s ? SIGNAL_COLOR[s] : "#777",
                  transition: "all 0.2s", display: "flex", justifyContent: "space-between", alignItems: "center"
                }}>
                  {SIGNAL_LABEL[s]}
                  {activeSignal === s && <span style={{ width: 6, height: 6, borderRadius: "50%", background: SIGNAL_COLOR[s] }}/>}
                </button>
              ))}
            </div>
            
            <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
              <button onClick={() => setRunning(r => !r)} style={{
                flex: 2, padding: "12px", borderRadius: 8, border: "none", fontSize: 13, fontWeight: 600,
                background: running ? "#333" : accent, color: running ? "#fff" : "#000", transition: "all 0.2s",
              }}>
                {running ? "Pause Execution" : "Start Agent"}
              </button>
              <button onClick={async () => {
                await fetch(`${API}/reset`, { method: "POST" });
                setRewardData([]); setSignalData([]); setActionLog([]); rewardRef.current = []; setStats(s => ({ ...s, episode: 0 }));
              }} style={{ flex: 1, padding: "12px", borderRadius: 8, border: "1px solid #333", background: "transparent", color: "#888", fontSize: 13 }}>
                Reset
              </button>
            </div>
          </div>

          <EpsilonGauge value={stats.epsilon} />

          {/* Action Log */}
          <div style={{ background: "#111", border: "1px solid #222", borderRadius: 12, padding: 20, flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            <div style={{ color: "#888", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12 }}>Agent Decision Log</div>
            <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
              {actionLog.length === 0 ? (
                <div style={{ color: "#444", fontSize: 12, fontStyle: "italic" }}>Waiting for data...</div>
              ) : actionLog.map((log, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 12px", background: i === 0 ? `${accent}15` : "#181818", borderLeft: `3px solid ${i === 0 ? accent : "#333"}`, borderRadius: "0 6px 6px 0", fontSize: 12, fontFamily: "'DM Mono', monospace" }}>
                  <span style={{ color: "#666" }}>Ep {log.ep}</span>
                  <span style={{ color: i === 0 ? accent : "#aaa" }}>{log.wave} • L{log.lvl} • {log.th}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* Right Content Area (Charts & Stats) */}
        <section style={{ flex: 1, display: "flex", flexDirection: "column", gap: 20, minWidth: 0 }}>
          
          {/* Top Stats Row */}
          <div style={{ display: "flex", gap: 16, flexShrink: 0 }}>
            <StatCard label="Compression Ratio" value={stats.compression_ratio} unit="×" accent={accent} />
            <StatCard label="Mean Squared Error" value={stats.reconstruction_error} accent="#aaa" />
            <StatCard label="Step Reward" value={stats.reward} accent={stats.reward > 0 ? "#4caf50" : "#ff4d4d"} />
            <StatCard label="Avg Reward (Roll)" value={stats.avg_reward ?? "—"} accent="#00bcd4" />
          </div>

          {/* Signal Chart Container */}
          <div style={{ flex: 2, background: "#111", border: "1px solid #222", borderRadius: 12, padding: "20px 24px", display: "flex", flexDirection: "column", position: "relative" }}>
            <div style={{ color: "#888", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12 }}>Real-time Signal Reconstruction</div>
            <div style={{ flex: 1, minHeight: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={signalData} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                  <XAxis dataKey="i" hide />
                  <YAxis tick={{ fill: "#666", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: "#050505", border: "1px solid #333", borderRadius: 8, fontSize: 12, fontFamily: "'DM Mono', monospace" }} />
                  <Line type="monotone" dataKey="original" stroke={accent} dot={false} strokeWidth={2} isAnimationActive={false} />
                  <Line type="monotone" dataKey="reconstructed" stroke="#ffffff" opacity={0.6} strokeDasharray="4 4" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
              {signalData.length === 0 && <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "#444", fontSize: 14 }}>Awaiting datastream...</div>}
            </div>
          </div>

          {/* Reward Chart Container */}
          <div style={{ flex: 1, background: "#111", border: "1px solid #222", borderRadius: 12, padding: "20px 24px", display: "flex", flexDirection: "column", position: "relative" }}>
            <div style={{ color: "#888", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12 }}>Q-Learning Convergence</div>
            <div style={{ flex: 1, minHeight: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rewardData} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                  <XAxis dataKey="ep" hide />
                  <YAxis tick={{ fill: "#666", fontSize: 11 }} axisLine={false} tickLine={false} domain={["auto", "auto"]} />
                  <Tooltip contentStyle={{ background: "#050505", border: "1px solid #333", borderRadius: 8, fontSize: 12, fontFamily: "'DM Mono', monospace" }} formatter={(v) => [v.toFixed(4), "Reward"]} />
                  <Line type="stepAfter" dataKey="reward" stroke="#4caf50" dot={false} strokeWidth={2} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
              {rewardData.length === 0 && <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "#444", fontSize: 14 }}>Initializing agent...</div>}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}