import { useEffect, useRef, useState } from "react";
import CameraPanel from "./components/CameraPanel";
import ControlPanel from "./components/ControlPanel";
import ScorePanel from "./components/ScorePanel";
import StatusBar from "./components/StatusBar";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8080";
const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8080/ws";
const WS_RECONNECT_MS = 2000;

const IP_CAM_STORAGE_KEY = "target-hit-ip-cameras";

const fallbackCameras = [
  { index: 0, label: "Camera 0" },
  { index: 1, label: "Camera 1" },
  { index: 2, label: "Camera 2" },
  { index: 3, label: "Camera 3" },
];

function isStreamUrl(value) {
  const t = String(value).trim().toLowerCase();
  return (
    t.startsWith("rtsp://") ||
    t.startsWith("rtsps://") ||
    t.startsWith("http://") ||
    t.startsWith("https://")
  );
}

function normalizeSavedIpCameras(raw) {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((x) => x && typeof x.url === "string" && x.url.trim())
    .map((x) => ({
      id: typeof x.id === "string" && x.id ? x.id : crypto.randomUUID(),
      name: typeof x.name === "string" ? x.name.trim() : "",
      url: x.url.trim(),
    }));
}

function loadSavedIpCameras() {
  try {
    return normalizeSavedIpCameras(
      JSON.parse(localStorage.getItem(IP_CAM_STORAGE_KEY) || "[]")
    );
  } catch {
    return [];
  }
}

const initialFeed = {
  frame: "",
  target_type: "unknown",
  status: "idle",
  hit_detected: false,
  bbox: null,
  last_score: 0,
  total_score: 0,
  tries: 0,
  actual_resolution: "",
  motion: 0,
  stable: false,
};

export default function App() {
  const [cameraSource, setCameraSource] = useState("0");
  const [webcams, setWebcams] = useState(fallbackCameras);
  const [savedIpCameras, setSavedIpCameras] = useState(loadSavedIpCameras);
  const [targetMode, setTargetMode] = useState("auto");
  const [busy, setBusy] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [feed, setFeed] = useState(initialFeed);
  const wsRef = useRef(null);
  const reconnectRef = useRef(null);

  const systemRunning = !["idle", "error: camera not found"].includes(feed.status);

  const connectWs = () => {
    if (wsRef.current && wsRef.current.readyState < 2) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => setWsConnected(true);
    ws.onmessage = (event) => {
      try {
        setFeed((prev) => ({ ...prev, ...JSON.parse(event.data) }));
      } catch {}
    };
    ws.onclose = () => {
      setWsConnected(false);
      reconnectRef.current = setTimeout(connectWs, WS_RECONNECT_MS);
    };
    ws.onerror = () => ws.close();
  };

  useEffect(() => {
    connectWs();
    return () => {
      clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, []);

  const loadWebcams = async () => {
    const c = new AbortController();
    const tid = setTimeout(() => c.abort(), 3000);
    try {
      const res = await fetch(`${API_BASE}/cameras`, { signal: c.signal });
      const data = await res.json();
      setWebcams(
        Array.isArray(data.cameras) && data.cameras.length > 0
          ? data.cameras
          : fallbackCameras
      );
    } catch {
      setWebcams(fallbackCameras);
    } finally {
      clearTimeout(tid);
    }
  };

  useEffect(() => {
    loadWebcams();
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(IP_CAM_STORAGE_KEY, JSON.stringify(savedIpCameras));
    } catch {
      /* ignore quota / private mode */
    }
  }, [savedIpCameras]);

  const addIpCamera = (name, url) => {
    const u = url.trim();
    if (!u) {
      alert("Enter a stream URL.");
      return false;
    }
    if (!isStreamUrl(u)) {
      alert(
        "URL must start with rtsp://, rtsps://, http://, or https:// (typical IP camera / NVR streams)."
      );
      return false;
    }
    if (savedIpCameras.some((c) => c.url === u)) {
      alert("That URL is already in your list.");
      return false;
    }
    setSavedIpCameras((prev) => [
      ...prev,
      { id: crypto.randomUUID(), name: name.trim(), url: u },
    ]);
    setCameraSource(u);
    return true;
  };

  const removeIpCamera = (id) => {
    const cam = savedIpCameras.find((c) => c.id === id);
    if (cam && cameraSource === cam.url) setCameraSource("0");
    setSavedIpCameras((prev) => prev.filter((c) => c.id !== id));
  };

  const callApi = async (path, body) => {
    setBusy(true);
    const c = new AbortController();
    const tid = setTimeout(() => c.abort(), 6000);
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
        signal: c.signal,
      });
      const data = await res.json();
      if (data.status !== "ok") throw new Error(data.message || "Failed");
      return true;
    } catch (err) {
      if (err.name === "AbortError") {
        alert("Backend did not respond. Is uvicorn running?");
      } else {
        alert(err.message);
      }
      return false;
    } finally {
      clearTimeout(tid);
      setBusy(false);
    }
  };

  return (
    <main className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark">TGT</span>
          <span className="brand-title">TARGET HIT DETECTION</span>
          <span className="brand-sub">/ TACTICAL FEED</span>
        </div>
        <StatusBar
          status={feed.status}
          targetType={feed.target_type}
          wsConnected={wsConnected}
          stable={feed.stable}
          motion={feed.motion}
          systemRunning={systemRunning}
        />
      </header>

      <div className="layout">
        <ControlPanel
          source={cameraSource}
          webcams={webcams}
          savedIpCameras={savedIpCameras}
          onAddIpCamera={addIpCamera}
          onRemoveIpCamera={removeIpCamera}
          targetMode={targetMode}
          onSourceChange={setCameraSource}
          onTargetModeChange={setTargetMode}
          onRefreshWebcams={loadWebcams}
          onStart={() =>
            callApi("/start", { camera_source: cameraSource, target_mode: targetMode })
          }
          onStop={async () => {
            const ok = await callApi("/stop");
            if (ok) {
              setFeed((prev) => ({
                ...prev,
                status: "idle",
                frame: "",
                hit_detected: false,
                bbox: null,
                target_type: "unknown",
                actual_resolution: "",
                motion: 0,
                stable: false,
              }));
            }
          }}
          onCheckNow={() => callApi("/check_now")}
          busy={busy}
          systemRunning={systemRunning}
        />

        <CameraPanel
          frame={feed.frame}
          status={feed.status}
          actualResolution={feed.actual_resolution}
          motion={feed.motion}
          stable={feed.stable}
          hitDetected={feed.hit_detected}
          targetType={feed.target_type}
        />
      </div>

      <ScorePanel
        tries={feed.tries}
        lastScore={feed.last_score}
        totalScore={feed.total_score}
        hitDetected={feed.hit_detected}
        stable={feed.stable}
        systemRunning={systemRunning}
      />
    </main>
  );
}
