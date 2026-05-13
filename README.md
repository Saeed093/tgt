# Target-Hit MVP

MVP target-hit detection system with a React tactical dashboard and a FastAPI + OpenCV backend.

## Project structure

```text
target-hit-mvp/
  backend/
    app/
      main.py
      camera_manager.py
      target_detector.py
      hit_detector.py
      score_engine.py
      session_manager.py
      schemas.py
    requirements.txt
    run_backend.bat
  frontend/
    src/
      App.jsx
      main.jsx
      styles.css
      components/
        CameraPanel.jsx
        ControlPanel.jsx
        ScorePanel.jsx
        StatusBar.jsx
    package.json
    run_frontend.bat
  README.md
```

## 1) Run on Windows

### Backend

```bat
cd backend
python -m venv venv
REM Use venv Python so OpenCV (cv2) is found — do not run uvicorn with a different Python.
venv\Scripts\python.exe -m pip install -r requirements.txt
REM Default: 127.0.0.1:8080 — avoids WinError 10013 on some Windows machines for 0.0.0.0:8000
venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
```

Or:

```bat
backend\run_backend.bat
```

### Frontend

```bat
cd frontend
npm install
npm run dev
```

Or:

```bat
frontend\run_frontend.bat
```

Open [http://localhost:5173](http://localhost:5173). The dev server reads `frontend/.env.development` so the UI calls the backend on port **8080**.

## Troubleshooting: `WinError 10013` (socket access forbidden)

Windows may block binding to **`0.0.0.0:8000`** (reserved TCP range, Hyper-V, VPN, or policy). This project defaults to **`127.0.0.1:8080`** instead.

If you still get **10013**:

1. Use another free port, e.g. `8765`: `uvicorn app.main:app --reload --host 127.0.0.1 --port 8765`
2. Set `VITE_API_BASE` and `VITE_WS_URL` in `frontend/.env.development` to that port and restart `npm run dev`.
3. Inspect excluded ranges (elevated PowerShell): `netsh interface ipv4 show excludedportrange protocol=tcp`

### `ModuleNotFoundError: No module named 'cv2'`

Uvicorn was started with **global Python** instead of **`backend\venv`**. Use `venv\Scripts\python.exe -m uvicorn ...` or run `backend\run_backend.bat`.

## 2) Use a USB webcam

- In `Camera Source`, enter an integer camera index like `0` or `1`.
- Keep `Target Mode` as `Auto`, or force `Figure 1`/`Figure 2`.
- Click `Start`.

## 3) Use an IP camera

- In `Camera Source`, enter URL such as:
  - `rtsp://user:pass@192.168.1.20:554/stream1`
  - `http://192.168.1.20:8080/video`
- Click `Start`.

## 4) How hit detection works

- A session folder is created under `backend/temp/session_<timestamp>/`.
- Once target detection is stable (or manually selected), backend stores a clean `reference.jpg`.
- For each new frame:
  - Convert reference/current frames to grayscale
  - Compute darkening difference (`reference - current`)
  - Threshold new dark pixels
  - Morphology open/close to reduce noise
  - Contour extraction + filters (area, compactness, aspect ratio)
  - Best contour is treated as hit marker
- Backend draws a red bounding box around the hit marker and streams the annotated frame through `/ws`.

## 5) Modify scoring zones later

Scoring zones are in `backend/app/score_engine.py` inside `self.zones`.

- Zones are normalized:
  - `center`: `(x, y)` in `[0, 1]` relative to frame size
  - `radius`: normalized radial distance
  - `score`: integer points
- Separate zone sets exist for `figure_1` and `figure_2`.
- Edit/add zones and restart backend.

## 6) Known limitations (MVP)

- One active camera pipeline at a time (code is shaped for expansion).
- Auto target detection uses template matching and needs templates at:
  - `backend/app/templates/figure_1.jpg`
  - `backend/app/templates/figure_2.jpg`
- No persistent database; all stats reset when backend restarts.
- Reference update is simple and optimized for marker-style hits, not full ballistic analytics.
- Frame alignment is minimal in MVP and may be sensitive to major camera movement.

## API endpoints

- `POST /start`
- `POST /stop`
- `GET /status`
- `GET /ws`

### WebSocket payload

```json
{
  "frame": "base64_encoded_jpeg",
  "target_type": "figure_1",
  "status": "monitoring",
  "hit_detected": true,
  "bbox": [12, 43, 20, 19],
  "last_score": 8,
  "total_score": 18,
  "tries": 3
}
```
