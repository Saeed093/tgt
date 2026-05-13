# Start Guide (PowerShell)

This file shows how to run backend and frontend directly without `.bat` files.

**Default backend port is `8080`** (not 8000) because Windows often returns **WinError 10013** when binding to `0.0.0.0:8000` (reserved port range, Hyper-V, VPN, or security policy).

## Backend (FastAPI)

Use the **virtual environment’s Python** for `pip` and `uvicorn`. If you run `uvicorn` with a different Python (e.g. global `Python313`), you get **`ModuleNotFoundError: No module named 'cv2'`** because OpenCV is only installed inside `venv`.

```powershell
cd "E:\kinotech\tgt systye\application\target-hit-mvp\backend"
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
```

If the venv already exists:

```powershell
cd "E:\kinotech\tgt systye\application\target-hit-mvp\backend"
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
```

Optional (same interpreter after activate):

```powershell
.\venv\Scripts\activate
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
```

Backend URLs:

- API: [http://localhost:8080](http://localhost:8080)
- WebSocket: `ws://localhost:8080/ws`

### If you still see `WinError 10013`

1. Pick another port (example **8765**):

   ```powershell
   .\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8765
   ```

2. Create or edit `frontend/.env.development` so it matches:

   ```
   VITE_API_BASE=http://localhost:8765
   VITE_WS_URL=ws://localhost:8765/ws
   ```

3. Check whether Windows has reserved the port (run in **elevated** PowerShell):

   ```powershell
   netsh interface ipv4 show excludedportrange protocol=tcp
   ```

4. Temporarily try **without** `--reload` (rarely fixes bind issues):

   ```powershell
   .\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8080
   ```

### `ModuleNotFoundError: No module named 'cv2'`

You started uvicorn with **global Python** (paths like `C:\Program Files\Python313\...`) instead of **`backend\venv`**.

Fix:

```powershell
cd "E:\kinotech\tgt systye\application\target-hit-mvp\backend"
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
```

Or run `backend\run_backend.bat` (it uses the venv interpreter automatically).

## Frontend (React + Vite)

Open a second PowerShell terminal:

```powershell
cd "E:\kinotech\tgt systye\application\target-hit-mvp\frontend"
npm install
npm run dev
```

Vite loads `frontend/.env.development` so the UI talks to port **8080** by default.

Frontend URL:

- App: [http://localhost:5173](http://localhost:5173)

## Typical local workflow

1. Start backend terminal first.
2. Start frontend terminal second.
3. Open frontend URL in browser.
4. In dashboard:
   - choose webcam from dropdown (or enter IP camera URL),
   - choose target mode,
   - click **Start**.
