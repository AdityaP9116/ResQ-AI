"""FastAPI server for ResQ-AI mission data.

Serves mission reports, fire data, civilian data, and live WebSocket
updates for the frontend dashboard.

Run standalone:
    python -m sim_bridge.api_server

Isaac Sim 5.1 compatibility: NO f-strings (this file runs outside Sim,
but we keep consistency).
"""

import asyncio
import glob
import json
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "frontend")
HOST = "0.0.0.0"
PORT = 8080

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ResQ-AI Mission API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files if directory exists
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_latest_report():
    """Read the latest.json report file."""
    path = os.path.join(REPORTS_DIR, "latest.json")
    if os.path.isfile(path):
        try:
            with open(path, "r") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"status": "no data", "message": "No mission reports yet"}


def _read_all_reports():
    """Read all mission report files, sorted by timestamp."""
    if not os.path.isdir(REPORTS_DIR):
        return []

    files = glob.glob(os.path.join(REPORTS_DIR, "mission_*.json"))
    files.sort()
    reports = []
    fi = 0
    while fi < len(files):
        try:
            with open(files[fi], "r") as fh:
                reports.append(json.load(fh))
        except Exception:
            pass
        fi = fi + 1
    return reports


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the frontend dashboard."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse({
        "service": "ResQ-AI Mission API",
        "endpoints": ["/api/status", "/api/history", "/api/fires",
                      "/api/civilians", "/ws/live"],
    })


@app.get("/api/status")
async def get_status():
    """Return the latest mission report."""
    return JSONResponse(_read_latest_report())


@app.get("/api/history")
async def get_history():
    """Return all mission reports."""
    return JSONResponse(_read_all_reports())


@app.get("/api/fires")
async def get_fires():
    """Return current fire data from latest report."""
    report = _read_latest_report()
    return JSONResponse(report.get("fires", {"active_count": 0, "zones": []}))


@app.get("/api/civilians")
async def get_civilians():
    """Return current civilian data from latest report."""
    report = _read_latest_report()
    return JSONResponse(report.get("civilians", {"total": 0}))


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

class ConnectionManager(object):
    """Manages WebSocket connections for live streaming."""

    def __init__(self):
        self.connections = []

    async def connect(self, ws):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, data):
        dead = []
        ci = 0
        while ci < len(self.connections):
            ws = self.connections[ci]
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
            ci = ci + 1
        di = 0
        while di < len(dead):
            self.disconnect(dead[di])
            di = di + 1


ws_manager = ConnectionManager()


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """Stream mission reports in real-time."""
    await ws_manager.connect(ws)
    try:
        # Send initial data
        await ws.send_json(_read_latest_report())

        # Poll for updates
        last_ts = 0.0
        while True:
            await asyncio.sleep(2.0)
            report = _read_latest_report()
            ts = report.get("timestamp", 0.0)
            if ts > last_ts:
                last_ts = ts
                await ws.send_json(report)
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:
        ws_manager.disconnect(ws)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    if not os.path.isdir(REPORTS_DIR):
        os.makedirs(REPORTS_DIR, exist_ok=True)

    print("[API Server] Reports dir: " + REPORTS_DIR)
    print("[API Server] Frontend dir: " + FRONTEND_DIR)
    print("[API Server] Starting on " + HOST + ":" + str(PORT))

    uvicorn.run(app, host=HOST, port=PORT)
