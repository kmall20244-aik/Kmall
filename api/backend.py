#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api/backend.py

Extended FastAPI backend with WebSocket and report management endpoints.
Designed for lawful, receive-only, offline defensive monitoring (PoC).
"""
from __future__ import annotations

import asyncio
import sqlite3
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any

from core.database import init_database
from core.security import require_approval
from analytics.report_generator import ReportGenerator
from api.websocket_manager import manager
from core.config import APP_CONFIG
from core.logger import logger

app = FastAPI(title="Defensive Monitoring API", version="0.2.0")
DB = init_database()
REPORTER = ReportGenerator(out_dir=APP_CONFIG.reports_dir)


class SignalRequest(BaseModel):
    signal_type: str
    frequency_hz: float
    strength_db: float
    source: str = "local-sdr"
    confidence: float = 0.0
    metadata: dict[str, Any] | None = None


class ReportCreateRequest(BaseModel):
    title: str
    content: str
    created_by: str


class ReportApproveRequest(BaseModel):
    report_id: int
    approver: str
    reason: str


@app.get("/health")
def health():
    return {"status": "ok", "mode": "receive-only", "requires_approval": APP_CONFIG.require_approval}


@app.get("/api/system")
def system_status():
    return {
        "name": APP_CONFIG.app_name,
        "mode": "offline-local",
        "purpose": "Lawful signal monitoring and reporting",
        "approval_required": APP_CONFIG.require_approval,
    }


@app.post("/api/signal")
def receive_signal(payload: SignalRequest):
    if not payload.signal_type or payload.frequency_hz <= 0:
        raise HTTPException(status_code=400, detail="Invalid monitoring payload")
    DB.add_signal(
        signal_type=payload.signal_type,
        frequency_hz=payload.frequency_hz,
        strength_db=payload.strength_db,
        source=payload.source,
        confidence=payload.confidence,
        metadata=payload.metadata,
    )
    return {"status": "accepted", "signal_type": payload.signal_type}


@app.post("/api/reports/create")
def create_report(req: ReportCreateRequest):
    DB.add_report(title=req.title, status="draft", content=req.content, approved_by=None, approval_reason=None)
    DB.add_audit_event(event=f"report_created:{req.title}", actor=req.created_by, details=req.title)
    return {"status": "created", "title": req.title}


@app.get("/api/reports/list")
def list_reports():
    conn = sqlite3.connect(str(APP_CONFIG.db_path))
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, title, status, approved_by FROM reports ORDER BY id DESC LIMIT 100")
    rows = cur.fetchall()
    conn.close()
    return {"reports": [dict(id=r[0], timestamp=r[1], title=r[2], status=r[3], approved_by=r[4]) for r in rows]}


@app.post("/api/reports/approve")
def approve_report(req: ReportApproveRequest):
    conn = sqlite3.connect(str(APP_CONFIG.db_path))
    cur = conn.cursor()
    cur.execute("SELECT id, title, content, status FROM reports WHERE id=?", (req.report_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Report not found")

    report_id, title, content, status = row
    approval_ok = require_approval(action=f"report:{report_id}", actor=req.approver, reason=req.reason)
    if not approval_ok:
        conn.close()
        raise HTTPException(status_code=403, detail="Approval failed")

    pdf_path = REPORTER.generate_report(report_id=str(report_id), title=title, body=content, approved_by=req.approver)
    cur.execute("UPDATE reports SET status=?, approved_by=?, approval_reason=? WHERE id=?", ("approved", req.approver, req.reason, report_id))
    conn.commit()
    conn.close()
    DB.add_audit_event(event=f"report_approved:{report_id}", actor=req.approver, details=str(pdf_path))
    asyncio.create_task(manager.broadcast({'type': 'report_created', 'payload': {'report_id': report_id, 'title': title, 'path': str(pdf_path)}}))
    return {"status": "approved", "report_id": report_id, "pdf": str(pdf_path)}


@app.get("/api/events/list")
def events_list(limit: int = 100):
    conn = sqlite3.connect(str(APP_CONFIG.db_path))
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, signal_type, frequency_hz, strength_db, source, confidence FROM signals ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return {"events": [dict(id=r[0], timestamp=r[1], signal_type=r[2], frequency_hz=r[3], strength_db=r[4], source=r[5], confidence=r[6]) for r in rows]}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            await manager.send_personal(ws, {'type': 'echo', 'payload': data})
    except WebSocketDisconnect:
        await manager.disconnect(ws)


@app.get("/")
def index():
    return HTMLResponse(content="<html><body><h3>Defensive Monitoring API</h3><p>Use the dashboard at /templates/index.html</p></body></html>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.backend:app", host="0.0.0.0", port=8000, reload=False)
