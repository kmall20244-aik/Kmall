#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api/backend.py

Local lawful monitoring backend.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

app = FastAPI(title="Defensive Monitoring API", version="0.1.0")


class SignalRequest(BaseModel):
    signal_type: str
    frequency_hz: float
    strength_db: float
    source: str = "local-sdr"
    confidence: float = 0.0
    metadata: dict[str, Any] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "mode": "receive-only", "requires_approval": True}


@app.get("/api/system")
def system_status():
    return {
        "name": "Defensive Monitoring System",
        "mode": "offline-local",
        "purpose": "Lawful signal monitoring and reporting",
        "approval_required": True,
    }


@app.post("/api/signal")
def receive_signal(payload: SignalRequest):
    if not payload.signal_type or payload.frequency_hz <= 0:
        raise HTTPException(status_code=400, detail="Invalid monitoring payload")
    return {
        "status": "accepted",
        "signal_type": payload.signal_type,
        "frequency_hz": payload.frequency_hz,
        "source": payload.source,
        "confidence": payload.confidence,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.backend:app", host="0.0.0.0", port=8000, reload=False)
