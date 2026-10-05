#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

Entry point for the local lawful monitoring system.
"""
from __future__ import annotations

import uvicorn

from api.backend import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
