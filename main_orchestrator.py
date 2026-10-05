#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_orchestrator.py

Orchestrator that wires together capture -> DSP -> feature extraction -> classification -> tracking -> reporting.
Designed for lawful, receive-only, offline defensive monitoring.

This module runs an async loop and emits events via websocket_manager.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

import numpy as np
from loguru import logger

from core.config import APP_CONFIG
from core.database import init_database, LocalDatabase
from core.security import APPROVAL_GATE
from sdr.simulator import SDRSimulator
from sdr.rtlsdr_driver import RTLSDRDriver
from dsp.fft_whitening import FFTWhitener
from ai.feature_extraction import FeatureExtractor
from ai.classifier import SignalClassifier
from dsp.kalman_3d import KalmanFilter3D
from api.websocket_manager import manager as ws_manager
from analytics.report_generator import ReportGenerator


class Orchestrator:
    def __init__(self):
        self.db = init_database()
        self.running = False
        self.capture_device = None
        self.simulator = SDRSimulator(fs=APP_CONFIG.default_sample_rate)
        self.whitener = FFTWhitener(fs=APP_CONFIG.default_sample_rate, fft_size=1024)
        self.fe = FeatureExtractor(sample_rate=APP_CONFIG.default_sample_rate)
        self.classifier = SignalClassifier(model_path="models/classifier.pkl", sample_rate=APP_CONFIG.default_sample_rate)
        # try load model if exists
        try:
            self.classifier.load_model()
            logger.info("Loaded classifier model")
        except Exception:
            logger.warning("Classifier model not found — running with no model")
            self.classifier.model = None
        self.tracker = KalmanFilter3D(dt=0.1)
        self.reporter = ReportGenerator(out_dir=APP_CONFIG.reports_dir)

    def select_device(self, device_type: str = "simulator", freq_hz: Optional[float] = None):
        if device_type == "simulator":
            self.capture_device = self.simulator
            logger.info("Selected SDR simulator")
        elif device_type == "rtl-sdr":
            freq = freq_hz or 100e6
            self.capture_device = RTLSDRDriver(freq_hz=freq, sample_rate_hz=APP_CONFIG.default_sample_rate)
            try:
                self.capture_device.open()
                logger.info("Opened RTL-SDR device at {} Hz", freq)
            except Exception as e:
                logger.error("Failed to open RTL-SDR: {}", e)
                self.capture_device = self.simulator
        else:
            logger.warning("Unknown device type: {} — falling back to simulator", device_type)
            self.capture_device = self.simulator

    async def start(self, device_type: str = "simulator", freq_hz: Optional[float] = None):
        self.select_device(device_type, freq_hz=freq_hz)
        self.running = True
        logger.info("Orchestrator started (device=%s)", device_type)
        loop = asyncio.get_event_loop()
        # start heartbeat background task via WS manager
        loop.create_task(ws_manager.heartbeat())
        try:
            while self.running:
                # capture a block of samples (simulate smaller block sizes)
                raw = None
                try:
                    if isinstance(self.capture_device, SDRSimulator):
                        raw = self.capture_device.generate_tone(freq_hz=10000, duration=0.05, amplitude=1.0)
                    else:
                        raw = self.capture_device.read_samples(4096)
                except Exception as e:
                    logger.error("Capture error: {}", e)
                    raw = self.simulator.generate_tone(freq_hz=10000, duration=0.05, amplitude=0.5)

                # ensure numpy array
                samples = np.asarray(raw, dtype=np.complex128)

                # DSP preprocessing: whiten
                try:
                    flat = self.whitener.spectral_flattening(samples.real)
                except Exception:
                    flat = samples.real

                # feature extraction
                feats = self.fe.extract_all(flat)

                # classification (if model available)
                label = "UNKNOWN"
                confidence = 0.0
                if self.classifier.model is not None:
                    try:
                        res = self.classifier.predict(samples)
                        label = res.get('label', 'UNKNOWN')
                        confidence = res.get('confidence', 0.0)
                    except Exception as e:
                        logger.warning("Classification error: {}", e)

                # simple detection threshold (example)
                if confidence > 0.6 or feats['entropy'] > 5.0:
                    # record to DB
                    self.db.add_signal(signal_type=label, frequency_hz=float(freq_hz or 0), strength_db=float(np.max(np.abs(samples))), source=str(device_type), confidence=float(confidence), metadata=feats)
                    event = {
                        'event_id': str(uuid.uuid4()),
                        'type': 'detection',
                        'label': label,
                        'confidence': confidence,
                        'features': feats,
                        'timestamp': time.time(),
                    }
                    # broadcast event
                    await ws_manager.broadcast({'type': 'detection', 'payload': event})
                    logger.info("Broadcasted detection: {} (conf={})", label, confidence)

                # simple tracking demo: feed centroid as measurement
                try:
                    centroid = feats.get('centroid', 0.0)
                    # use centroid as x, arbitrary y/z for demo
                    self.tracker.predict()
                    state = self.tracker.update(centroid, centroid * 0.1, 0.0)
                except Exception as e:
                    logger.debug("Tracking update failed: {}", e)

                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled")
        finally:
            self.running = False
            logger.info("Orchestrator stopped")

    async def stop(self):
        self.running = False


# Simple CLI-based run
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run orchestrator (receive-only)')
    parser.add_argument('--device', default='simulator', choices=['simulator', 'rtl-sdr'], help='capture device')
    parser.add_argument('--freq', type=float, default=100e6, help='center frequency (Hz)')
    args = parser.parse_args()

    orch = Orchestrator()
    try:
        asyncio.run(orch.start(device_type=args.device, freq_hz=args.freq))
    except KeyboardInterrupt:
        print('Interrupted')
