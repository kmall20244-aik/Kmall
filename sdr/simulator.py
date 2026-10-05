#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sdr/simulator.py

Safe signal simulator for offline testing of lawful monitoring pipelines.
"""
from __future__ import annotations

import numpy as np


class SDRSimulator:
    def __init__(self, fs: float = 2.4e6):
        self.fs = fs

    def generate_tone(self, freq_hz: float, duration: float, amplitude: float = 1.0) -> np.ndarray:
        t = np.arange(0, duration, 1 / self.fs)
        return amplitude * np.exp(1j * 2 * np.pi * freq_hz * t)

    def generate_qpsk(self, symbol_rate: float, duration: float) -> np.ndarray:
        t = np.arange(0, duration, 1 / self.fs)
        symbols = np.random.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], size=len(t))
        return symbols.astype(np.complex128)

    def add_awgn(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        signal_pow = np.mean(np.abs(signal) ** 2)
        noise_pow = signal_pow / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_pow / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
        return signal + noise

    def add_multipath(self, signal: np.ndarray, delays: list[float], gains: list[float]) -> np.ndarray:
        out = np.zeros_like(signal, dtype=np.complex128)
        for d, g in zip(delays, gains):
            delayed = np.roll(signal, int(d))
            out += g * delayed
        return signal + out


if __name__ == "__main__":
    sim = SDRSimulator(fs=1_500_000)
    tone = sim.generate_tone(100_000, 0.02, amplitude=0.8)
    noisy = sim.add_awgn(tone, snr_db=15)
    print("Tone samples:", noisy.shape)
