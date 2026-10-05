#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dsp/noise_whitening.py

Adaptive noise whitening utilities for lawful monitoring (offline).
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from loguru import logger


class NoiseWhitener:
    def __init__(self, fs: float):
        self.fs = float(fs)

    def estimate_noise_floor(self, signal_in: np.ndarray, window: int = 1024) -> np.ndarray:
        # Estimate noise floor by median smoothing of PSD
        f, Pxx = signal.welch(signal_in, fs=self.fs, nperseg=window)
        nf = np.median(Pxx) * np.ones_like(Pxx)
        logger.debug("Estimated noise floor with median {}", float(np.median(Pxx)))
        return f, nf

    def spectral_subtraction(self, sig: np.ndarray, noise_psd: np.ndarray, nfft: int = 1024) -> np.ndarray:
        # Frame-based spectral subtraction
        frames = self._frame_signal(sig, nfft)
        win = np.hanning(nfft)
        out = np.zeros_like(sig, dtype=np.complex128)
        hop = nfft // 2
        idx = 0
        for frm in frames:
            S = np.fft.rfft(win * frm, n=nfft)
            S_mag = np.abs(S)
            S_phase = np.angle(S)
            S_mag_clean = np.maximum(S_mag - np.sqrt(noise_psd), 1e-8)
            S_clean = S_mag_clean * np.exp(1j * S_phase)
            y = np.fft.irfft(S_clean, n=nfft)
            out[idx:idx + nfft] += y
            idx += hop
        return out

    def wiener_filter(self, sig: np.ndarray, noise_psd: np.ndarray, nfft: int = 1024) -> np.ndarray:
        frames = self._frame_signal(sig, nfft)
        win = np.hanning(nfft)
        out = np.zeros_like(sig, dtype=np.complex128)
        hop = nfft // 2
        idx = 0
        for frm in frames:
            S = np.fft.rfft(win * frm, n=nfft)
            Pxx = np.abs(S) ** 2
            H = Pxx / (Pxx + noise_psd + 1e-12)
            S_est = H * S
            y = np.fft.irfft(S_est, n=nfft)
            out[idx:idx + nfft] += y
            idx += hop
        return out

    def adaptive_noise_whitening(self, sig: np.ndarray, nfft: int = 1024, n_avg: int = 8) -> np.ndarray:
        # Simple spectral whitening by dividing by smoothed spectrum
        N = len(sig)
        win = np.hanning(nfft)
        hop = nfft // 2
        out = np.zeros(N, dtype=np.complex128)
        idx = 0
        # compute average spectrum
        spectra = []
        frames = self._frame_signal(sig, nfft)
        for frm in frames:
            S = np.fft.rfft(win * frm, n=nfft)
            spectra.append(np.abs(S))
        avg_spec = np.mean(spectra, axis=0)
        avg_spec = np.maximum(avg_spec, 1e-8)
        # whitening
        for frm in frames:
            S = np.fft.rfft(win * frm, n=nfft)
            S_white = S / avg_spec
            y = np.fft.irfft(S_white, n=nfft)
            out[idx:idx + nfft] += y
            idx += hop
        return out

    def _frame_signal(self, sig: np.ndarray, nfft: int = 1024):
        hop = nfft // 2
        n_frames = max(1, (len(sig) - nfft) // hop + 1)
        frames = [sig[i * hop:i * hop + nfft] if i * hop + nfft <= len(sig) else np.pad(sig[i * hop:], (0, i * hop + nfft - len(sig)))
                  for i in range(n_frames)]
        return frames


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fs = 48000
    t = np.arange(0, 1.0, 1 / fs)
    tone = np.exp(1j * 2 * np.pi * 1000 * t)
    noise = 0.5 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    sig = tone + noise
    nw = NoiseWhitener(fs)
    f, nf = nw.estimate_noise_floor(sig.real, window=2048)
    proc = nw.adaptive_noise_whitening(sig, nfft=1024)
    plt.figure()
    plt.plot(t[0:1000], sig.real[0:1000], label='raw')
    plt.plot(t[0:1000], proc.real[0:1000], label='whitened')
    plt.legend()
    plt.show()
