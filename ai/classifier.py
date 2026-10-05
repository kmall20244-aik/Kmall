#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai/classifier.py

Simple Random Forest classifier for lawful signal classes (offline training on synthetic/allowed data).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from loguru import logger

from ai.feature_extraction import FeatureExtractor


class SignalClassifier:
    def __init__(self, model_path: str = 'models/classifier.pkl', sample_rate: float = 200000):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.model: RandomForestClassifier | None = None
        self.fe = FeatureExtractor(sample_rate)

    def generate_synthetic(self, n_per_class: int = 200):
        X = []
        y = []
        for i in range(n_per_class):
            # DRONE-like (narrowband with freq hopping simulated via noise)
            t = np.arange(0, 0.05, 1 / self.sample_rate)
            sig = np.exp(1j * 2 * np.pi * 10000 * t) + 0.5 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
            X.append(list(self.fe.extract_all(sig).values()))
            y.append('DRONE')

            # WIFI-like (wideband)
            sig2 = np.random.randn(len(t))
            X.append(list(self.fe.extract_all(sig2).values()))
            y.append('WIFI')

            # NOISE
            sig3 = 0.5 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
            X.append(list(self.fe.extract_all(sig3).values()))
            y.append('NOISE')

        return np.array(X), np.array(y)

    def train(self, X: np.ndarray, y: np.ndarray):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        logger.info('Training results:\n{}', classification_report(y_test, preds))
        self.model = clf
        return clf

    def save_model(self):
        if self.model is None:
            raise RuntimeError('Model is not trained')
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        self.model = joblib.load(self.model_path)
        return self.model

    def predict(self, signal_in: np.ndarray) -> dict:
        feats = list(self.fe.extract_all(signal_in).values())
        if self.model is None:
            raise RuntimeError('Model not loaded')
        pred = self.model.predict([feats])[0]
        proba = self.model.predict_proba([feats])[0].max()
        return {'label': pred, 'confidence': float(proba)}


if __name__ == '__main__':
    sc = SignalClassifier()
    X, y = sc.generate_synthetic(n_per_class=50)
    clf = sc.train(X, y)
    sc.model = clf
    sc.save_model()
    print('Model trained and saved')
