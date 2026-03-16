"""Tests for drift detection module."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.drift_detector import (
    ADWINDetector,
    PageHinkleyDetector,
    KSWINDetector,
    NoDriftDetector,
    DriftResult,
)


class TestADWINDetector:
    def test_no_drift_on_stable_stream(self):
        det = ADWINDetector(delta=0.002)
        for _ in range(100):
            result = det.update(0.0)
        assert not result.is_drift

    def test_drift_on_distribution_change(self):
        det = ADWINDetector(delta=0.01)
        # Stable period
        for _ in range(200):
            det.update(0.0)
        # Sudden change
        detected = False
        for _ in range(200):
            result = det.update(1.0)
            if result.is_drift:
                detected = True
                break
        assert detected

    def test_name(self):
        assert ADWINDetector().name == "ADWIN"

    def test_reset(self):
        det = ADWINDetector()
        det.update(1.0)
        det.reset()
        result = det.update(0.0)
        assert not result.is_drift

    def test_clone(self):
        det = ADWINDetector(delta=0.01)
        clone = det.clone()
        assert clone.name == "ADWIN"
        assert clone._delta == 0.01


class TestPageHinkleyDetector:
    def test_no_drift_stable(self):
        det = PageHinkleyDetector()
        for _ in range(100):
            result = det.update(0.0)
        assert not result.is_drift

    def test_name(self):
        assert PageHinkleyDetector().name == "PageHinkley"


class TestKSWINDetector:
    def test_no_drift_stable(self):
        det = KSWINDetector()
        rng = np.random.default_rng(42)
        for _ in range(200):
            result = det.update(rng.random())
        assert isinstance(result, DriftResult)

    def test_name(self):
        assert KSWINDetector().name == "KSWIN"


class TestNoDriftDetector:
    def test_never_drifts(self):
        det = NoDriftDetector()
        for v in [0.0, 1.0, 0.5, 1.0]:
            result = det.update(v)
            assert not result.is_drift

    def test_name(self):
        assert NoDriftDetector().name == "NoDrift"
