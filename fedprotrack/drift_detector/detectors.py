"""Concrete drift detector implementations wrapping River's drift module."""

from __future__ import annotations

from river import drift

from .base import BaseDriftDetector, DriftResult


class ADWINDetector(BaseDriftDetector):
    """Adaptive Windowing (ADWIN) drift detector.

    Parameters
    ----------
    delta : float
        Confidence parameter for ADWIN. Smaller values make detection
        more conservative. Default 0.002.
    """

    def __init__(self, delta: float = 0.002):
        self._delta = delta
        self._detector = drift.ADWIN(delta=delta)

    def update(self, value: float) -> DriftResult:
        self._detector.update(value)
        return DriftResult(
            is_drift=self._detector.drift_detected,
            detector_name=self.name,
        )

    def reset(self) -> None:
        self._detector = drift.ADWIN(delta=self._delta)

    @property
    def name(self) -> str:
        return "ADWIN"

    def _init_kwargs(self) -> dict:
        return {"delta": self._delta}


class PageHinkleyDetector(BaseDriftDetector):
    """Page-Hinkley drift detector.

    Parameters
    ----------
    min_instances : int
        Minimum number of observations before detection is active.
    delta : float
        Magnitude threshold for change detection.
    threshold : float
        Cumulative sum threshold for signaling drift.
    """

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
    ):
        self._min_instances = min_instances
        self._delta = delta
        self._threshold = threshold
        self._detector = drift.PageHinkley(
            min_instances=min_instances,
            delta=delta,
            threshold=threshold,
        )

    def update(self, value: float) -> DriftResult:
        self._detector.update(value)
        return DriftResult(
            is_drift=self._detector.drift_detected,
            detector_name=self.name,
        )

    def reset(self) -> None:
        self._detector = drift.PageHinkley(
            min_instances=self._min_instances,
            delta=self._delta,
            threshold=self._threshold,
        )

    @property
    def name(self) -> str:
        return "PageHinkley"

    def _init_kwargs(self) -> dict:
        return {
            "min_instances": self._min_instances,
            "delta": self._delta,
            "threshold": self._threshold,
        }


class KSWINDetector(BaseDriftDetector):
    """Kolmogorov-Smirnov Windowing (KSWIN) drift detector.

    Parameters
    ----------
    alpha : float
        KS test significance level. Default 0.005.
    window_size : int
        Size of the sliding window. Default 100.
    stat_size : int
        Size of the statistic window. Default 30.
    """

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
    ):
        self._alpha = alpha
        self._window_size = window_size
        self._stat_size = stat_size
        self._detector = drift.KSWIN(
            alpha=alpha,
            window_size=window_size,
            stat_size=stat_size,
        )

    def update(self, value: float) -> DriftResult:
        self._detector.update(value)
        return DriftResult(
            is_drift=self._detector.drift_detected,
            detector_name=self.name,
        )

    def reset(self) -> None:
        self._detector = drift.KSWIN(
            alpha=self._alpha,
            window_size=self._window_size,
            stat_size=self._stat_size,
        )

    @property
    def name(self) -> str:
        return "KSWIN"

    def _init_kwargs(self) -> dict:
        return {
            "alpha": self._alpha,
            "window_size": self._window_size,
            "stat_size": self._stat_size,
        }


class NoDriftDetector(BaseDriftDetector):
    """Dummy detector that never signals drift. Used as a baseline."""

    def update(self, value: float) -> DriftResult:
        return DriftResult(is_drift=False, detector_name=self.name)

    def reset(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "NoDrift"

    def _init_kwargs(self) -> dict:
        return {}
