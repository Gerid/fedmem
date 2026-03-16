"""Base interface for online drift detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DriftResult:
    """Result from a single drift detection step."""

    is_drift: bool
    is_warning: bool = False
    detector_name: str = ""


class BaseDriftDetector(ABC):
    """Unified interface for online drift detectors.

    All detectors work in a streaming fashion: they receive one error indicator
    at a time (0 = correct, 1 = error) and maintain internal statistics to
    signal when a distribution change has occurred.
    """

    @abstractmethod
    def update(self, value: float) -> DriftResult:
        """Feed one observation and return drift status.

        Parameters
        ----------
        value : float
            Typically a binary error indicator (0 or 1), but some detectors
            accept continuous values.

        Returns
        -------
        DriftResult
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (called after drift is handled)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name."""

    def clone(self) -> BaseDriftDetector:
        """Create a fresh detector with the same hyperparameters."""
        return self.__class__(**self._init_kwargs())

    @abstractmethod
    def _init_kwargs(self) -> dict:
        """Return constructor kwargs for cloning."""
