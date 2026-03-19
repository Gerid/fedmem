from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class GeneratorConfig:
    """Configuration for the distributed drift generator."""

    # Grid dimensions
    K: int = 10              # number of clients
    T: int = 10              # number of time steps
    n_samples: int = 500     # samples per (client, time_step)

    # Controllable axes
    rho: float = 5.0         # recurrence frequency {2, 3, 5, 10, inf}
    alpha: float = 0.5       # asynchrony level [0, 1]
    delta: float = 0.5       # separability [0.1, 1.0]

    # Data generation
    generator_type: str = "sine"  # "sine", "sea", "circle", real-data variants
    seed: int = 42

    # Output
    output_dir: str = "outputs"

    def __post_init__(self):
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.T < 2:
            raise ValueError(f"T must be >= 2, got {self.T}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if not (0.0 < self.delta <= 1.0):
            raise ValueError(f"delta must be in (0, 1], got {self.delta}")
        if self.generator_type not in (
            "sine",
            "sea",
            "circle",
            "rotating_mnist",
            "cifar100_recurrence",
            "fmow",
            "cifar100_overlap",
            "cifar100_label_overlap",
        ):
            raise ValueError(f"Unknown generator_type: {self.generator_type}")

    @property
    def n_concepts(self) -> int:
        """Number of distinct concepts in the shared pool."""
        if math.isinf(self.rho):
            return self.T
        return max(2, round(self.T / self.rho))

    @property
    def dir_name(self) -> str:
        rho_str = "inf" if math.isinf(self.rho) else f"{self.rho:.0f}"
        return f"{self.generator_type}_rho{rho_str}_alpha{self.alpha:.2f}_delta{self.delta:.2f}_seed{self.seed}"

    def to_json(self, path: str | Path) -> None:
        d = asdict(self)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> GeneratorConfig:
        with open(path) as f:
            d = json.load(f)
        return cls(**d)
