"""Orchestrator: combines concept matrix and data streams into a complete DriftDataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .concept_matrix import generate_concept_matrix
from .configs import GeneratorConfig
from .data_streams import ConceptSpec, generate_samples, make_concept_specs


@dataclass
class DriftDataset:
    """Complete drift dataset with ground truth concept assignments."""

    concept_matrix: np.ndarray  # (K, T) int array of concept IDs
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]  # (k, t) -> (X, y)
    config: GeneratorConfig
    concept_specs: list[ConceptSpec]

    def save(self, base_dir: str | Path | None = None) -> Path:
        """Save dataset to disk.

        Returns the output directory path.
        """
        if base_dir is None:
            base_dir = self.config.output_dir
        out_dir = Path(base_dir) / self.config.dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.to_json(out_dir / "config.json")

        # Save concept matrix
        np.save(out_dir / "concept_matrix.npy", self.concept_matrix)
        np.savetxt(
            out_dir / "concept_matrix.csv",
            self.concept_matrix,
            fmt="%d",
            delimiter=",",
        )

        # Save concept specs
        specs_data = [
            {
                "concept_id": s.concept_id,
                "generator_type": s.generator_type,
                "variant": s.variant,
                "noise_scale": s.noise_scale,
            }
            for s in self.concept_specs
        ]
        with open(out_dir / "concept_specs.json", "w") as f:
            json.dump(specs_data, f, indent=2)

        # Save data per (client, time_step)
        data_dir = out_dir / "data"
        data_dir.mkdir(exist_ok=True)
        for (k, t), (X, y) in self.data.items():
            np.savez_compressed(
                data_dir / f"client_{k:02d}_step_{t:02d}.npz",
                X=X,
                y=y,
                concept_id=self.concept_matrix[k, t],
            )

        return out_dir


def generate_drift_dataset(config: GeneratorConfig) -> DriftDataset:
    """Main entry point: generate a complete drift dataset with ground truth.

    Parameters
    ----------
    config : GeneratorConfig
        Full configuration.

    Returns
    -------
    DriftDataset
    """
    # Step 1: Generate concept matrix
    matrix = generate_concept_matrix(
        K=config.K,
        T=config.T,
        n_concepts=config.n_concepts,
        alpha=config.alpha,
        seed=config.seed,
    )

    # Step 2: Create concept specs
    n_concepts_actual = int(matrix.max()) + 1
    specs = make_concept_specs(n_concepts_actual, config.generator_type, config.delta)

    # Step 3: Generate data for each (client, time_step)
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(config.K):
        for t in range(config.T):
            concept_id = int(matrix[k, t])
            # Deterministic seed per (k, t) for reproducibility
            sample_seed = config.seed + k * config.T + t + 10000
            X, y = generate_samples(specs[concept_id], config.n_samples, sample_seed)
            data[(k, t)] = (X, y)

    return DriftDataset(
        concept_matrix=matrix,
        data=data,
        config=config,
        concept_specs=specs,
    )
