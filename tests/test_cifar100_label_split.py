from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.real_data.cifar100_recurrence import (
    CIFAR100RecurrenceConfig,
    _concept_class_subsets,
)


class TestConceptClassSubsets:
    """Unit tests for _concept_class_subsets label-split logic."""

    # ---- "none" mode ----

    def test_none_returns_all_classes_for_every_concept(self) -> None:
        subsets = _concept_class_subsets(3, "none")
        for c in range(3):
            np.testing.assert_array_equal(subsets[c], np.arange(20))

    def test_none_different_concept_counts(self) -> None:
        for n in (1, 4, 7):
            subsets = _concept_class_subsets(n, "none")
            assert len(subsets) == n
            for c in range(n):
                assert len(subsets[c]) == 20

    # ---- "disjoint" mode ----

    def test_disjoint_partitions_are_non_overlapping(self) -> None:
        subsets = _concept_class_subsets(4, "disjoint")
        seen: set[int] = set()
        for c in range(4):
            classes = set(int(x) for x in subsets[c])
            assert seen.isdisjoint(classes), f"Concept {c} overlaps with earlier concepts"
            seen.update(classes)

    def test_disjoint_covers_all_classes(self) -> None:
        subsets = _concept_class_subsets(4, "disjoint")
        all_classes = set()
        for arr in subsets.values():
            all_classes.update(int(x) for x in arr)
        assert all_classes == set(range(20))

    def test_disjoint_even_split(self) -> None:
        """With 4 concepts and 20 classes, each gets exactly 5."""
        subsets = _concept_class_subsets(4, "disjoint")
        for c in range(4):
            assert len(subsets[c]) == 5

    def test_disjoint_uneven_split_last_gets_remainder(self) -> None:
        """With 3 concepts and 20 classes: 6, 6, 8."""
        subsets = _concept_class_subsets(3, "disjoint")
        assert len(subsets[0]) == 6
        assert len(subsets[1]) == 6
        assert len(subsets[2]) == 8  # remainder goes to last

    def test_disjoint_single_concept_gets_all(self) -> None:
        subsets = _concept_class_subsets(1, "disjoint")
        np.testing.assert_array_equal(subsets[0], np.arange(20))

    def test_disjoint_20_concepts_one_class_each(self) -> None:
        subsets = _concept_class_subsets(20, "disjoint")
        for c in range(20):
            assert len(subsets[c]) == 1

    # ---- "overlap" mode ----

    def test_overlap_correct_number_of_classes_per_concept(self) -> None:
        subsets = _concept_class_subsets(4, "overlap", n_classes_per_concept=8)
        for c in range(4):
            assert len(subsets[c]) == 8

    def test_overlap_adjacent_concepts_share_classes(self) -> None:
        subsets = _concept_class_subsets(4, "overlap", n_classes_per_concept=10)
        # Adjacent concepts should share some classes
        overlap_01 = set(subsets[0].tolist()) & set(subsets[1].tolist())
        assert len(overlap_01) > 0, "Adjacent concepts should share classes"

    def test_overlap_wraps_around(self) -> None:
        """Overlap mode wraps around the 20 classes circularly."""
        subsets = _concept_class_subsets(4, "overlap", n_classes_per_concept=10)
        # All class indices should be in [0, 20)
        for c in range(4):
            assert all(0 <= x < 20 for x in subsets[c])

    def test_overlap_with_small_window(self) -> None:
        """Small windows can be fully disjoint if stride > window size."""
        subsets = _concept_class_subsets(4, "overlap", n_classes_per_concept=3)
        # stride = 20/4 = 5, window = 3 => disjoint
        overlap_01 = set(subsets[0].tolist()) & set(subsets[1].tolist())
        assert len(overlap_01) == 0

    def test_overlap_returns_sorted_arrays(self) -> None:
        subsets = _concept_class_subsets(5, "overlap", n_classes_per_concept=8)
        for c in range(5):
            arr = subsets[c]
            assert np.all(arr[:-1] <= arr[1:]), f"Concept {c} not sorted"


class TestConfigLabelSplitValidation:
    """Tests for CIFAR100RecurrenceConfig label_split parameter."""

    def test_default_is_none(self) -> None:
        cfg = CIFAR100RecurrenceConfig()
        assert cfg.label_split == "none"

    def test_accepts_none(self) -> None:
        cfg = CIFAR100RecurrenceConfig(label_split="none")
        assert cfg.label_split == "none"

    def test_accepts_disjoint(self) -> None:
        cfg = CIFAR100RecurrenceConfig(label_split="disjoint")
        assert cfg.label_split == "disjoint"

    def test_accepts_overlap(self) -> None:
        cfg = CIFAR100RecurrenceConfig(label_split="overlap", n_classes_per_concept=8)
        assert cfg.label_split == "overlap"

    # ---- Alias tests ----

    def test_shared_alias_normalizes_to_none(self) -> None:
        cfg = CIFAR100RecurrenceConfig(label_split="shared")
        assert cfg.label_split == "none"

    def test_overlapping_alias_normalizes_to_overlap(self) -> None:
        cfg = CIFAR100RecurrenceConfig(
            label_split="overlapping", n_classes_per_concept=8
        )
        assert cfg.label_split == "overlap"

    # ---- Rejection tests ----

    def test_rejects_invalid_label_split(self) -> None:
        with pytest.raises(ValueError, match="label_split"):
            CIFAR100RecurrenceConfig(label_split="random")

    def test_overlap_rejects_too_few_classes(self) -> None:
        with pytest.raises(ValueError, match="n_classes_per_concept"):
            CIFAR100RecurrenceConfig(label_split="overlap", n_classes_per_concept=1)

    def test_overlap_rejects_too_many_classes(self) -> None:
        with pytest.raises(ValueError, match="n_classes_per_concept"):
            CIFAR100RecurrenceConfig(label_split="overlap", n_classes_per_concept=21)

    def test_overlap_accepts_boundary_values(self) -> None:
        cfg2 = CIFAR100RecurrenceConfig(label_split="overlap", n_classes_per_concept=2)
        assert cfg2.n_classes_per_concept == 2
        cfg20 = CIFAR100RecurrenceConfig(label_split="overlap", n_classes_per_concept=20)
        assert cfg20.n_classes_per_concept == 20


class TestConceptClassSubsetsRecurrence:
    """Verify that recurring concepts get the same label distribution."""

    def test_same_concept_id_returns_same_classes(self) -> None:
        """Two calls with the same parameters must produce identical subsets."""
        subsets_a = _concept_class_subsets(4, "disjoint")
        subsets_b = _concept_class_subsets(4, "disjoint")
        for c in range(4):
            np.testing.assert_array_equal(subsets_a[c], subsets_b[c])

    def test_disjoint_concepts_have_different_classes(self) -> None:
        subsets = _concept_class_subsets(4, "disjoint")
        for i in range(4):
            for j in range(i + 1, 4):
                overlap = set(subsets[i].tolist()) & set(subsets[j].tolist())
                assert len(overlap) == 0, (
                    f"Concepts {i} and {j} share classes: {overlap}"
                )
