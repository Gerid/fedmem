from __future__ import annotations

import pytest

from fedprotrack.experiments.cifar_full_sweep import (
    available_cifar_sweep_methods,
    build_parser,
    resolve_cifar_sweep_methods,
)
from fedprotrack.experiments.method_registry import (
    FOCUSED_CIFAR_SWEEP_METHODS,
    FULL_CIFAR_SWEEP_METHODS,
)


def test_available_methods_include_requested_baselines() -> None:
    methods = set(available_cifar_sweep_methods())
    assert "FedProTrack-linear-split" in methods
    assert "FedProto" in methods
    assert "pFedMe" in methods
    assert "APFL" in methods
    assert "IFCA-8" in methods
    assert "FedAvg" in methods


def test_method_selectors_expand_to_registry_groups() -> None:
    assert resolve_cifar_sweep_methods("focused") == list(FOCUSED_CIFAR_SWEEP_METHODS)
    assert resolve_cifar_sweep_methods("all") == list(FULL_CIFAR_SWEEP_METHODS)


def test_method_resolution_preserves_explicit_order() -> None:
    methods = resolve_cifar_sweep_methods("FedProto,FedAvg,FedProTrack-linear-split")
    assert methods == ["FedProto", "FedAvg", "FedProTrack-linear-split"]


def test_unknown_method_selector_raises() -> None:
    with pytest.raises(ValueError):
        resolve_cifar_sweep_methods("FedAvg,NotARealMethod")


def test_parser_quick_mode_keeps_full_sweep_default() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.methods == "all"
