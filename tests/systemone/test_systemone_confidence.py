"""The official Jev confidence formulas, on hand-computed distributions."""

from __future__ import annotations

import pytest

from gmlx.systemone.confidence import choice_confidence, score_confidence


def test_choice_confidence_scales_from_uniform_to_certain():
    assert choice_confidence([0.25] * 4) == pytest.approx(0.0)
    assert choice_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert choice_confidence([0.25, 0.75]) == pytest.approx(0.5)
    assert choice_confidence([0.7]) == 1.0


def test_score_confidence_measures_spread_around_the_mode():
    assert score_confidence([0.0, 1.0, 0.0]) == pytest.approx(1.0)
    # 1 - (0.1 * 2 + 0.2 * 1) / (2/3)
    assert score_confidence([0.1, 0.2, 0.7]) == pytest.approx(0.4)
    assert score_confidence([0.5, 0.0, 0.5]) == 0.0
    assert score_confidence([0.3]) == 1.0


def test_inputs_are_normalized_first():
    assert choice_confidence([1.0, 3.0]) == pytest.approx(0.5)
    assert choice_confidence([0.0, 0.0]) == pytest.approx(0.0)
    assert score_confidence([2.0, 4.0, 14.0]) == pytest.approx(0.4)
