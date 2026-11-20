from __future__ import annotations

import pytest

from src.ml.power import ic_se, required_weeks


def test_required_weeks_valores_del_handoff():
    assert required_weeks(0.04, 0.225) in (195, 196)
    assert required_weeks(0.04, 0.26) in (261, 262)


def test_two_sided_pide_mas_y_se():
    assert required_weeks(0.04, 0.225, two_sided=True) > required_weeks(0.04, 0.225)
    assert ic_se(0.22, 52) == pytest.approx(0.0305, abs=1e-3)
