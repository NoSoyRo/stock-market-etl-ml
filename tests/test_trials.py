from __future__ import annotations

from src.ml.paper_trading import freeze_rule, log_counterfactual
from src.ml.trials import count_trials, log_trial


def test_log_trial_idempotente_y_conteo(tmp_path):
    p = str(tmp_path / "t.jsonl")
    assert log_trial("a_x", {"k": 5, "b": 1}, {"sharpe": 1.0}, 100, path=p)
    assert not log_trial("a_x", {"b": 1, "k": 5}, {"sharpe": 1.0}, 100, path=p)
    assert log_trial("a_x", {"k": 10}, {"sharpe": 0.5}, 100, path=p)
    assert log_trial("b_y", {"k": 5}, {}, 50, path=p)
    assert count_trials(path=p) == 3
    assert count_trials("a_", path=p) == 2
    assert count_trials(path=str(tmp_path / "no.jsonl")) == 0


def test_freeze_rule_y_contrafactual(tmp_path):
    assert freeze_rule({"a": 1, "b": 2}) == freeze_rule({"b": 2, "a": 1})
    assert freeze_rule({"a": 1}) != freeze_rule({"a": 2})
    u = [f"T{i}" for i in range(20)]
    assert log_counterfactual("2026-01-02", u, u[:5], "h", out_dir=tmp_path) is None
    p1 = log_counterfactual("2026-01-02", u, u[:5], "h", seed=3, enabled=True, out_dir=tmp_path)
    r1 = p1.read_text()
    p1.unlink()
    log_counterfactual("2026-01-02", u, u[:5], "h", seed=3, enabled=True, out_dir=tmp_path)
    assert p1.read_text() == r1
