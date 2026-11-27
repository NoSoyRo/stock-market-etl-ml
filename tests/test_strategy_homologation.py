"""Pruebas de validacion de estrategias homologadas."""

from copy import deepcopy

import pytest

from src.agent.strategy_validator import StrategyValidator
from src.agent.strategy_comparator import StrategyComparator


@pytest.fixture
def datos():

    return {
        "strategy_name": "modelo_prueba",
        "strategy_type": "machine_learning",
        "universe": "bmv",

        "strategy": {
            "sharpe_net": 1.2,
            "total_return_net": 0.25,
            "max_drawdown": -0.15,
            "n_periods": 876,
            "auc": 0.65
        },

        "config": {
            "commission_pct": 0.0025,
            "slippage_pct": 0.0005
        },

        "vs_benchmark": {
            "excess_return_net": 0.10
        },

        "model": {
            "train_start": "2008-01-01",
            "train_end": "2023-04-27",
            "train_last_exit": "2023-05-08",
            "test_start": "2023-05-09",
            "test_end": "2026-10-30"
        },

        "evaluation_context": {
            "no_trades": False,
            "exploratory": False
        }
    }


def test_estrategia_tradicional(datos):

    datos["strategy_type"] = "rule_based"
    datos["strategy"]["auc"] = None
    datos.pop("model")

    resultado = StrategyValidator().act(datos)

    assert resultado["preguntas"][5]["estado"] == "no_aplica"
    assert resultado["preguntas"][7]["estado"] == "no_aplica"

    assert resultado["resumen"]["no_aplica"] == 2
    assert resultado["resumen"]["aplicables"] == 6
    assert resultado["resumen"]["cumple"] == 6
    assert resultado["aprobada"] is True


@pytest.mark.parametrize("auc", [0.4997, 0.5, 1.1])
def test_auc_insuficiente(datos, auc):

    datos["strategy"]["auc"] = auc

    resultado = StrategyValidator().act(datos)

    assert resultado["preguntas"][7]["estado"] == "no_cumple"
    assert resultado["aprobada"] is False


def test_auc_ausente(datos):

    datos["strategy"]["auc"] = None

    resultado = StrategyValidator().act(datos)

    assert resultado["preguntas"][7]["estado"] == "pendiente"
    assert resultado["aprobada"] is False


def test_sin_operaciones(datos):

    datos["evaluation_context"]["no_trades"] = True

    resultado = StrategyValidator().act(datos)

    assert resultado["sin_operaciones"] is True
    assert resultado["preguntas"][0]["estado"] == "no_cumple"
    assert resultado["aprobada"] is False


def test_exploratoria(datos):

    datos["evaluation_context"]["exploratory"] = True

    resultado = StrategyValidator().act(datos)

    assert resultado["exploratoria"] is True
    assert resultado["resumen"]["cumple"] == 8
    assert resultado["aprobada"] is False


def test_fuga_temporal(datos):

    datos["model"]["train_last_exit"] = "2023-05-10"

    resultado = StrategyValidator().act(datos)

    assert resultado["preguntas"][5]["estado"] == "no_cumple"
    assert resultado["aprobada"] is False


def test_fecha_invalida(datos):

    datos["model"]["test_start"] = "2023-99-99"

    resultado = StrategyValidator().act(datos)

    assert resultado["preguntas"][5]["estado"] == "no_cumple"


def test_comparacion_proporcional(datos):

    tradicional = deepcopy(datos)

    tradicional["strategy_type"] = "rule_based"
    tradicional["strategy"]["auc"] = None
    tradicional.pop("model")

    tradicional["vs_benchmark"]["excess_return_net"] = -0.01

    ml = deepcopy(datos)

    ml["vs_benchmark"]["excess_return_net"] = -0.01
    ml["strategy"]["auc"] = 0.4997

    resultado = StrategyComparator().evaluate({
        "tradicional": tradicional,
        "ml": ml
    })

    assert resultado["comparacion"][0]["estrategia"] == "tradicional"

    assert resultado["evaluaciones"]["tradicional"]["resumen"][
        "proporcion_cumplimiento"
    ] == pytest.approx(5 / 6)

    assert resultado["evaluaciones"]["ml"]["resumen"][
        "proporcion_cumplimiento"
    ] == pytest.approx(6 / 8)
