import numpy as np
import pandas as pd
import pytest

from src.agent.traditional_backtest import backtest_asset


def create_market(prices):

    prices = np.asarray(prices, dtype=float)

    dates = pd.bdate_range(
        "2020-01-01",
        periods=len(prices)
    )

    return pd.DataFrame({
        "Open": prices,
        "High": prices * 1.01,
        "Low": prices * 0.99,
        "Close": prices,
        "Volume": 10000
    }, index=dates)


def run_backtest(market, strategy="sma", **kwargs):

    return backtest_asset(
        market=market,
        calendar=market.index,
        start=market.index[220],
        end=market.index[-1],
        strategy=strategy,
        **kwargs
    )


def test_sma_uptrend():

    market = create_market(
        np.linspace(100, 200, 300)
    )

    result = run_backtest(market)

    assert result["strategy"]["total_return_gross"] > 0
    assert result["open_position"] is not None
    assert result["evaluation_period"]["active_days"] == 80


def test_sma_downtrend():

    market = create_market(
        np.linspace(200, 100, 300)
    )

    result = run_backtest(market)

    assert result["strategy"]["final_equity_net"] == 100000
    assert result["evaluation_period"]["active_days"] == 0
    assert result["open_position"] is None


def test_transaction_costs():

    market = create_market(
        np.linspace(100, 200, 300)
    )

    result = run_backtest(market)

    gross = result["strategy"]["final_equity_gross"]
    net = result["strategy"]["final_equity_net"]

    assert net < gross
    assert gross - net == pytest.approx(600)


def test_zero_costs():

    market = create_market(
        np.linspace(100, 200, 300)
    )

    result = run_backtest(
        market,
        commission_pct=0,
        slippage_pct=0
    )

    assert np.allclose(
        result["curves"]["gross"],
        result["curves"]["net"]
    )


def test_daily_curves():

    market = create_market(
        np.linspace(100, 200, 300)
    )

    result = run_backtest(market)

    assert len(result["dates"]) == 80
    assert len(result["curves"]["net"]) == 81
    assert len(result["daily_returns"]["net"]) == 80

    compounded = np.prod(
        1 + np.array(result["daily_returns"]["net"])
    )

    assert compounded * 100000 == pytest.approx(
        result["strategy"]["final_equity_net"]
    )


def test_blocked_entry():

    market = create_market(
        np.linspace(100, 200, 300)
    )

    market.loc[market.index[220], "Volume"] = 0

    result = run_backtest(market)

    assert result["evaluation_period"]["blocked_entries"] == 1

    assert result["open_position"]["entry_date"] == str(
        market.index[221].date()
    )


def test_stale_valuation():

    market = create_market(
        np.linspace(100, 200, 300)
    )

    market.loc[market.index[230], "Volume"] = 0

    result = run_backtest(market)

    assert result["evaluation_period"]["stale_valuation_days"] == 1

    assert result["curves"]["net"][11] == pytest.approx(
        result["curves"]["net"][10]
    )


def test_rsi_buy_and_sell():

    prices = np.concatenate([
        np.full(220, 100.0),
        np.linspace(100, 60, 30),
        np.linspace(60, 150, 50)
    ])

    market = create_market(prices)

    result = run_backtest(
        market,
        strategy="rsi"
    )

    assert len(result["operations"]) >= 1

    trade = result["operations"][0]

    assert trade["entry_date"] < trade["exit_date"]
    assert trade["status"] == "closed"

    assert trade["net_return"] == pytest.approx(
        trade["gross_return"] - 0.006
    )


def test_auc_not_applicable():

    market = create_market(
        np.linspace(100, 200, 300)
    )

    result = run_backtest(market)

    assert result["strategy"]["auc"] is None


def test_no_lookahead():

    market = create_market(
        np.linspace(100, 200, 300)
    )

    original = run_backtest(market)

    modified = market.copy()

    modified.loc[
        modified.index[260]:,
        ["Open", "High", "Low", "Close"]
    ] *= 0.5

    changed = run_backtest(modified)

    assert np.allclose(
        original["curves"]["net"][:41],
        changed["curves"]["net"][:41]
    )
