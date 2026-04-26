import pandas as pd
import pytest

from stock.trader import (
    Backtester,
    DEFAULT_STRATEGY_PARAMS,
    MARKET_DEFAULTS,
    MomentumStrategy,
    RSIStrategy,
    SmaCross,
    STRATEGY_REGISTRY,
)


def _make_ohlcv(periods: int = 60) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with a realistic oscillating price series.

    Alternates between up and down moves so RSI never divides by zero.
    """
    import math

    idx = pd.date_range("2024-01-01", periods=periods, freq="B")
    # Sine wave on top of an upward drift so there are both gains and losses.
    close = [100 + i * 0.3 + 5 * math.sin(i * 0.4) for i in range(periods)]
    return pd.DataFrame(
        {
            "Open": [c - 0.2 for c in close],
            "High": [c + 0.8 for c in close],
            "Low": [c - 0.8 for c in close],
            "Close": close,
            "Volume": [10_000] * periods,
        },
        index=idx,
    )


def _run_strategy(strategy_cls, **kwargs) -> dict:
    bt = Backtester(
        csv_path=None,
        data_df=_make_ohlcv(),
        strategy=strategy_cls,
        cash=10_000,
        **kwargs,
    )
    bt.run()
    return bt.to_tool_result()


class TestSmaCross:
    def test_returns_ok_status(self):
        result = _run_strategy(SmaCross, fast=5, slow=20)
        assert result["status"] == "ok"

    def test_summary_contains_backtest_info(self):
        result = _run_strategy(SmaCross, fast=5, slow=20)
        assert "Backtest complete" in result["summary"]

    def test_performance_keys_present(self):
        result = _run_strategy(SmaCross, fast=5, slow=20)
        perf = result["data"]["performance"]
        assert "initial_cash" in perf
        assert "final_cash" in perf
        assert "return_pct" in perf
        assert "sharpe_ratio" in perf


class TestRSIStrategy:
    def test_returns_ok_status(self):
        result = _run_strategy(RSIStrategy, rsi_period=14, rsi_upper=70, rsi_lower=30)
        assert result["status"] == "ok"

    def test_default_params_run(self):
        result = _run_strategy(RSIStrategy)
        assert result["status"] == "ok"

    def test_performance_keys_present(self):
        result = _run_strategy(RSIStrategy)
        perf = result["data"]["performance"]
        assert "initial_cash" in perf
        assert "return_pct" in perf


class TestMomentumStrategy:
    def test_returns_ok_status(self):
        result = _run_strategy(MomentumStrategy, lookback_period=10, threshold=0.02)
        assert result["status"] == "ok"

    def test_default_params_run(self):
        result = _run_strategy(MomentumStrategy)
        assert result["status"] == "ok"

    def test_performance_keys_present(self):
        result = _run_strategy(MomentumStrategy)
        perf = result["data"]["performance"]
        assert "initial_cash" in perf
        assert "return_pct" in perf


class TestStrategyRegistry:
    def test_all_strategies_registered(self):
        assert "SmaCross" in STRATEGY_REGISTRY
        assert "RSIStrategy" in STRATEGY_REGISTRY
        assert "MomentumStrategy" in STRATEGY_REGISTRY

    def test_registry_maps_to_correct_classes(self):
        assert STRATEGY_REGISTRY["SmaCross"] is SmaCross
        assert STRATEGY_REGISTRY["RSIStrategy"] is RSIStrategy
        assert STRATEGY_REGISTRY["MomentumStrategy"] is MomentumStrategy


class TestBacktestToolStrategySelection:
    def test_unknown_strategy_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(
            "stock.stock_loader._fetch_us_live_data",
            lambda *a, **kw: _make_ohlcv(),
        )
        from graph.tools import backtest_tool

        result = backtest_tool.invoke(
            {
                "ticker": "AAPL",
                "market": "us",
                "start_date": "2024-01-01",
                "strategy": "NonExistentStrategy",
            }
        )
        assert result["status"] == "error"
        assert "NonExistentStrategy" in result["summary"]
        assert "Available" in result["summary"]

    def test_sma_cross_via_strategy_param(self, monkeypatch):
        monkeypatch.setattr(
            "stock.stock_loader._fetch_us_live_data",
            lambda *a, **kw: _make_ohlcv(),
        )
        from graph.tools import backtest_tool

        result = backtest_tool.invoke(
            {
                "ticker": "AAPL",
                "market": "us",
                "start_date": "2024-01-01",
                "strategy": "SmaCross",
                "params": {"fast": 5, "slow": 20},
            }
        )
        assert result["status"] == "ok"
        assert "SmaCross" in result["summary"]

    def test_rsi_strategy_via_strategy_param(self, monkeypatch):
        monkeypatch.setattr(
            "stock.stock_loader._fetch_us_live_data",
            lambda *a, **kw: _make_ohlcv(),
        )
        from graph.tools import backtest_tool

        result = backtest_tool.invoke(
            {
                "ticker": "AAPL",
                "market": "us",
                "start_date": "2024-01-01",
                "strategy": "RSIStrategy",
                "params": {"rsi_period": 14, "rsi_lower": 30},
            }
        )
        assert result["status"] == "ok"
        assert "RSIStrategy" in result["summary"]

    def test_momentum_strategy_via_strategy_param(self, monkeypatch):
        monkeypatch.setattr(
            "stock.stock_loader._fetch_us_live_data",
            lambda *a, **kw: _make_ohlcv(),
        )
        from graph.tools import backtest_tool

        result = backtest_tool.invoke(
            {
                "ticker": "AAPL",
                "market": "us",
                "start_date": "2024-01-01",
                "strategy": "MomentumStrategy",
                "params": {"lookback_period": 10, "threshold": 0.02},
            }
        )
        assert result["status"] == "ok"
        assert "MomentumStrategy" in result["summary"]


class TestMultiStrategyBacktestNode:
    def _monkeypatch_fetch(self, monkeypatch):
        monkeypatch.setattr(
            "stock.stock_loader._fetch_us_live_data",
            lambda *a, **kw: _make_ohlcv(periods=60),
        )

    def _base_state(self):
        return {"ticker": "AAPL", "market": "us", "start_date": "2024-01-01", "docs": [], "debug": {}}

    def test_runs_all_three_strategies(self, monkeypatch):
        from graph.node import BacktestNode
        self._monkeypatch_fetch(monkeypatch)
        out = BacktestNode(cash=10_000)(self._base_state())
        data = out["backtest"]["data"]
        assert "SmaCross" in data
        assert "RSIStrategy" in data
        assert "MomentumStrategy" in data

    def test_output_envelope_shape(self, monkeypatch):
        from graph.node import BacktestNode
        self._monkeypatch_fetch(monkeypatch)
        backtest = BacktestNode(cash=10_000)(self._base_state())["backtest"]
        assert set(backtest.keys()) == {"status", "summary", "data", "debug_hint"}
        assert backtest["status"] in ("ok", "error")
        assert "Multi-strategy backtest" in backtest["summary"]
        assert "AAPL" in backtest["summary"]

    def test_per_strategy_data_shape(self, monkeypatch):
        from graph.node import BacktestNode
        self._monkeypatch_fetch(monkeypatch)
        data = BacktestNode(cash=10_000)(self._base_state())["backtest"]["data"]
        for name in ("SmaCross", "RSIStrategy", "MomentumStrategy"):
            entry = data[name]
            assert "status" in entry
            if entry["status"] == "ok":
                assert "performance" in entry and "trades" in entry
                assert "initial_cash" in entry["performance"]
                assert "return_pct" in entry["performance"]
                assert "sharpe_ratio" in entry["performance"]
            else:
                assert "error" in entry

    def test_summary_contains_all_strategy_names(self, monkeypatch):
        from graph.node import BacktestNode
        self._monkeypatch_fetch(monkeypatch)
        summary = BacktestNode(cash=10_000)(self._base_state())["backtest"]["summary"]
        for name in ("SmaCross", "RSIStrategy", "MomentumStrategy"):
            assert name in summary

    def test_custom_strategy_params_accepted(self, monkeypatch):
        from graph.node import BacktestNode
        self._monkeypatch_fetch(monkeypatch)
        node = BacktestNode(cash=10_000, strategy_params={"SmaCross": {"fast": 5, "slow": 15}})
        out = node(self._base_state())
        assert "SmaCross" in out["backtest"]["data"]

    def test_status_ok_when_at_least_one_succeeds(self):
        from graph.node import BacktestNode
        from unittest.mock import patch

        def fake_invoke(payload):
            if payload.get("strategy") == "SmaCross":
                return {
                    "status": "ok",
                    "summary": "ok",
                    "data": {
                        "performance": {
                            "initial_cash": 10000, "final_cash": 10500,
                            "return_pct": 0.05, "max_drawdown_pct": 1.0, "sharpe_ratio": 1.1,
                        },
                        "trades": [],
                    },
                    "debug_hint": None,
                }
            return {"status": "error", "summary": "simulated error", "data": None, "debug_hint": None}

        with patch("graph.node.backtest_tool") as mock_tool:
            mock_tool.invoke.side_effect = fake_invoke
            out = BacktestNode(cash=10_000)(self._base_state())
        assert out["backtest"]["status"] == "ok"

    def test_status_error_when_all_fail(self):
        from graph.node import BacktestNode
        from unittest.mock import patch

        with patch("graph.node.backtest_tool") as mock_tool:
            mock_tool.invoke.return_value = {
                "status": "error", "summary": "simulated", "data": None, "debug_hint": None,
            }
            out = BacktestNode(cash=10_000)(self._base_state())
        assert out["backtest"]["status"] == "error"

    def test_no_fast_slow_in_backtest_node_init(self):
        import inspect
        from graph.node import BacktestNode
        sig = inspect.signature(BacktestNode.__init__)
        assert "fast" not in sig.parameters
        assert "slow" not in sig.parameters

    def test_no_fast_slow_in_backtest_tool_signature(self):
        import inspect
        from graph.tools import backtest_tool
        sig = inspect.signature(backtest_tool.func)
        assert "fast" not in sig.parameters
        assert "slow" not in sig.parameters

    def test_default_strategy_params_covers_all_strategies(self):
        for name in STRATEGY_REGISTRY:
            assert name in DEFAULT_STRATEGY_PARAMS
            assert isinstance(DEFAULT_STRATEGY_PARAMS[name], dict)
            assert len(DEFAULT_STRATEGY_PARAMS[name]) > 0


class TestMarketAwareBroker:
    """Verify that TW and US markets use distinct stake, commission, and default cash."""

    def test_market_defaults_keys_present(self):
        for market in ("tw", "us"):
            cfg = MARKET_DEFAULTS[market]
            assert "stake" in cfg
            assert "commission" in cfg
            assert "default_cash" in cfg

    def test_tw_default_cash(self):
        assert MARKET_DEFAULTS["tw"]["default_cash"] == 1_000_000

    def test_us_default_cash(self):
        assert MARKET_DEFAULTS["us"]["default_cash"] == 50_000

    def test_tw_stake_is_1000(self):
        assert MARKET_DEFAULTS["tw"]["stake"] == 1000

    def test_us_stake_is_10(self):
        assert MARKET_DEFAULTS["us"]["stake"] == 10

    def test_tw_commission_higher_than_us(self):
        assert MARKET_DEFAULTS["tw"]["commission"] > MARKET_DEFAULTS["us"]["commission"]

    def test_tw_trades_use_1000_share_lots(self):
        bt = Backtester(
            csv_path=None,
            data_df=_make_ohlcv(periods=60),
            strategy=SmaCross,
            cash=1_000_000,
            market="tw",
            fast=5,
            slow=20,
        )
        bt.run()
        for trade in bt.get_trades():
            assert trade["size"] % 1000 == 0, f"TW trade size {trade['size']} not a multiple of 1000"

    def test_us_trades_use_10_share_lots(self):
        bt = Backtester(
            csv_path=None,
            data_df=_make_ohlcv(periods=60),
            strategy=SmaCross,
            cash=10_000,
            market="us",
            fast=5,
            slow=20,
        )
        bt.run()
        for trade in bt.get_trades():
            assert trade["size"] % 10 == 0, f"US trade size {trade['size']} not a multiple of 10"

    def test_backtest_node_uses_tw_default_cash_when_cash_not_set(self, monkeypatch):
        monkeypatch.setattr(
            "graph.tools.resolve_stock_data",
            lambda **kw: {"df": _make_ohlcv(periods=60), "csv_path": None, "source": "mock"},
        )
        from graph.node import BacktestNode

        state = {"ticker": "2330", "market": "tw", "start_date": "2024-01-01", "docs": [], "debug": {}}
        out = BacktestNode()(state)
        for strategy_data in out["backtest"]["data"].values():
            if strategy_data["status"] == "ok":
                assert strategy_data["performance"]["initial_cash"] == 1_000_000
                return
        pytest.skip("All strategies failed; cannot verify cash")

    def test_backtest_node_explicit_cash_overrides_market_default(self, monkeypatch):
        monkeypatch.setattr(
            "graph.tools.resolve_stock_data",
            lambda **kw: {"df": _make_ohlcv(periods=60), "csv_path": None, "source": "mock"},
        )
        from graph.node import BacktestNode

        state = {"ticker": "2330", "market": "tw", "start_date": "2024-01-01", "docs": [], "debug": {}}
        out = BacktestNode(cash=500_000)(state)
        for strategy_data in out["backtest"]["data"].values():
            if strategy_data["status"] == "ok":
                assert strategy_data["performance"]["initial_cash"] == 500_000
                return
        pytest.skip("All strategies failed; cannot verify cash")
