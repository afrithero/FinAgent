import pandas as pd

from stock.stock_loader import StockDataCache, resolve_stock_data


def _sample_df():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "Open": range(10, 20),
            "High": range(11, 21),
            "Low": range(9, 19),
            "Close": range(10, 20),
            "Volume": [1000] * 10,
        },
        index=idx,
    )


def test_resolve_stock_data_cache_hit(monkeypatch):
    cache = StockDataCache()
    calls = {"count": 0}

    def fake_fetch(*args, **kwargs):
        calls["count"] += 1
        return _sample_df()

    monkeypatch.setattr("stock.stock_loader._fetch_us_live_data", fake_fetch)

    first = resolve_stock_data(
        ticker="AAPL",
        market="us",
        start_date="2024-01-02",
        end_date="2024-01-05",
        csv_path="does_not_exist.csv",
        cache=cache,
    )
    second = resolve_stock_data(
        ticker="AAPL",
        market="us",
        start_date="2024-01-02",
        end_date="2024-01-05",
        csv_path="does_not_exist.csv",
        cache=cache,
    )

    assert first["source"] == "live"
    assert second["source"] == "cache"
    assert calls["count"] == 1


def test_resolve_stock_data_csv_hit(tmp_path, monkeypatch):
    csv_path = tmp_path / "AAPL.csv"
    _sample_df().reset_index(names="Date").to_csv(csv_path, index=False)

    def should_not_fetch(*args, **kwargs):
        raise AssertionError("live fetch should not run when csv covers range")

    monkeypatch.setattr("stock.stock_loader._fetch_us_live_data", should_not_fetch)

    cache = StockDataCache()
    result = resolve_stock_data(
        ticker="AAPL",
        market="us",
        start_date="2024-01-03",
        end_date="2024-01-04",
        csv_path=str(csv_path),
        cache=cache,
    )

    assert result["source"] == "csv"
    assert not result["df"].empty


def test_resolve_stock_data_live_fetch_when_csv_missing(monkeypatch):
    def fake_fetch(*args, **kwargs):
        return _sample_df()

    monkeypatch.setattr("stock.stock_loader._fetch_us_live_data", fake_fetch)

    result = resolve_stock_data(
        ticker="AAPL",
        market="us",
        start_date="2024-01-01",
        end_date="2024-01-03",
        csv_path="missing.csv",
        cache=StockDataCache(),
    )

    assert result["source"] == "live"
    assert len(result["df"]) == 10


def test_resolve_stock_data_tw_live_fetch_when_csv_missing(monkeypatch):
    def fake_tw_fetch(*args, **kwargs):
        return _sample_df()

    monkeypatch.setattr("stock.stock_loader._fetch_tw_live_data", fake_tw_fetch)

    result = resolve_stock_data(
        ticker="2330",
        market="tw",
        start_date="2024-01-01",
        end_date="2024-01-03",
        csv_path="missing_tw.csv",
        cache=StockDataCache(),
    )

    assert result["source"] == "live"
    assert len(result["df"]) == 10


def test_resolve_stock_data_tw_live_fetch_and_download_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "2330.csv"

    def fake_tw_fetch(*args, **kwargs):
        return _sample_df()

    monkeypatch.setattr("stock.stock_loader._fetch_tw_live_data", fake_tw_fetch)

    result = resolve_stock_data(
        ticker="2330",
        market="tw",
        start_date="2024-01-01",
        end_date="2024-01-03",
        csv_path=str(csv_path),
        download_stock_data=True,
        cache=StockDataCache(),
    )

    assert result["source"] == "live+csv_download"
    assert csv_path.exists()
