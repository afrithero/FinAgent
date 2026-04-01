import concurrent.futures
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)
from datetime import timedelta
from typing import Any

try:
    import yfinance as yf
except ModuleNotFoundError:  # pragma: no cover - optional dependency in some test envs
    yf = None

try:
    from twstock import Stock
except ModuleNotFoundError:  # pragma: no cover - optional dependency in some test envs
    Stock = None


def _to_timestamp(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.today().normalize()
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid date value: {value}")
    return ts.normalize()


def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.droplevel(-1)

    date_col = next((c for c in out.columns if c.lower() == "date"), None)
    if date_col:
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.set_index(date_col)
        out.index.name = "Date"

    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "capacity": "Volume",
    }
    out = out.rename(columns=rename_map)
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[required].sort_index()
    return out


class StockDataCache:
    """In-memory, process-local stock data cache."""

    def __init__(self):
        self._store: dict[tuple[str, str], dict[str, Any]] = {}

    def clear(self):
        self._store.clear()

    def set(self, ticker: str, market: str, df: pd.DataFrame):
        normalized = _normalize_ohlcv_df(df)
        if normalized.empty:
            return
        key = (ticker.upper(), market.lower())
        self._store[key] = {
            "df": normalized,
            "start": normalized.index.min().normalize(),
            "end": normalized.index.max().normalize(),
        }

    def get(
        self, ticker: str, market: str, start_date: Any, end_date: Any
    ) -> pd.DataFrame | None:
        key = (ticker.upper(), market.lower())
        item = self._store.get(key)
        if not item:
            return None

        req_start = _to_timestamp(start_date)
        req_end = _to_timestamp(end_date)
        if req_start < item["start"] or req_end > item["end"]:
            return None
        return item["df"].loc[req_start:req_end].copy()


STOCK_DATA_CACHE = StockDataCache()


def _read_csv_if_covered(csv_path: str, start_date: Any, end_date: Any) -> pd.DataFrame | None:
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    normalized = _normalize_ohlcv_df(df)
    if normalized.empty:
        return None

    req_start = _to_timestamp(start_date)
    req_end = _to_timestamp(end_date)
    data_start = normalized.index.min().normalize()
    data_end = normalized.index.max().normalize()
    if req_start < data_start or req_end > data_end:
        return None
    return normalized.loc[req_start:req_end].copy()


def _fetch_us_live_data(ticker: str, start_date: Any, end_date: Any) -> pd.DataFrame:
    if yf is None:
        raise ModuleNotFoundError("yfinance is required for US live data fetch.")
    start = _to_timestamp(start_date)
    end = _to_timestamp(end_date) + timedelta(days=1)  # yfinance end date is exclusive
    df = yf.download(
        ticker.upper(),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    normalized = _normalize_ohlcv_df(df)
    if normalized.empty:
        raise ValueError(
            f"No live market data returned for {ticker.upper()} in range "
            f"{start.strftime('%Y-%m-%d')} to {(end - timedelta(days=1)).strftime('%Y-%m-%d')}."
        )
    return normalized


def _fetch_tw_live_data(ticker: str, start_date: Any, end_date: Any) -> pd.DataFrame:
    # yfinance supports TWSE tickers via the ".TW" suffix and is more reliable
    # than twstock, which scrapes TWSE directly and can fail on malformed rows.
    if yf is not None:
        try:
            return _fetch_us_live_data(f"{ticker}.TW", start_date, end_date)
        except Exception as e:
            logger.warning(
                "yfinance fetch failed for %s.TW (%s); falling back to twstock.", ticker, e
            )

    if Stock is None:
        raise ModuleNotFoundError(
            "Neither yfinance nor twstock is available for Taiwan live data fetch."
        )

    start = _to_timestamp(start_date)
    end = _to_timestamp(end_date)
    stock = Stock(ticker)
    stock.fetch_from(year=start.year, month=start.month)
    data_dicts = [d._asdict() for d in stock.data]
    normalized = _normalize_ohlcv_df(pd.DataFrame(data_dicts))
    filtered = normalized.loc[start:end].copy()
    if filtered.empty:
        raise ValueError(
            f"No live market data returned for {ticker.upper()} in range "
            f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}."
        )
    return filtered


def resolve_stock_data(
    ticker: str,
    market: str,
    start_date: Any,
    end_date: Any,
    csv_path: str | None = None,
    download_stock_data: bool = False,
    cache: StockDataCache | None = None,
) -> dict[str, Any]:
    """Resolve stock data path in deterministic order: cache -> CSV -> live fetch."""
    cache = cache or STOCK_DATA_CACHE
    ticker = ticker.upper()
    market = market.lower()

    if not download_stock_data:
        cached = cache.get(ticker, market, start_date, end_date)
        if cached is not None and not cached.empty:
            return {"df": cached, "source": "cache", "csv_path": csv_path}

        from_csv = _read_csv_if_covered(csv_path or "", start_date, end_date)
        if from_csv is not None and not from_csv.empty:
            cache.set(ticker, market, from_csv)
            return {"df": from_csv, "source": "csv", "csv_path": csv_path}

    if market == "us":
        live = _fetch_us_live_data(ticker, start_date, end_date)
    elif market == "tw":
        live = _fetch_tw_live_data(ticker, start_date, end_date)
    else:
        raise ValueError("Currently only supports 'tw' or 'us' market.")

    cache.set(ticker, market, live)
    if download_stock_data and csv_path:
        dirpath = os.path.dirname(csv_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        live.reset_index().to_csv(csv_path, index=False)
        return {"df": live, "source": "live+csv_download", "csv_path": csv_path}
    return {"df": live, "source": "live", "csv_path": csv_path}


class StockLoader:
    def __init__(self, stocks, market, start_date, save_dir):
        self.stocks = stocks # e.g. ["2330"]
        self.market = market # "us" or "tw"
        self.start_date = start_date
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_one_stock_to_csv(self, stock_id: str) -> None:
        logger.info("Processing: %s", stock_id)
        try:
            if self.market == "us":
                if yf is None:
                    raise ModuleNotFoundError("yfinance is required for US market download.")
                start = f"{self.start_date[0]:04d}-{self.start_date[1]:02d}-01"
                df = yf.download(stock_id, start=start, progress=False)
                df = _normalize_ohlcv_df(df).reset_index()
            elif self.market == "tw":
                if Stock is None:
                    raise ModuleNotFoundError("twstock is required for Taiwan market download.")
                stock = Stock(stock_id)
                stock.fetch_from(year=self.start_date[0], month=self.start_date[1])
                data_dicts = [d._asdict() for d in stock.data]
                df = _normalize_ohlcv_df(pd.DataFrame(data_dicts)).reset_index()
            else:
                raise ValueError("Currently only supports 'tw' or 'us' market.")

            filepath = os.path.join(self.save_dir, f"{stock_id}.csv")
            df.to_csv(filepath, index=False)
            logger.info("Saved: %s", filepath)
        except (ValueError, OSError, ModuleNotFoundError) as e:
            logger.error("Error processing %s: %s", stock_id, e, exc_info=True)
    
    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self.save_one_stock_to_csv, stock_id)
                for stock_id in self.stocks
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error("Unexpected error in worker thread: %s", e, exc_info=True)

        logger.info("Finished all runs.")
