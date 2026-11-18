"""Market data container and loaders (Yahoo Finance, CSV, synthetic)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")

# Liquid multi-asset ETFs. Using ETFs rather than single stocks largely avoids
# survivorship bias: the universe is fixed up front and none of these delisted.
DEFAULT_UNIVERSE = (
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ",  # equity and real estate
    "TLT", "IEF", "LQD", "HYG",  # rates and credit
    "GLD", "DBC",  # commodities
)


@dataclass(frozen=True, eq=False)
class MarketData:
    """Aligned daily OHLCV panel; every array has shape (n_bars, n_assets)."""

    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __post_init__(self) -> None:
        shape = (len(self.dates), len(self.symbols))
        for field in FIELDS:
            values = getattr(self, field)
            if values.shape != shape:
                raise ValueError(f"{field} has shape {values.shape}, expected {shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{field} contains NaN or infinite values")
        if (self.open <= 0).any() or (self.close <= 0).any():
            raise ValueError("prices must be strictly positive")
        if not self.dates.is_monotonic_increasing or self.dates.has_duplicates:
            raise ValueError("dates must be strictly increasing")

    @property
    def n_bars(self) -> int:
        return len(self.dates)

    @property
    def n_assets(self) -> int:
        return len(self.symbols)

    def truncate(self, end_bar: int) -> "MarketData":
        """Return a copy holding only bars ``[0, end_bar)``."""
        return MarketData(
            self.dates[:end_bar],
            self.symbols,
            *(getattr(self, field)[:end_bar].copy() for field in FIELDS),
        )


def _from_frames(frames: dict[str, pd.DataFrame]) -> MarketData:
    """Inner-join per-symbol frames on date and build a panel."""
    common: pd.Index | None = None
    for frame in frames.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or len(common) == 0:
        raise ValueError("symbols share no common dates")
    symbols = tuple(frames)
    arrays = {
        field: np.column_stack([frames[s].loc[common, field].to_numpy(float) for s in symbols])
        for field in FIELDS
    }
    return MarketData(pd.DatetimeIndex(common), symbols, **arrays)


def load_csv(directory: str | Path, symbols: Sequence[str] | None = None) -> MarketData:
    """Load ``<SYMBOL>.csv`` files with date, open, high, low, close, volume columns."""
    directory = Path(directory)
    paths = (
        [directory / f"{s}.csv" for s in symbols]
        if symbols
        else sorted(directory.glob("*.csv"))
    )
    if not paths:
        raise FileNotFoundError(f"no CSV files found in {directory}")
    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        frame = pd.read_csv(path)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        missing = {"date", *FIELDS} - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frame = frame.set_index("date").sort_index()
        frames[path.stem] = frame[~frame.index.duplicated(keep="last")].dropna(subset=list(FIELDS))
    return _from_frames(frames)


def load_yahoo(
    symbols: Sequence[str] = DEFAULT_UNIVERSE,
    start: str = "2007-05-01",
    end: str | None = None,
    cache_dir: str | Path = "data/cache",
) -> MarketData:
    """Download dividend- and split-adjusted daily bars, caching them as CSV."""
    stamp = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cache = Path(cache_dir) / f"{start}_{stamp}"
    if all((cache / f"{s}.csv").exists() for s in symbols):
        return load_csv(cache, symbols)

    import yfinance as yf

    raw = yf.download(
        list(symbols), start=start, end=end, auto_adjust=True,
        group_by="ticker", progress=False, threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    cache.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        if symbol not in raw.columns.get_level_values(0):
            raise RuntimeError(f"Yahoo Finance returned no data for {symbol}")
        frame = raw[symbol].rename(columns=str.lower)[list(FIELDS)].dropna()
        frame.index.name = "date"
        frame.to_csv(cache / f"{symbol}.csv")
    return load_csv(cache, symbols)


def synthetic_market(n_bars: int = 3000, n_assets: int = 8, seed: int = 0) -> MarketData:
    """Generate a regime-switching market for offline demos and tests.

    A hidden Markov chain moves between calm, stressed and crisis regimes. The
    first half of the assets are risk-on, the second half defensive, so allocation
    decisions conditioned on the regime have real, learnable value.
    """
    rng = np.random.default_rng(seed)
    transition = np.array([[0.985, 0.013, 0.002], [0.030, 0.950, 0.020], [0.010, 0.060, 0.930]])
    market_mu = np.array([0.0007, 0.0001, -0.0020])
    market_vol = np.array([0.007, 0.012, 0.028])

    regimes = np.zeros(n_bars, dtype=int)
    for t in range(1, n_bars):
        regimes[t] = rng.choice(3, p=transition[regimes[t - 1]])

    risk_on = np.arange(n_assets) < n_assets / 2
    beta = np.where(risk_on, rng.uniform(0.8, 1.4, n_assets), rng.uniform(-0.3, 0.2, n_assets))
    carry = np.where(risk_on, 0.0, 0.00015)
    idio_vol = rng.uniform(0.003, 0.009, n_assets)

    market = market_mu[regimes] + market_vol[regimes] * rng.standard_normal(n_bars)
    log_ret = market[:, None] * beta + carry + idio_vol * rng.standard_normal((n_bars, n_assets))
    overnight_share = rng.uniform(0.2, 0.5, (n_bars, n_assets))

    log_close = np.log(100.0) + np.cumsum(log_ret, axis=0)
    log_prev = np.vstack([np.full(n_assets, np.log(100.0)), log_close[:-1]])
    log_open = log_prev + overnight_share * log_ret
    close, open_ = np.exp(log_close), np.exp(log_open)
    wick = np.abs(rng.normal(0, 0.003, (n_bars, n_assets)))
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    volume = np.exp(rng.normal(np.log(5e6), 0.3, (n_bars, n_assets)))

    dates = pd.bdate_range("2005-01-03", periods=n_bars)
    symbols = tuple(f"SYN{i:02d}" for i in range(n_assets))
    return MarketData(dates, symbols, open_, high, low, close, volume)
