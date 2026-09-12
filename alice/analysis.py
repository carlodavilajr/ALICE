"""ALICE analysis engine.

Every function here is pure: it takes a DataFrame or Series and returns
plain Python values or a new DataFrame. Flask only wraps these for HTTP,
so they can be unit-tested and reused from a notebook without the server.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- loading

def load_csv(source) -> pd.DataFrame:
    """Read a CSV from a path, file object, or raw bytes into a DataFrame."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    df = pd.read_csv(source)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_time_column(df: pd.DataFrame) -> str | None:
    """Best guess at the column that indexes time.

    Priority: a column named like a date, then a monotonic integer column
    that looks like years, then any column pandas can parse as datetime.
    """
    names = {c.lower(): c for c in df.columns}
    for key in ("date", "year", "period", "time", "quarter", "month"):
        for lower, original in names.items():
            if key in lower:
                return original
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing:
            if s.between(1800, 2200).all():
                return c
    for c in df.columns:
        if df[c].dtype == object:
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except (ValueError, TypeError):
                continue
    return None


def numeric_columns(df: pd.DataFrame, exclude: str | None = None) -> list[str]:
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in cols if c != exclude]


# ---------------------------------------------------------------- helpers

def _clean(v):
    """Convert numpy scalars and NaN into JSON-safe Python values."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 6)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    out = df if limit is None else df.head(limit)
    return [{k: _clean(v) for k, v in row.items()} for row in out.to_dict("records")]


# ---------------------------------------------------------------- summary

def summarize(df: pd.DataFrame) -> dict:
    """Shape, column types, and descriptive statistics for numeric columns."""
    desc = df.describe().T
    stats = {}
    for col, row in desc.iterrows():
        stats[col] = {k: _clean(v) for k, v in row.items()}
        stats[col]["missing"] = int(df[col].isna().sum())
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "time_column": detect_time_column(df),
        "numeric_columns": numeric_columns(df),
        "stats": stats,
    }


# ---------------------------------------------------------------- transforms

def pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    """Period-over-period growth in percent."""
    return s.pct_change(periods=periods) * 100


def log_diff(s: pd.Series) -> pd.Series:
    """Continuously compounded growth in percent (log difference)."""
    return np.log(s).diff() * 100


def moving_average(s: pd.Series, window: int = 3) -> pd.Series:
    return s.rolling(window=window, min_periods=1).mean()


def index_to_base(s: pd.Series, base_position: int = 0) -> pd.Series:
    """Rebase a series so the value at base_position equals 100."""
    base = s.iloc[base_position]
    return s / base * 100


def real_from_nominal(nominal: pd.Series, deflator: pd.Series, base_position: int = -1) -> pd.Series:
    """Deflate a nominal series to constant prices of the base period.

    real_t = nominal_t * (P_base / P_t)
    """
    base_price = deflator.iloc[base_position]
    return nominal * (base_price / deflator)


def cagr(s: pd.Series, periods_per_year: float = 1.0) -> float | None:
    """Compound annual growth rate between first and last valid observation."""
    s = s.dropna()
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return None
    years = (len(s) - 1) / periods_per_year
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100


def apply_transform(df: pd.DataFrame, column: str, kind: str, **kw) -> pd.Series:
    s = df[column].astype(float)
    if kind == "level":
        return s
    if kind == "pct_change":
        return pct_change(s, periods=int(kw.get("periods", 1)))
    if kind == "log_diff":
        return log_diff(s)
    if kind == "moving_average":
        return moving_average(s, window=int(kw.get("window", 3)))
    if kind == "index":
        return index_to_base(s, base_position=int(kw.get("base_position", 0)))
    if kind == "real":
        deflator = kw.get("deflator")
        if not deflator or deflator not in df.columns:
            raise ValueError("A deflator column (e.g. CPI) is required to compute real values.")
        return real_from_nominal(s, df[deflator].astype(float), base_position=int(kw.get("base_position", -1)))
    raise ValueError(f"Unknown transform: {kind}")


TRANSFORMS = {
    "level": "Level",
    "pct_change": "Percent change",
    "log_diff": "Log difference",
    "moving_average": "Moving average",
    "index": "Index (base = 100)",
    "real": "Real (deflated)",
}


# ---------------------------------------------------------------- statistics

def correlation(df: pd.DataFrame, columns: list[str]) -> dict:
    corr = df[columns].astype(float).corr()
    return {
        "columns": columns,
        "matrix": [[_clean(v) for v in row] for row in corr.values],
    }


def linear_regression(x: pd.Series, y: pd.Series) -> dict:
    """OLS of y on x with a constant. Returns slope, intercept, R^2, and residual info."""
    mask = x.notna() & y.notna()
    xv = x[mask].astype(float).to_numpy()
    yv = y[mask].astype(float).to_numpy()
    n = len(xv)
    if n < 3:
        raise ValueError("At least three paired observations are required for a regression.")
    slope, intercept = np.polyfit(xv, yv, 1)
    fitted = slope * xv + intercept
    resid = yv - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    se_slope = math.sqrt(ss_res / (n - 2) / np.sum((xv - xv.mean()) ** 2)) if n > 2 else float("nan")
    t_stat = slope / se_slope if se_slope and se_slope > 0 else float("nan")
    return {
        "n": n,
        "slope": _clean(slope),
        "intercept": _clean(intercept),
        "r_squared": _clean(r2),
        "correlation": _clean(np.corrcoef(xv, yv)[0, 1]),
        "slope_std_error": _clean(se_slope),
        "t_statistic": _clean(t_stat),
        "equation": f"y = {slope:.4g}·x + {intercept:.4g}",
    }


# ---------------------------------------------------------------- interpretation

def _fmt(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}f}"


def interpret_series(df: pd.DataFrame, column: str, time_col: str | None) -> list[str]:
    """Plain-language observations about one series over the sample."""
    s = df[column].astype(float)
    labels = df[time_col] if time_col else pd.Series(range(len(df)), index=df.index)
    valid = s.dropna()
    notes: list[str] = []
    if len(valid) < 2:
        return [f"{column}: not enough data to describe a trend."]

    first, last = valid.iloc[0], valid.iloc[-1]
    total = (last / first - 1) * 100 if first not in (0, None) and first > 0 else None
    direction = "rose" if last > first else "fell" if last < first else "was unchanged"
    span = f"{labels.iloc[valid.index[0]]} to {labels.iloc[valid.index[-1]]}"
    if total is not None:
        notes.append(f"{column} {direction} from {_fmt(first)} to {_fmt(last)} over {span}, a change of {total:+.1f}%.")
    else:
        notes.append(f"{column} {direction} from {_fmt(first)} to {_fmt(last)} over {span}.")

    g = cagr(valid)
    if g is not None:
        notes.append(f"Compound growth averaged {g:.2f}% per period.")

    changes = valid.pct_change().dropna() * 100
    if len(changes) >= 3 and (valid > 0).all():
        notes.append(
            f"Period growth averaged {changes.mean():.2f}% with a standard deviation of {changes.std():.2f} points; "
            f"the largest gain was {changes.max():+.2f}% and the sharpest contraction {changes.min():+.2f}%."
        )
        negatives = int((changes < 0).sum())
        if negatives:
            notes.append(f"{column} contracted in {negatives} of {len(changes)} periods.")

    peak_i, trough_i = valid.idxmax(), valid.idxmin()
    notes.append(
        f"Peak of {_fmt(valid.max())} at {labels.loc[peak_i]}; trough of {_fmt(valid.min())} at {labels.loc[trough_i]}."
    )

    # Trend slope over the index, normalized by mean level, gives a rough drift rate.
    x = np.arange(len(valid), dtype=float)
    slope = np.polyfit(x, valid.to_numpy(), 1)[0]
    if valid.mean() != 0:
        drift = slope / abs(valid.mean()) * 100
        notes.append(f"Linear trend implies a drift of {drift:+.2f}% of the mean level per period.")
    return notes


def interpret(df: pd.DataFrame, columns: list[str], time_col: str | None) -> dict:
    """Series-level notes plus the strongest cross-series relationships."""
    out = {c: interpret_series(df, c, time_col) for c in columns}
    relationships: list[str] = []
    if len(columns) >= 2:
        corr = df[columns].astype(float).corr()
        pairs = []
        for i, a in enumerate(columns):
            for b in columns[i + 1:]:
                v = corr.loc[a, b]
                if not math.isnan(v):
                    pairs.append((abs(v), v, a, b))
        pairs.sort(reverse=True)
        for _, v, a, b in pairs[:3]:
            strength = "strong" if abs(v) >= 0.7 else "moderate" if abs(v) >= 0.4 else "weak"
            sign = "positive" if v > 0 else "negative"
            relationships.append(f"{a} and {b}: {strength} {sign} correlation (r = {v:.2f}).")
        relationships.append("Correlation over a time sample is not causation; trending series correlate by construction. Compare growth rates or differences before drawing conclusions.")
    return {"series": out, "relationships": relationships}
