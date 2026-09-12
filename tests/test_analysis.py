import numpy as np
import pandas as pd
import pytest

from alice import analysis


@pytest.fixture
def df():
    return pd.DataFrame({
        "year": [2000, 2001, 2002, 2003, 2004],
        "gdp": [100.0, 110.0, 121.0, 133.1, 146.41],
        "cpi": [100.0, 102.0, 104.04, 106.12, 108.24],
    })


def test_detect_time_column(df):
    assert analysis.detect_time_column(df) == "year"


def test_pct_change(df):
    g = analysis.pct_change(df["gdp"]).dropna()
    assert np.allclose(g, 10.0)


def test_cagr_matches_constant_growth(df):
    assert analysis.cagr(df["gdp"]) == pytest.approx(10.0, rel=1e-6)


def test_index_to_base(df):
    idx = analysis.index_to_base(df["gdp"])
    assert idx.iloc[0] == 100 and idx.iloc[-1] == pytest.approx(146.41)


def test_real_from_nominal_base_last(df):
    real = analysis.real_from_nominal(df["gdp"], df["cpi"], base_position=-1)
    assert real.iloc[-1] == pytest.approx(df["gdp"].iloc[-1])
    assert real.iloc[0] == pytest.approx(100.0 * 108.24 / 100.0)


def test_linear_regression_exact_line():
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = 2 * x + 1
    fit = analysis.linear_regression(x, y)
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["intercept"] == pytest.approx(1.0)
    assert fit["r_squared"] == pytest.approx(1.0)


def test_regression_requires_three_points():
    with pytest.raises(ValueError):
        analysis.linear_regression(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))


def test_interpret_returns_notes(df):
    out = analysis.interpret(df, ["gdp", "cpi"], "year")
    assert "gdp" in out["series"] and len(out["series"]["gdp"]) >= 3
    assert any("correlation" in r for r in out["relationships"])
