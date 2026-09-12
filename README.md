# ALICE

By Carlo S. Davila

An economic analytical tool. ALICE reads a CSV of economic time series, detects the time column, and lets you transform, plot, regress, and interpret the data from a browser interface. Python does the analysis (Pandas, NumPy, Matplotlib); Flask serves a plain HTML, CSS, and JavaScript front end.

Built for students and researchers who want quick, correct views of macroeconomic data without writing a notebook each time.

## What it does

Transforms (applied to any numeric column)
- Level
- Percent change (period over period, configurable lag)
- Log difference (continuously compounded growth)
- Moving average (configurable window)
- Index to a base period (base = 100)
- Real values from nominal, deflated by a price index column (CPI or a deflator) at a chosen base period

Analysis
- Descriptive statistics for every numeric column (count, mean, standard deviation, quartiles, missing values)
- Compound annual growth rate for each plotted series
- Ordinary least squares regression of one series on another, with slope, intercept, R², correlation, standard error, and t statistic; either variable can be transformed first (e.g. unemployment on GDP growth for an Okun's law style fit)
- Correlation matrix across selected series, on levels or on any transform
- Histogram of a series with the mean marked
- Written notes: total change, compound growth, average and volatility of period growth, number of contractions, peak and trough with dates, linear drift, and the strongest pairwise correlations, with a warning about spurious correlation between trending series

Output
- Line and bar charts rendered by Matplotlib, downloadable as PNG
- Transformed data exported as CSV

## Project layout

```
alice/
  app.py                  Flask server and JSON API
  alice/
    analysis.py           pure functions: loading, transforms, regression, interpretation
    charts.py             Matplotlib rendering to PNG
  static/
    style.css
    app.js                front end logic
  templates/
    index.html
  sample_data/
    sample_macro.csv      synthetic annual data, 1990 to 2025 (not real statistics)
  tests/
    test_analysis.py
    test_app.py
  requirements.txt
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/carlodavilajr/alice.git
cd alice
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open http://127.0.0.1:5001. Click "Load sample dataset" or upload your own CSV. The first row must be column names, and one column should hold the date or year.

Real data that works out of the box: any FRED download (https://fred.stlouisfed.org) exported as CSV, or a World Bank indicator CSV with the metadata rows removed.

## Test

```bash
python -m pytest
```

## API

All analysis endpoints take JSON with a `dataset_id` returned by `/api/upload` or `/api/sample`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/upload` | multipart CSV upload; returns summary and preview |
| GET | `/api/sample` | load the bundled sample dataset |
| POST | `/api/series` | transform columns and render a line or bar chart |
| POST | `/api/regression` | OLS of `y` on `x`, each optionally transformed |
| POST | `/api/correlation` | correlation matrix and heatmap |
| POST | `/api/histogram` | distribution of one column |
| POST | `/api/interpret` | written notes on the selected columns |
| POST | `/api/export` | transformed table as CSV download |

## Notes on method

Percent change is arithmetic: (x_t / x_{t-1} - 1) × 100. Log difference is ln(x_t) - ln(x_{t-1}), scaled to percent, which is symmetric and additive across periods. Real values use real_t = nominal_t × (P_base / P_t). Regression is OLS with a constant; standard errors assume homoskedastic, uncorrelated errors, which time series often violate, so treat t statistics as indicative rather than conclusive.

## Roadmap

- Fetch series directly from the FRED API by series id
- Multiple regression and lagged regressors
- Seasonal adjustment and frequency conversion
- Persistent sessions and saved analyses
