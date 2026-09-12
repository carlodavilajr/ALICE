"""ALICE web server. Run with `python app.py` and open http://127.0.0.1:5001."""
from __future__ import annotations

import io
import os
import uuid

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from alice import analysis, charts

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload cap

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "sample_macro.csv")

# In-memory dataset store. V1 keeps one process; a restart clears it.
DATASETS: dict[str, pd.DataFrame] = {}


def _get_df(dataset_id: str) -> pd.DataFrame:
    df = DATASETS.get(dataset_id)
    if df is None:
        raise KeyError("Dataset not found. Upload it again.")
    return df


def _register(df: pd.DataFrame, name: str) -> dict:
    dataset_id = uuid.uuid4().hex
    DATASETS[dataset_id] = df
    summary = analysis.summarize(df)
    return {"dataset_id": dataset_id, "name": name, "summary": summary,
            "preview": analysis.records(df, limit=12), "transforms": analysis.TRANSFORMS}


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return _error("Choose a CSV file first.")
    try:
        df = analysis.load_csv(file.stream)
    except Exception as exc:  # pandas raises many types; surface the message
        return _error(f"Could not parse CSV: {exc}")
    if df.empty:
        return _error("The file contains no rows.")
    return jsonify(_register(df, file.filename))


@app.get("/api/sample")
def sample():
    df = analysis.load_csv(SAMPLE_PATH)
    return jsonify(_register(df, "sample_macro.csv (synthetic)"))


@app.post("/api/series")
def series():
    """Compute transformed series and render a line or bar chart."""
    body = request.get_json(force=True)
    try:
        df = _get_df(body["dataset_id"])
        columns = body["columns"]
        transform = body.get("transform", "level")
        time_col = body.get("time_column") or analysis.detect_time_column(df)
        chart_kind = body.get("chart", "line")
        params = body.get("params", {})
        out = {}
        for col in columns:
            out[f"{col} ({analysis.TRANSFORMS[transform]})" if transform != "level" else col] = analysis.apply_transform(df, col, transform, **params)
        result = pd.DataFrame(out)
        if time_col:
            result.insert(0, time_col, df[time_col])
        title = ", ".join(columns) + ("" if transform == "level" else f" ({analysis.TRANSFORMS[transform]})")
        ylabel = "percent" if transform in ("pct_change", "log_diff") else ("index" if transform == "index" else "")
        render = charts.bar_chart if chart_kind == "bar" else charts.line_chart
        image = render(df, out, time_col, title, ylabel)
        stats = {name: {"mean": analysis._clean(s.mean()), "std": analysis._clean(s.std()),
                        "min": analysis._clean(s.min()), "max": analysis._clean(s.max()),
                        "cagr": analysis._clean(analysis.cagr(s)) if transform in ("level", "real", "index", "moving_average") else None}
                 for name, s in out.items()}
        return jsonify({"image": image, "table": analysis.records(result), "stats": stats,
                        "columns": list(result.columns)})
    except (KeyError, ValueError) as exc:
        return _error(str(exc))


@app.post("/api/regression")
def regression():
    body = request.get_json(force=True)
    try:
        df = _get_df(body["dataset_id"])
        x_col, y_col = body["x"], body["y"]
        tx, ty = body.get("x_transform", "level"), body.get("y_transform", "level")
        x = analysis.apply_transform(df, x_col, tx)
        y = analysis.apply_transform(df, y_col, ty)
        fit = analysis.linear_regression(x, y)
        time_col = body.get("time_column") or analysis.detect_time_column(df)
        labels = df[time_col] if time_col else None
        xl = x_col if tx == "level" else f"{x_col} ({analysis.TRANSFORMS[tx]})"
        yl = y_col if ty == "level" else f"{y_col} ({analysis.TRANSFORMS[ty]})"
        image = charts.scatter_regression(x, y, xl, yl, fit, labels)
        return jsonify({"fit": fit, "image": image, "x_label": xl, "y_label": yl})
    except (KeyError, ValueError) as exc:
        return _error(str(exc))


@app.post("/api/correlation")
def correlation():
    body = request.get_json(force=True)
    try:
        df = _get_df(body["dataset_id"])
        columns = body["columns"]
        transform = body.get("transform", "level")
        if len(columns) < 2:
            return _error("Pick at least two columns for a correlation matrix.")
        work = pd.DataFrame({c: analysis.apply_transform(df, c, transform) for c in columns})
        corr = analysis.correlation(work, columns)
        image = charts.heatmap(corr["columns"], corr["matrix"])
        return jsonify({"correlation": corr, "image": image})
    except (KeyError, ValueError) as exc:
        return _error(str(exc))


@app.post("/api/histogram")
def histogram():
    body = request.get_json(force=True)
    try:
        df = _get_df(body["dataset_id"])
        col = body["column"]
        transform = body.get("transform", "level")
        s = analysis.apply_transform(df, col, transform)
        name = col if transform == "level" else f"{col} ({analysis.TRANSFORMS[transform]})"
        return jsonify({"image": charts.histogram(s, name, bins=int(body.get("bins", 20)))})
    except (KeyError, ValueError) as exc:
        return _error(str(exc))


@app.post("/api/interpret")
def interpret():
    body = request.get_json(force=True)
    try:
        df = _get_df(body["dataset_id"])
        columns = body["columns"]
        time_col = body.get("time_column") or analysis.detect_time_column(df)
        return jsonify(analysis.interpret(df, columns, time_col))
    except (KeyError, ValueError) as exc:
        return _error(str(exc))


@app.post("/api/export")
def export():
    """Return the transformed table as a downloadable CSV."""
    body = request.get_json(force=True)
    try:
        df = _get_df(body["dataset_id"])
        columns = body["columns"]
        transform = body.get("transform", "level")
        time_col = body.get("time_column") or analysis.detect_time_column(df)
        out = pd.DataFrame({c: analysis.apply_transform(df, c, transform, **body.get("params", {})) for c in columns})
        if time_col:
            out.insert(0, time_col, df[time_col])
        buf = io.BytesIO(out.to_csv(index=False).encode("utf-8"))
        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name=f"alice_{transform}.csv")
    except (KeyError, ValueError) as exc:
        return _error(str(exc))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
