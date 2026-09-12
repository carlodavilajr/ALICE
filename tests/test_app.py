import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_sample_and_series_roundtrip(client):
    r = client.get("/api/sample")
    assert r.status_code == 200
    dataset_id = r.get_json()["dataset_id"]

    r = client.post("/api/series", json={"dataset_id": dataset_id, "columns": ["nominal_gdp_trillions"], "transform": "pct_change"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["image"].startswith("iVBOR")  # PNG base64 header
    assert len(body["table"]) == 36


def test_regression_endpoint(client):
    dataset_id = client.get("/api/sample").get_json()["dataset_id"]
    r = client.post("/api/regression", json={"dataset_id": dataset_id, "x": "nominal_gdp_trillions", "y": "unemployment_rate",
                                             "x_transform": "pct_change", "y_transform": "level"})
    assert r.status_code == 200
    assert -1 <= r.get_json()["fit"]["correlation"] <= 1


def test_bad_dataset_id(client):
    r = client.post("/api/interpret", json={"dataset_id": "nope", "columns": ["x"]})
    assert r.status_code == 400
