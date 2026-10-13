import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.main as main
import src.research_evidence as evidence

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(main.app)


def payload():
    return {**{f"A{i}_Score": int(i <= 7) for i in range(1, 11)}, "age": 27,
            "gender": "m", "ethnicity": "Latino", "jaundice": "no",
            "country_of_res": "United States", "relation": "Self"}


@pytest.mark.parametrize("endpoint", ["/predict", "/screening-report"])
def test_live_results_are_actual_saved_model_outputs(endpoint):
    response = client.post(endpoint, json=payload())
    assert response.status_code == 200
    body = response.json()
    result = body if endpoint == "/predict" else body["screening_result"]
    assert result["sum_score"] == 7
    assert result["dataset_label"] == ("YES" if result["asd_traits_detected"] else "NO")
    assert result["model_label_score"] == result["probability"]
    assert "not a clinical probability" in result["interpretation"]
    assert result["observed_rule_label"] == "YES"


@pytest.mark.parametrize("value", [-1, 2, 0.5, "1", True])
def test_aq_requires_pre_scored_binary_integers(value):
    response = client.post("/predict", json={**payload(), "A1_Score": value})
    assert response.status_code == 422


@pytest.mark.parametrize("age", [0, 17, "NaN", "Infinity", "27", True])
def test_api_enforces_finite_adult_age(age):
    assert client.post("/predict", json={**payload(), "age": age}).status_code == 422


def test_numeric_integer_and_fractional_api_ages_remain_supported():
    for age in [18, 27.5]:
        assert client.post("/predict", json={**payload(), "age": age}).status_code == 200


def test_blank_categories_rejected():
    assert client.post("/predict", json={**payload(), "ethnicity": "  "}).status_code == 422


@pytest.mark.parametrize("endpoint", ["/predict", "/screening-report"])
def test_missing_model_has_actionable_503(monkeypatch, endpoint):
    monkeypatch.setattr(main, "model", None)
    assert client.post(endpoint, json=payload()).status_code == 503


def test_report_preserves_supplied_text_without_clinical_interpretation():
    supplied = '<img src=x onerror="alert(1)"> & notes'
    response = client.post("/screening-report", json={**payload(), "subject_name": supplied, "caregiver_concerns": supplied})
    body = response.json()
    assert response.status_code == 200
    assert body["subject_name"] == supplied and body["report"]["caregiver_concerns"] == supplied
    assert "estimated model probability" not in body["report"]["clinical_impression"]
    assert "not a clinical" in body["report"]["clinical_impression"]
    assert not any("Recommend follow-up" in step for step in body["report"]["recommended_next_steps"])


def test_committed_evidence_provenance_and_metrics():
    response = client.get("/research-evidence")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "ace0186c2ee3c6a8f40d0b20803acc0a59be84c7"
    assert body["rows"] == 704 and body["seeds"] == [42, 137, 2026]
    assert body["rule_disagreements"] == {"ge6": 70, "ge7": 0}
    assert next(row for row in body["metrics"] if row["method"] == "dummy")["accuracy"] == pytest.approx(515 / 704)
    assert len(body["figures"]) == 6
    for path, checksum in body["files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == checksum
    assert body["country_held_out_auc"] == pytest.approx(0.6864591359736991)


def test_evidence_failure_does_not_break_live_predictions(monkeypatch, tmp_path):
    changed = json.loads(evidence.MANIFEST_PATH.read_text())
    changed["files"]["research/evaluation/audit.json"] = "0" * 64
    manifest = tmp_path / "changed.json"
    manifest.write_text(json.dumps(changed))
    monkeypatch.setattr(evidence, "MANIFEST_PATH", manifest)
    assert client.get("/research-evidence").status_code == 503
    assert client.get("/research-figures/fig1_label_by_score.png").status_code == 503
    assert client.post("/predict", json=payload()).status_code == 200


def test_figure_routes_are_allowlisted():
    assert client.get("/research-figures/not-a-figure.png").status_code == 404
    response = client.get("/research-figures/fig1_label_by_score.png")
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"


class Markup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.labels, self.controls = [], [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "label":
            self.labels.append(attrs.get("for"))
        if tag in ["input", "textarea", "select"]:
            self.controls.append(attrs.get("id"))


def test_native_markup_has_unique_ids_associated_labels_and_no_unsafe_sinks():
    markup = Markup()
    markup.feed(client.get("/").text)
    assert len(markup.ids) == len(set(markup.ids))
    assert all(control in markup.labels for control in markup.controls)
    script = client.get("/static/app.js").text
    assert "innerHTML" not in script and "insertAdjacentHTML" not in script
    assert client.get("/static/research-version.json").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
