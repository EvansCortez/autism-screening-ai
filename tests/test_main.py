from fastapi.testclient import TestClient

from src.main import (
    app,
    build_clinical_impression,
    extract_domain_flags,
    risk_band,
    summarize_follow_up,
)


client = TestClient(app)


def base_payload():
    return {
        "A1_Score": 1,
        "A2_Score": 1,
        "A3_Score": 0,
        "A4_Score": 1,
        "A5_Score": 1,
        "A6_Score": 0,
        "A7_Score": 1,
        "A8_Score": 1,
        "A9_Score": 1,
        "A10_Score": 0,
        "age": 27,
        "gender": "m",
        "ethnicity": "Latino",
        "jaundice": "yes",
        "country_of_res": "Spain",
        "relation": "Self",
    }


def report_payload():
    return {
        **base_payload(),
        "subject_name": "Sample Participant",
        "screener_name": "Self-report",
        "context": "Adult self-report screening",
        "caregiver_concerns": "limited eye contact and repetitive play",
        "transcript_excerpt": "repeated phrases and turn taking difficulty",
        "observed_behaviors": ["limited eye contact", "conversation difficulty"],
        "follow_up_responses": [
            "Concern is consistent across settings or situations",
            "Challenges are affecting routines, work, or study",
        ],
        "prior_screening_status": "Prior screening flagged developmental concerns",
        "referral_status": "Referral pending clinician review",
        "follow_up_plan": "Review with a clinician and discuss specialist follow-up.",
    }


def test_predict_endpoint_returns_expected_shape():
    response = client.post("/predict", json=base_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["asd_traits_detected"] is True
    assert body["sum_score"] == 7
    assert body["risk_band"] == "high"
    assert 0 <= body["probability"] <= 1


def test_screening_report_endpoint_returns_follow_up_summary():
    response = client.post("/screening-report", json=report_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["subject_name"] == "Sample Participant"
    assert body["screening_result"]["risk_band"] == "high"
    assert body["report"]["follow_up_summary"]["referral_status"] == "Referral pending clinician review"
    assert "Concern is consistent across settings or situations" in body["report"]["follow_up_summary"]["follow_up_items"]


def test_risk_band_thresholds():
    assert risk_band(0.2) == "low"
    assert risk_band(0.6) == "moderate"
    assert risk_band(0.95) == "high"


def test_extract_domain_flags_finds_keywords():
    flags = extract_domain_flags(
        "limited eye contact, speech delay, repetitive routines, and turn taking difficulty"
    )
    assert "eye contact" in flags["social_interaction"]
    assert "speech delay" in flags["communication"]
    assert "repetitive" in flags["restricted_or_repetitive_patterns"]


def test_summarize_follow_up_defaults_when_empty():
    summary = summarize_follow_up([], "No prior screening documented", "No referral started", "Monitor")
    assert summary["follow_up_items"] == ["No structured follow-up responses documented."]


def test_clinical_impression_mentions_demo_scope():
    impression = build_clinical_impression(
        "Sample Participant",
        0.91,
        "high",
        7,
        {"social_interaction": ["eye contact"], "communication": [], "restricted_or_repetitive_patterns": []},
    )
    assert "adult AQ-10 demo" in impression
