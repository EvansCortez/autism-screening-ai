from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.research_evidence import FIGURES, evidence_payload, verified_manifest

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "autism_model.pkl"
WEB_PATH = ROOT_DIR / "src" / "static" / "index.html"
AQ_COLUMNS = [f"A{i}_Score" for i in range(1, 11)]

SOCIAL_KEYWORDS = {
    "eye contact",
    "name response",
    "pointing",
    "shared attention",
    "joint attention",
    "turn taking",
    "social reciprocity",
    "play",
    "gesture",
}
COMMUNICATION_KEYWORDS = {
    "speech delay",
    "language delay",
    "echolalia",
    "nonverbal",
    "repetitive language",
    "limited words",
    "limited phrases",
    "conversation",
    "pronoun reversal",
}
BEHAVIOR_KEYWORDS = {
    "stimming",
    "sensory",
    "routine",
    "meltdown",
    "repetitive",
    "hand flapping",
    "rigid",
    "fixation",
    "toe walking",
}


class ScreeningData(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
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
                "relation": "Parent",
            }
        }
    )

    A1_Score: int = Field(strict=True, ge=0, le=1)
    A2_Score: int = Field(strict=True, ge=0, le=1)
    A3_Score: int = Field(strict=True, ge=0, le=1)
    A4_Score: int = Field(strict=True, ge=0, le=1)
    A5_Score: int = Field(strict=True, ge=0, le=1)
    A6_Score: int = Field(strict=True, ge=0, le=1)
    A7_Score: int = Field(strict=True, ge=0, le=1)
    A8_Score: int = Field(strict=True, ge=0, le=1)
    A9_Score: int = Field(strict=True, ge=0, le=1)
    A10_Score: int = Field(strict=True, ge=0, le=1)
    age: float = Field(strict=True, ge=18)
    gender: str
    ethnicity: str
    jaundice: str
    country_of_res: str
    relation: str

    @field_validator("gender", "ethnicity", "jaundice", "country_of_res", "relation")
    @classmethod
    def nonblank_category(cls, value):
        if not value.strip():
            raise ValueError("Enter a nonblank dataset category")
        return value.strip()


class ScreeningReportRequest(ScreeningData):
    subject_name: str = "Participant"
    screener_name: str = "Respondent"
    context: str = "Adult self-report screening"
    caregiver_concerns: str = ""
    transcript_excerpt: str = ""
    observed_behaviors: list[str] = Field(default_factory=list)
    follow_up_responses: list[str] = Field(default_factory=list)
    prior_screening_status: str = "No prior screening documented"
    referral_status: str = "No referral started"
    follow_up_plan: str = "Monitor concerns and review with a clinician."


app = FastAPI(title="ASD Screening API", version="1.0.0")
app.mount("/static", StaticFiles(directory=WEB_PATH.parent), name="static")

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@app.get("/")
def home():
    if WEB_PATH.exists():
        return FileResponse(WEB_PATH)
    return {"message": "ASD Screening API is running"}


@app.get("/api")
def api_status():
    return {"message": "ASD Screening API is running"}


@app.get("/health")
def health():
    return {"model_loaded": model is not None}


@app.get("/research-evidence")
def research_evidence():
    try:
        return evidence_payload()
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=503, detail="Pinned research evidence is unavailable. Live demo predictions are separate.") from error


@app.get("/research-figures/{filename}")
def research_figure(filename: str):
    if filename.removesuffix(".png") not in FIGURES or not filename.endswith(".png"):
        raise HTTPException(status_code=404, detail="Research figure not found")
    try:
        verified_manifest()
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=503, detail="Pinned research evidence is unavailable") from error
    return FileResponse(ROOT_DIR / "research/report/figures" / filename)


@app.post("/predict")
def predict_asd(data: ScreeningData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Run src/train.py first.")

    payload = normalize_payload(data.model_dump())
    features = pd.DataFrame([payload])
    prediction = model.predict(features)[0]
    probability = float(model.predict_proba(features)[0][1])

    return {
        "asd_traits_detected": bool(prediction),
        "probability": round(probability, 4),
        "sum_score": payload["sum_score"],
        "risk_band": risk_band(probability),
        "dataset_label": "YES" if prediction else "NO",
        "observed_rule_label": "YES" if payload["sum_score"] >= 7 else "NO",
        "model_label_score": round(probability, 4),
        "interpretation": "Uncalibrated class-YES model score; not a clinical probability of autism. Legacy probability and risk_band keys are retained for compatibility only.",
        "disclaimer": "Educational adult dataset-label demo. Not clinical diagnostic accuracy or a diagnosis.",
    }


@app.post("/screening-report")
def screening_report(data: ScreeningReportRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Run src/train.py first.")

    payload = normalize_payload(data.model_dump())
    features = pd.DataFrame([payload])
    prediction = model.predict(features)[0]
    probability = float(model.predict_proba(features)[0][1])
    risk = risk_band(probability)
    notes_blob = " ".join(
        [
            data.caregiver_concerns,
            data.transcript_excerpt,
            " ".join(data.observed_behaviors),
        ]
    ).lower()
    domain_flags = extract_domain_flags(notes_blob)
    aq_summary = summarize_aq(payload)
    next_steps = build_next_steps(probability, domain_flags, aq_summary["high_response_count"])
    follow_up_summary = summarize_follow_up(
        data.follow_up_responses,
        data.prior_screening_status,
        data.referral_status,
        data.follow_up_plan,
    )
    report = {
        "subject_name": data.subject_name,
        "screener_name": data.screener_name,
        "context": data.context,
        "screening_result": {
            "asd_traits_detected": bool(prediction),
            "probability": round(probability, 4),
            "risk_band": risk,
            "sum_score": payload["sum_score"],
            "dataset_label": "YES" if prediction else "NO",
            "observed_rule_label": "YES" if payload["sum_score"] >= 7 else "NO",
            "model_label_score": round(probability, 4),
            "interpretation": "Uncalibrated class-YES model score; not a clinical probability. Legacy risk_band is not a clinically validated risk category.",
        },
        "report": {
            "research_explanation": "The saved model predicts the dataset's YES/NO screening labels; research metrics measure agreement with those labels. The observed >=7 score rule reconstructs those labels in the adult file; it differs from the documented >=6 referral-consideration rule. Neither establishes a diagnosis. Supplied notes do not enter this model.",
            "clinical_impression": build_clinical_impression(
                data.subject_name,
                probability,
                risk,
                aq_summary["high_response_count"],
                domain_flags,
            ),
            "aq_summary": aq_summary,
            "behavioral_domains_flagged": domain_flags,
            "follow_up_summary": follow_up_summary,
            "caregiver_concerns": data.caregiver_concerns or "None provided.",
            "transcript_excerpt": data.transcript_excerpt or "None provided.",
            "recommended_next_steps": next_steps,
        },
        "disclaimer": "Educational adult dataset-label summary, not a clinical diagnosis or a clinical probability of autism.",
    }

    return report


def normalize_payload(payload: dict) -> dict:
    normalized = dict(payload)
    normalized["gender"] = normalized["gender"].strip().lower()
    normalized["jaundice"] = normalized["jaundice"].strip().lower()
    normalized["sum_score"] = int(sum(normalized[column] for column in AQ_COLUMNS))
    return normalized


def risk_band(probability: float) -> str:
    if probability >= 0.8:
        return "high"
    if probability >= 0.45:
        return "moderate"
    return "low"


def extract_domain_flags(notes_blob: str) -> dict:
    return {
        "social_interaction": keyword_hits(notes_blob, SOCIAL_KEYWORDS),
        "communication": keyword_hits(notes_blob, COMMUNICATION_KEYWORDS),
        "restricted_or_repetitive_patterns": keyword_hits(notes_blob, BEHAVIOR_KEYWORDS),
    }


def keyword_hits(notes_blob: str, keywords: set[str]) -> list[str]:
    return sorted(keyword for keyword in keywords if keyword in notes_blob)


def summarize_aq(payload: dict) -> dict:
    positive_items = [column for column in AQ_COLUMNS if payload[column] == 1]
    return {
        "positive_items": positive_items,
        "high_response_count": len(positive_items),
    }


def build_next_steps(probability: float, domain_flags: dict, high_response_count: int) -> list[str]:
    # Preserve the helper signature, but do not derive healthcare recommendations from this model.
    return [
        "Compare the predicted dataset label with the observed >=7 score rule.",
        "Read the committed research methods, provenance and uncertainty limitations.",
        "Treat supplied notes and workflow fields as unvalidated context, not diagnostic evidence.",
        "Do not use this model output to make healthcare decisions.",
    ]


def summarize_follow_up(
    follow_up_responses: list[str],
    prior_screening_status: str,
    referral_status: str,
    follow_up_plan: str,
) -> dict:
    normalized_responses = [item.strip() for item in follow_up_responses if item.strip()]
    return {
        "prior_screening_status": prior_screening_status,
        "referral_status": referral_status,
        "follow_up_items": normalized_responses or ["No structured follow-up responses documented."],
        "follow_up_plan": follow_up_plan,
    }


def build_clinical_impression(
    subject_name: str,
    probability: float,
    risk: str,
    high_response_count: int,
    domain_flags: dict,
) -> str:
    # Legacy response key/helper retained; this text is a research explanation, not an impression.
    return (
        f"{subject_name}'s adult AQ-10 demo contains {high_response_count} pre-scored item points out of 10. "
        f"The class-YES model score is {probability:.4f}; it is uncalibrated and is not a clinical "
        "probability of autism. The model was trained to reproduce dataset screening labels. "
        "Notes are supplied context only and do not establish diagnostic findings."
    )
