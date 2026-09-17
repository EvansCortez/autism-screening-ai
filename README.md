# ASD-Predict

ASD-Predict is an end-to-end machine learning project for autism trait screening support. It trains a classifier on the UCI Adult Autism Screening dataset and serves both raw predictions and a structured screening-report workflow through FastAPI.

This project is designed as a portfolio-ready example of:
- Data cleaning + feature engineering
- Imbalanced classification with oversampling
- Reproducible model artifact generation
- Lightweight ML model serving with FastAPI
- Turning model output into a clinician-style intake summary for an adult AQ-10 demo

## Highlights

- Trains a logistic regression pipeline with:
  - Numeric and categorical preprocessing
  - Missing-value imputation
  - One-hot encoding
  - Class balancing via `RandomOverSampler`
- Engineers `sum_score` from AQ-10 responses (`A1_Score` to `A10_Score`)
- Exposes endpoints for health checks and real-time predictions
- Generates structured screening reports from questionnaire data plus caregiver notes
- Persists:
  - Model at `models/autism_model.pkl`
  - Evaluation summary at `models/metrics.json`

## Tech Stack

- Python
- pandas, scikit-learn, imbalanced-learn
- FastAPI + Uvicorn
- joblib (model serialization)

## Repository Structure

- `src/train.py` - Data loading, preprocessing, training, evaluation, artifact export
- `src/main.py` - FastAPI application serving web UI + API endpoints
- `src/static/index.html` - Browser interface for intake, prediction, and report generation
- `data/train.csv` - Training data
- `models/autism_model.pkl` - Trained model pipeline
- `models/metrics.json` - Saved metrics, cross-validation scores, baseline comparison, and feature list
- `notebooks/` - Exploratory analysis, model experiments, and workflow research
- `requirements.txt` - Python dependencies
- `tests/` - API, training pipeline, and helper tests
- `Dockerfile` - Container image for serving the app
- `.github/workflows/ci.yml` - CI pipeline that runs the test suite

## Quick Start

### 1) Clone and set up environment

```bash
git clone <your-repo-url>
cd autism-screening-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train model artifacts

```bash
python -m src.train
```

Expected outputs:
- `models/autism_model.pkl`
- `models/metrics.json`

### 3) Start the app

```bash
uvicorn src.main:app --reload
```

Open:
- Web UI: `http://127.0.0.1:8000/`
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### 4) Run tests

```bash
pytest
```

### 5) Run notebooks

```bash
jupyter lab
```

Open `notebooks/01_data_exploration.ipynb` to start with the shared helpers in `notebooks/utils.py`.

### 6) Run with Docker (optional)

```bash
docker build -t asd-predict .
docker run -p 8000:8000 asd-predict
```

This serves the app using the model artifacts already committed in `models/`. Re-run `python -m src.train` and rebuild the image if you want the container to reflect a freshly trained model.

## Continuous Integration

`.github/workflows/ci.yml` runs `pytest` on every push and pull request to `main` against the committed dataset and model artifacts.

## API Endpoints

- `GET /` - Web interface for intake, prediction, and report generation
- `GET /api` - API status message
- `GET /health` - Reports whether model artifact is loaded
- `POST /predict` - Returns ASD trait screening prediction + probability
- `POST /screening-report` - Returns a structured screening report with triage-oriented summary fields for the current adult AQ-10 demo

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "relation": "Parent"
  }'
```

### Example Response

```json
{
  "asd_traits_detected": true,
  "probability": 0.9731,
  "sum_score": 7,
  "risk_band": "high",
  "disclaimer": "This is a screening result, not a clinical diagnosis."
}
```

### Example Screening Report Request

```bash
curl -X POST http://127.0.0.1:8000/screening-report \
  -H "Content-Type: application/json" \
  -d '{
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
    "subject_name": "Sample Participant",
    "screener_name": "Self-report",
    "context": "Adult self-report screening",
    "caregiver_concerns": "Respondent notes limited eye contact, repetitive interests, and difficulty with conversation.",
    "transcript_excerpt": "Participant described repeated phrases and difficulty with turn taking.",
    "observed_behaviors": ["limited eye contact", "conversation difficulty", "rigid routines"],
    "follow_up_responses": [
      "Concern is consistent across settings or situations",
      "Challenges are affecting routines, work, or study"
    ],
    "prior_screening_status": "Prior screening flagged developmental concerns",
    "referral_status": "Referral pending clinician review",
    "follow_up_plan": "Review with a clinician and discuss specialist follow-up."
  }'
```

### Example Screening Report Fields

- `screening_result` - risk band, probability, and model screening output
- `report.clinical_impression` - concise narrative summary for handoff
- `report.aq_summary` - positive AQ items and count
- `report.behavioral_domains_flagged` - keyword-derived flags from notes
- `report.follow_up_summary` - prior screening history, referral status, follow-up items, and current plan
- `report.recommended_next_steps` - triage-oriented next actions

## Model Performance (Current Artifacts)

From `models/metrics.json`:
- Held-out accuracy: `1.0`
- 5-fold cross-validated accuracy: `1.0 (+/- 0.0)`
- Majority-class baseline accuracy: `0.7305`
- Training rows: `563`
- Test rows: `141`

Important interpretation note:
- Cross-validation confirms this isn't a fluke of one train/test split: accuracy is `1.0` across all 5 folds. The reason is that the AQ-10 `sum_score` deterministically implies the target label in this dataset (by the design of the AQ-10 screening tool itself, not a code bug), so a perfect score here should be treated as a demo artifact, not evidence of clinical-grade predictive performance.
- The baseline accuracy (always predicting the majority class) is included in `metrics.json` specifically so this gap is visible rather than hidden.
- For production-grade evaluation, add external validation and a dataset that matches the intended population and use case, since cross-validation alone cannot detect leakage that is baked into how the source dataset itself was labeled.

## Research Context

This demo aligns with a broader research direction: AI can help make autism screening workflows faster and more scalable when used as a support tool alongside clinical pathways.

- A 2025 review summarizes how AI is being applied to early screening, diagnosis support, and intervention in young children, while also emphasizing limitations around generalization, data diversity, and ethics.
- Duke's SenseToKnow work shows how objective, multimodal digital behavioral signals (for example gaze, facial features, movement, and confidence scoring) can complement traditional parent-report screening.

Important scope note for this repository:
- This project currently trains on an **adult AQ-10 dataset** and should be treated as an educational ML prototype.
- The report workflow is product-style scaffolding layered on top of that adult model, not a validated pediatric screening system.
- It is not validated for clinical decision-making and should not be used as a diagnostic tool.

## Troubleshooting

- `Model not found. Run src/train.py first.`  
  Run `python -m src.train` to generate `models/autism_model.pkl`.

- `ModuleNotFoundError` or dependency issues  
  Ensure virtual environment is active and reinstall:
  ```bash
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

- `Dataset not found at .../data/train.csv`  
  Confirm `data/train.csv` exists and filename matches exactly.

- API starts but `/predict` returns validation errors  
  Ensure:
  - AQ fields are integers in `{0,1}`
  - `age` is a positive number
  - Required categorical fields are included

- `pytest` or `jupyter` command not found
  Re-activate the virtual environment and reinstall dependencies:
  ```bash
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

- Port already in use
  ```bash
  uvicorn src.main:app --reload --port 8001
  ```

## Roadmap Ideas

- Add model versioning and experiment tracking
- Compare additional classifiers and calibration methods
- Add external validation against a different population/dataset

## Ethical and Clinical Disclaimer

This project is for educational and screening-support purposes only. It is **not** a medical device and does **not** provide a clinical diagnosis. Any autism-related concern should be evaluated by qualified healthcare professionals.

## References

- Zhang, S. (2025). AI-assisted early screening, diagnosis, and intervention for autism in young children. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12036476/).
- Duke Psychiatry (2023). Tablet-based AI app measures multiple behavioral indicators to screen for autism. [Duke article](https://psychiatry.duke.edu/news/tablet-based-ai-app-measures-multiple-behavioral-indicators-screen-autism).
