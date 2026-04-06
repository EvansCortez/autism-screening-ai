import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "train.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "autism_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

AQ_COLUMNS = [f"A{i}_Score" for i in range(1, 11)]
CATEGORICAL_COLUMNS = [
    "gender",
    "ethnicity",
    "jaundice",
    "country_of_res",
    "relation",
]
NUMERIC_COLUMNS = AQ_COLUMNS + ["age", "sum_score"]
DROP_COLUMNS = ["ID", "age_desc", "used_app_before", "austim", "result", "Class/ASD"]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"jundice": "jaundice", "contry_of_res": "country_of_res"})
    df = df.replace({"?": "Unknown", "others": "Others"})

    for column in AQ_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sum_score"] = df[AQ_COLUMNS].sum(axis=1)

    return df


def build_pipeline() -> ImbPipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )

    return ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("oversample", RandomOverSampler(random_state=42)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_model() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = load_dataset()
    y = df["Class/ASD"].map({"NO": 0, "YES": 1}).astype(int)
    X = df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": list(X.columns),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    return metrics


if __name__ == "__main__":
    train_model()
