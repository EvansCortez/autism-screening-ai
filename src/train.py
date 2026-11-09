import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
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


def cross_validate_pipeline(X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5) -> dict:
    cv_folds = min(cv_folds, y_train.value_counts().min())
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_validate(
        build_pipeline(),
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "f1"],
    )
    return {
        "folds": cv_folds,
        "accuracy_mean": round(float(scores["test_accuracy"].mean()), 4),
        "accuracy_std": round(float(scores["test_accuracy"].std()), 4),
        "f1_mean": round(float(scores["test_f1"].mean()), 4),
        "f1_std": round(float(scores["test_f1"].std()), 4),
    }


def baseline_accuracy(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    return round(float(accuracy_score(y_test, baseline.predict(X_test))), 4)


def train_model(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict:
    df = load_dataset(data_path)
    y = df["Class/ASD"].map({"NO": 0, "YES": 1}).astype(int)
    X = df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cv_metrics = cross_validate_pipeline(X_train, y_train)

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
        "cross_validation": cv_metrics,
        "baseline_accuracy": baseline_accuracy(X_train, y_train, X_test, y_test),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": list(X.columns),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Held-out accuracy: {metrics['accuracy']:.4f}")
    print(f"Cross-validated accuracy: {cv_metrics['accuracy_mean']:.4f} (+/- {cv_metrics['accuracy_std']:.4f})")
    print(f"Baseline (majority class) accuracy: {metrics['baseline_accuracy']:.4f}")
    return metrics


if __name__ == "__main__":
    train_model()
