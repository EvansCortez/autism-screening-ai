from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "train.csv"
AQ_COLUMNS = [f"A{i}_Score" for i in range(1, 11)]


def load_autism_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"jundice": "jaundice", "contry_of_res": "country_of_res"})
    df = df.replace({"?": "Unknown", "others": "Others"})

    for column in AQ_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sum_score"] = df[AQ_COLUMNS].sum(axis=1)
    return df


def dataset_snapshot(df: pd.DataFrame) -> dict:
    target_counts = (
        df["Class/ASD"].value_counts(dropna=False).to_dict()
        if "Class/ASD" in df.columns
        else {}
    )
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "target_counts": target_counts,
        "aq_columns": AQ_COLUMNS,
    }
