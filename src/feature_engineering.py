"""Feature engineering, codificação e normalização."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from config.settings import (
    CATEGORICAL_COLUMNS,
    DATA_CLEAN,
    DATA_ENCODED,
    DATA_MINMAX,
    DATA_ZSCORE,
    EDUCATION_LEVELS,
    NUMERIC_FEATURES,
    OUTPUT_MODELS,
    TARGET,
    TARGET_NEGATIVE,
    TARGET_POSITIVE,
    ensure_directories,
)
from src.utils import show_dataframe_info


def categorize_education(education: str) -> str:
    if education in EDUCATION_LEVELS["fundamental"]:
        return "Fundamental"
    if education in EDUCATION_LEVELS["medio"]:
        return "Medio"
    if education in EDUCATION_LEVELS["superior"]:
        return "Superior"
    return "Outro"


def encode_target(income: pd.Series) -> pd.Series:
    return income.map({TARGET_NEGATIVE: 0, TARGET_POSITIVE: 1})


def encode_categorical_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders: dict[str, dict] = {}
    cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns and c != "Sex"]

    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = {
            "classes": le.classes_.tolist(),
            "mapping": {cls: int(i) for i, cls in enumerate(le.classes_)},
        }
    return df, encoders


def encode_sex_onehot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sex"] = df["Sex"].str.strip()
    dummies = pd.get_dummies(df["Sex"], prefix="Sex", drop_first=True)
    return pd.concat([df.drop(columns=["Sex"]), dummies], axis=1)


def normalize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica Z-Score e Min-Max nas features numéricas."""
    features = [c for c in NUMERIC_FEATURES if c in df.columns]
    x = df[features].values
    target = df[TARGET]

    df_zscore = pd.DataFrame(StandardScaler().fit_transform(x), columns=features)
    df_zscore[TARGET] = target.values

    df_minmax = pd.DataFrame(MinMaxScaler().fit_transform(x), columns=features)
    df_minmax[TARGET] = target.values

    return df_zscore, df_minmax


def engineer_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        if not DATA_CLEAN.exists():
            raise FileNotFoundError(f"Execute a etapa de limpeza primeiro: {DATA_CLEAN}")
        df = pd.read_csv(DATA_CLEAN)

    df = df.copy()
    df["Education-Level"] = df["Education"].apply(categorize_education)
    df = encode_sex_onehot(df)

    target = encode_target(df[TARGET])
    df = df.drop(columns=[TARGET])
    df[TARGET] = target
    df, encoders = encode_categorical_columns(df)

    ensure_directories()
    with open(OUTPUT_MODELS / "label_encoders.json", "w", encoding="utf-8") as f:
        json.dump(encoders, f, indent=2, ensure_ascii=False)

    show_dataframe_info(df, "DataFrame codificado")
    return df


def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Executa feature engineering, normalização e salva artefatos."""
    result = engineer_features(df)

    ensure_directories()
    result.to_csv(DATA_ENCODED, index=False)
    print(f"Dataset codificado salvo em: {DATA_ENCODED}")

    df_zscore, df_minmax = normalize_features(result)
    df_zscore.to_csv(DATA_ZSCORE, index=False)
    df_minmax.to_csv(DATA_MINMAX, index=False)
    print(f"Z-Score salvo em: {DATA_ZSCORE}")
    print(f"Min-Max salvo em: {DATA_MINMAX}")

    return result
