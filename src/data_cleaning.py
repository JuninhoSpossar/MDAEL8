"""Limpeza e imputação de dados ausentes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import COLUMNS, COLUMNS_WITHOUT_FNLWGT, DATA_CLEAN, DATA_RAW, ensure_directories
from src.utils import show_dataframe_info, strip_whitespace


def impute_missing_values(df: pd.DataFrame, column: str, method: str = "median") -> None:
    """Imputa valores ausentes em uma coluna."""
    if method == "median":
        value = df[column].median()
    elif method == "mean":
        value = df[column].mean()
    elif method == "mode":
        value = df[column].mode()[0]
    else:
        raise ValueError(f"Método de imputação inválido: {method}")
    df[column] = df[column].fillna(value)


def load_raw_data() -> pd.DataFrame:
    """Carrega o dataset UCI Adult original."""
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DATA_RAW}")
    df = pd.read_csv(DATA_RAW, names=COLUMNS, na_values="?", skipinitialspace=True)
    return strip_whitespace(df)


def clean_data(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Remove fnlwgt e imputa valores ausentes (mediana/moda)."""
    if df is None:
        df = load_raw_data()

    df = df[COLUMNS_WITHOUT_FNLWGT].copy()
    show_dataframe_info(df, "DataFrame original")

    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            impute_missing_values(df, col, method="median")

    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            impute_missing_values(df, col, method="mode")

    print(f"Valores ausentes restantes: {df.isnull().sum().sum()}")
    return df


def run() -> pd.DataFrame:
    """Executa limpeza e salva resultado."""
    df = clean_data()
    ensure_directories()
    df.to_csv(DATA_CLEAN, index=False)
    print(f"Dataset limpo salvo em: {DATA_CLEAN}")
    return df
