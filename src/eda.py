"""Análise exploratória de dados."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.settings import DATA_ENCODED, EDA_REPORT, NUMERIC_FEATURES, TARGET, ensure_directories
from src.utils import save_figure, show_dataframe_info


def compute_descriptive_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            stats[col] = {
                "media": float(df[col].mean()),
                "mediana": float(df[col].median()),
                "desvio_padrao": float(df[col].std()),
                "minimo": float(df[col].min()),
                "maximo": float(df[col].max()),
            }
    return stats


def compute_income_correlations(df: pd.DataFrame) -> dict:
    cols = [c for c in NUMERIC_FEATURES if c in df.columns] + [TARGET]
    corr = df[cols].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    return {col: float(val) for col, val in corr.items()}


def _plot_income_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[TARGET].value_counts().sort_index()
    counts.index = ["<=50K", ">50K"]
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Distribuição da Variável Alvo")
    ax.set_xlabel("Classe de Renda")
    ax.set_ylabel("Frequência")
    plt.tight_layout()
    save_figure(fig, "income_distribution.png")


def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = [c for c in NUMERIC_FEATURES if c in df.columns] + [TARGET]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de Correlação")
    plt.tight_layout()
    save_figure(fig, "correlation_heatmap.png")


def _plot_income_correlations(df: pd.DataFrame) -> None:
    correlations = compute_income_correlations(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(correlations).plot(kind="barh", ax=ax, color="#55A868")
    ax.set_title("Correlação das Features com Income")
    ax.set_xlabel("Correlação de Pearson")
    plt.tight_layout()
    save_figure(fig, "income_correlations.png")


def run(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        if not DATA_ENCODED.exists():
            raise FileNotFoundError(f"Execute feature engineering primeiro: {DATA_ENCODED}")
        df = pd.read_csv(DATA_ENCODED)

    show_dataframe_info(df, "DataFrame para EDA")

    summary = {
        "descritivas": compute_descriptive_stats(df),
        "correlacoes_income": compute_income_correlations(df),
        "distribuicao_alvo": df[TARGET].value_counts().to_dict(),
    }

    _plot_income_distribution(df)
    _plot_correlation_heatmap(df)
    _plot_income_correlations(df)

    ensure_directories()
    with open(EDA_REPORT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Resumo EDA salvo em: {EDA_REPORT}")
    return summary
