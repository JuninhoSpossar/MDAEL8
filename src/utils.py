"""Funções utilitárias compartilhadas."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import OUTPUT_FIGURES, ensure_directories


def show_dataframe_info(df: pd.DataFrame, message: str = "") -> None:
    if message:
        print(f"\n{message}")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print()


def save_figure(fig: plt.Figure, filename: str) -> Path:
    ensure_directories()
    path = OUTPUT_FIGURES / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {path}")
    return path


def plot_confusion_matrix(
    cm: np.ndarray, classes: list, title: str = "Matriz de Confusão",
    normalize: bool = False, save_as: str | None = None,
) -> None:
    display_cm = cm.astype("float")
    fmt = ".2f" if normalize else ".0f"
    if normalize:
        display_cm = display_cm / display_cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(display_cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Rótulo Verdadeiro")
    ax.set_xlabel("Rótulo Predito")
    plt.tight_layout()
    save_figure(fig, save_as) if save_as else plt.show()


def plot_pca_projection(
    df: pd.DataFrame, pc1_col: str, pc2_col: str, label_col: str,
    title: str = "Projeção PCA 2D", save_as: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = df[label_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for label, color in zip(labels, colors):
        mask = df[label_col] == label
        ax.scatter(df.loc[mask, pc1_col], df.loc[mask, pc2_col],
                   c=[color], label=str(label), alpha=0.6, edgecolors="none")
    ax.set_xlabel(pc1_col)
    ax.set_ylabel(pc2_col)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, save_as) if save_as else plt.show()


def plot_pca_clusters(
    principal_df: pd.DataFrame, cluster_labels_list: list[np.ndarray],
    k_values: range, save_as: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = ["blue", "green", "red", "purple", "orange", "cyan"]
    for i, (k, labels) in enumerate(zip(k_values, cluster_labels_list)):
        ax = axes.flatten()[i]
        for cluster, color in zip(range(k), colors):
            mask = labels == cluster
            ax.scatter(principal_df.loc[mask, "PC1"], principal_df.loc[mask, "PC2"],
                       c=color, label=f"Cluster {cluster}", alpha=0.6, edgecolors="none")
        ax.set_title(f"K-Means k={k}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
    plt.tight_layout()
    save_figure(fig, save_as) if save_as else plt.show()


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df
