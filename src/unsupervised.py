"""Análise não supervisionada: PCA e K-Means."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    completeness_score,
    homogeneity_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler

from config.settings import (
    CLUSTERING_REPORT,
    DATA_ENCODED,
    NUMERIC_FEATURES,
    OUTPUT_REPORTS,
    RANDOM_STATE,
    TARGET,
    ensure_directories,
)
from src.utils import plot_pca_clusters, plot_pca_projection, show_dataframe_info


def _load_features(df: pd.DataFrame | None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    if df is None:
        if not DATA_ENCODED.exists():
            raise FileNotFoundError(f"Execute feature engineering primeiro: {DATA_ENCODED}")
        df = pd.read_csv(DATA_ENCODED)

    features = [c for c in NUMERIC_FEATURES if c in df.columns]
    x = df[features].values
    y = df[TARGET].values
    x_scaled = StandardScaler().fit_transform(x)
    return df, x_scaled, y, features


def run_pca(df: pd.DataFrame | None = None, n_components: int = 2) -> dict:
    """Executa PCA e salva projeção 2D."""
    _, x_scaled, y, features = _load_features(df)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(x_scaled)
    pc_cols = [f"PC{i + 1}" for i in range(n_components)]
    result_df = pd.DataFrame(components, columns=pc_cols)
    result_df[TARGET] = y

    metrics = {
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": float(np.cumsum(pca.explained_variance_ratio_)[-1]),
        "features_used": features,
    }

    if n_components == 2:
        plot_pca_projection(result_df, "PC1", "PC2", TARGET, "PCA 2D — Adult Income", "pca_2d.png")

    show_dataframe_info(result_df, f"Projeção PCA ({n_components}D)")
    return metrics


def run_clustering(df: pd.DataFrame | None = None) -> dict:
    """Executa K-Means (k=2..6) com métricas de qualidade."""
    _, x_scaled, y, features = _load_features(df)
    show_dataframe_info(pd.DataFrame(x_scaled, columns=features), "Features para clustering")

    pca = PCA(n_components=2)
    principal_df = pd.DataFrame(pca.fit_transform(x_scaled), columns=["PC1", "PC2"])

    k_values = range(2, 7)
    cluster_labels_list = []
    metrics = {"k_values": [], "silhouette": [], "homogeneity": [], "completeness": [], "v_measure": []}

    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(x_scaled)
        cluster_labels_list.append(labels)
        metrics["k_values"].append(k)
        metrics["silhouette"].append(float(silhouette_score(x_scaled, labels)))
        metrics["homogeneity"].append(float(homogeneity_score(y, labels)))
        metrics["completeness"].append(float(completeness_score(y, labels)))
        metrics["v_measure"].append(float(v_measure_score(y, labels)))
        print(f"k={k}: silhouette={metrics['silhouette'][-1]:.4f}, v_measure={metrics['v_measure'][-1]:.4f}")

    plot_pca_clusters(principal_df, cluster_labels_list, k_values, "kmeans_pca_clusters.png")

    ensure_directories()
    with open(CLUSTERING_REPORT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Métricas de clustering salvas em: {CLUSTERING_REPORT}")
    return metrics


def run(df: pd.DataFrame | None = None) -> dict:
    """Executa PCA (2D e 3D) e clustering."""
    pca_2d = run_pca(df, n_components=2)
    pca_3d = run_pca(df, n_components=3)
    clustering = run_clustering(df)

    report = {"pca_2d": pca_2d, "pca_3d": pca_3d, "clustering": clustering}
    ensure_directories()
    report_path = OUTPUT_REPORTS / "unsupervised_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Relatório não supervisionado salvo em: {report_path}")
    return report
