"""Testes de smoke do pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_CLEAN, DATA_ENCODED, DATA_RAW, NUMERIC_FEATURES, TARGET, ensure_directories
from src import classification, data_cleaning, feature_engineering


@pytest.fixture(scope="module")
def setup_dirs():
    ensure_directories()
    yield


def test_raw_data_exists(setup_dirs):
    assert DATA_RAW.exists()


def test_load_raw_data(setup_dirs):
    df = data_cleaning.load_raw_data()
    assert len(df) > 0 and "Income" in df.columns


def test_clean_data_removes_missing(setup_dirs):
    df = data_cleaning.clean_data()
    assert df.isnull().sum().sum() == 0
    assert "fnlwgt" not in df.columns


def test_feature_engineering(setup_dirs):
    clean_df = data_cleaning.clean_data()
    encoded_df = feature_engineering.engineer_features(clean_df)
    assert encoded_df[TARGET].isin([0, 1]).all()
    assert len([c for c in NUMERIC_FEATURES if c in encoded_df.columns]) == len(NUMERIC_FEATURES)


def test_model_trains_above_baseline(setup_dirs):
    clean_df = data_cleaning.clean_data()
    encoded_df = feature_engineering.engineer_features(clean_df)
    sample = encoded_df.sample(n=2000, random_state=42)
    benchmark = classification.run_benchmark(sample)
    assert len(benchmark["modelos"]) == 4
    for result in benchmark["modelos"]:
        assert result["accuracy"] > 0.5


def test_pipeline_integration(setup_dirs):
    data_cleaning.run()
    feature_engineering.run()
    encoded = pd.read_csv(DATA_ENCODED)
    assert encoded.shape[0] > 30000
    assert DATA_CLEAN.exists() and DATA_ENCODED.exists()
