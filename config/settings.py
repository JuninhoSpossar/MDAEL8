"""Configurações centralizadas: colunas, caminhos e hiperparâmetros."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Dataset ---
COLUMNS = [
    "Age", "Workclass", "fnlwgt", "Education", "Education-Num",
    "Marital-Status", "Occupation", "Relationship", "Race", "Sex",
    "Capital-Gain", "Capital-Loss", "Hours-per-week", "Native-Country", "Income",
]
COLUMNS_WITHOUT_FNLWGT = [c for c in COLUMNS if c != "fnlwgt"]

CATEGORICAL_COLUMNS = [
    "Workclass", "Education", "Marital-Status", "Occupation",
    "Relationship", "Race", "Sex", "Native-Country",
]

NUMERIC_FEATURES = [
    "Age", "Education-Num", "Capital-Gain", "Capital-Loss", "Hours-per-week",
]

TARGET = "Income"
TARGET_POSITIVE = ">50K"
TARGET_NEGATIVE = "<=50K"

RANDOM_STATE = 42
TEST_SIZE = 0.3
CV_FOLDS = 10

EDUCATION_LEVELS = {
    "fundamental": ["5th-6th", "1st-4th", "7th-8th", "Preschool"],
    "medio": ["9th", "10th", "11th", "12th"],
    "superior": [
        "Bachelors", "Some-college", "HS-grad", "Prof-school",
        "Assoc-acdm", "Assoc-voc", "Masters", "Doctorate",
    ],
}

# --- Caminhos ---
DATA_DIR = ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

DATA_RAW = DATA_RAW_DIR / "adult.data"
DATA_CLEAN = DATA_PROCESSED_DIR / "adult_clean.csv"
DATA_ENCODED = DATA_PROCESSED_DIR / "adult_encoded.csv"
DATA_ZSCORE = DATA_PROCESSED_DIR / "adult_zscore.csv"
DATA_MINMAX = DATA_PROCESSED_DIR / "adult_minmax.csv"

OUTPUTS_DIR = ROOT / "outputs"
OUTPUT_FIGURES = OUTPUTS_DIR / "figures"
OUTPUT_MODELS = OUTPUTS_DIR / "models"
OUTPUT_REPORTS = OUTPUTS_DIR / "reports"

BENCHMARK_REPORT = OUTPUT_REPORTS / "model_benchmark.json"
CLUSTERING_REPORT = OUTPUT_REPORTS / "clustering_metrics.json"
EDA_REPORT = OUTPUT_REPORTS / "eda_summary.json"


def ensure_directories() -> None:
    """Cria diretórios necessários se não existirem."""
    for path in (DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUT_FIGURES, OUTPUT_MODELS, OUTPUT_REPORTS):
        path.mkdir(parents=True, exist_ok=True)
