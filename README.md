# Adult Income Classification

> Pipeline de mineração de dados para prever renda anual > US$ 50K com o dataset UCI Adult Census Income.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![pandas](https://img.shields.io/badge/pandas-2.0+-green)
![License](https://img.shields.io/badge/License-MIT-green)

Projeto de **Mineração de Dados** — classificação de renda anual superior a **US$ 50.000** com o dataset [UCI Adult](https://archive.ics.uci.edu/ml/datasets/adult).

## Estrutura

```
MDAEL8/
├── config/settings.py       # Colunas, caminhos e hiperparâmetros
├── data/raw/                # Dataset UCI original
├── src/
│   ├── data_cleaning.py     # Limpeza e imputação
│   ├── feature_engineering.py  # Encoding + normalização
│   ├── eda.py               # Análise exploratória
│   ├── unsupervised.py      # PCA + K-Means
│   ├── classification.py    # Benchmark de modelos
│   └── utils.py             # Utilitários compartilhados
├── scripts/run_pipeline.py  # Orquestrador
├── tests/test_pipeline.py
└── outputs/                 # Figuras e relatórios (gerados)
```

## Execução

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py --step all
python -m pytest tests/ -v
```

### Etapas disponíveis

| Etapa | Comando | Descrição |
|-------|---------|-----------|
| `cleaning` | `--step cleaning` | Limpeza e imputação |
| `features` | `--step features` | Encoding + Z-Score/Min-Max |
| `eda` | `--step eda` | Estatísticas e correlações |
| `unsupervised` | `--step unsupervised` | PCA e K-Means |
| `classification` | `--step classification` | Benchmark de 4 modelos |

## Resultados

| Modelo | Accuracy | F1 | ROC-AUC |
|--------|----------|-----|---------|
| **MLP** | **0.824** | **0.809** | **0.851** |
| Decision Tree | 0.808 | 0.804 | 0.790 |
| SVM | 0.804 | 0.773 | 0.816 |
| KNN | 0.801 | 0.794 | 0.785 |

## Metodologia

```
adult.data → limpeza → feature engineering → EDA
                                        → PCA / K-Means
                                        → classificação (DT, KNN, SVM, MLP)
```

## Tecnologias

Python, pandas, numpy, scikit-learn, matplotlib, seaborn, pytest.

## Licença

[MIT](LICENSE) — consulte [CONTRIBUTING.md](CONTRIBUTING.md) para contribuir.
