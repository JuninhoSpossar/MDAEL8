# Contribuindo

## Setup

```bash
git clone <repo-url>
cd MDAEL8
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Executar

```bash
python scripts/run_pipeline.py --step all
python -m pytest tests/ -v
```

## Estrutura

- `config/settings.py` — configuração única
- `src/` — módulos do pipeline (um arquivo por responsabilidade)
- `scripts/run_pipeline.py` — entry point
- `data/raw/` — dados UCI (não alterar)
- `data/processed/` e `outputs/` — gerados automaticamente

## Commits

Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`

## Adicionar modelo

Edite `MODELS` em `src/classification.py`, execute `--step classification` e atualize o README.
