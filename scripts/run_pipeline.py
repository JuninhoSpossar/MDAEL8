"""Orquestrador do pipeline de mineração de dados."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import ensure_directories

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

STEPS = {
    "cleaning": "src.data_cleaning",
    "features": "src.feature_engineering",
    "eda": "src.eda",
    "unsupervised": "src.unsupervised",
    "classification": "src.classification",
}

STEP_ORDER = list(STEPS.keys())


def run_step(step_name: str, context: dict | None = None) -> object:
    if step_name not in STEPS:
        raise ValueError(f"Etapa desconhecida: {step_name}. Opções: {STEP_ORDER + ['all']}")

    module = importlib.import_module(STEPS[step_name])
    logger.info("Iniciando etapa: %s", step_name)
    start = time.time()

    if context and "df" in context and step_name in ("cleaning", "features"):
        result = module.run(context["df"])
    else:
        result = module.run()

    logger.info("Etapa '%s' concluída em %.1fs", step_name, time.time() - start)

    if step_name in ("cleaning", "features") and hasattr(result, "columns"):
        context["df"] = result
    return result


def run_pipeline(steps: list[str] | None = None) -> None:
    ensure_directories()
    if steps is None or "all" in steps:
        steps = STEP_ORDER

    context: dict = {}
    for step in steps:
        run_step(step, context)
    logger.info("Pipeline concluído com sucesso.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline — Adult Income Classification")
    parser.add_argument("--step", nargs="+", default=["all"], choices=STEP_ORDER + ["all"])
    args = parser.parse_args()
    run_pipeline(args.step)


if __name__ == "__main__":
    main()
