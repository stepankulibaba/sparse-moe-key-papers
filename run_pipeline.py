"""Воспроизвести анализ Sparse MoE без интернета.

Установка: python -m pip install -r requirements.txt
Полный расчёт: python -X utf8 run_pipeline.py
Быстрая проверка: python -X utf8 run_pipeline.py --skip-stress --no-figures
Входы: data/ и settings.json. Результаты: output/.
"""

import argparse

from src.corpus import build
from src.metrics import build_semantic_scores, build_structural_scores
from src.verify import run_checks


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip-stress", action="store_true", help="Пропустить 2 000 возмущений графа"
    )
    parser.add_argument("--no-figures", action="store_true", help="Не строить рисунки")
    args = parser.parse_args()
    print("1. Корпус, граф цитирования и базовые рейтинги", flush=True)
    build()
    print("2. Охват, разнообразие ветвей и текстовый пилот", flush=True)
    build_structural_scores()
    build_semantic_scores()
    if not args.skip_stress:
        from src.experiments import main as run_experiments

        print("3. Проверки устойчивости: 4 сценария по 500 запусков", flush=True)
        run_experiments()
    print("Проверка входов и независимые численные расчёты", flush=True)
    run_checks(include_stress=not args.skip_stress)
    if not args.no_figures:
        from src.figures import build_figures

        print("Построение рисунков", flush=True)
        build_figures(include_stress=not args.skip_stress)
    print("Готово. Таблицы, рисунки и записи проверок сохранены в output/.")


if __name__ == "__main__":
    main()
