"""Model comparison with statistical significance tests and registry promotion."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from src.models.metrics import to_dollar_scale
from src.models.registry import register_best_model
from src.models.train import (
    PROCESSED_DIR,
    TARGET_COLUMN,
    TRAINING_SUMMARY_PATH,
    _build_estimator,
    build_experiment_grid,
)
from src.utils.config import load_params

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

SIGNIFICANCE_PATH = REPORTS_DIR / "significance_tests.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.md"
EVALUATION_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"


def _load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")[TARGET_COLUMN]
    return X_train, y_train


def _load_training_summary() -> dict[str, Any]:
    if not TRAINING_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Training summary not found at {TRAINING_SUMMARY_PATH}. Run train stage first."
        )
    return json.loads(TRAINING_SUMMARY_PATH.read_text(encoding="utf-8"))


def _dollar_mae_scorer(log_scale: bool):
    def scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(
            to_dollar_scale(y_true, log_scale=log_scale),
            to_dollar_scale(y_pred, log_scale=log_scale),
        )

    return make_scorer(scorer, greater_is_better=False)


def _get_experiment_by_run_name(run_name: str, params: dict[str, Any]) -> dict[str, Any]:
    for experiment in build_experiment_grid(params):
        if experiment["run_name"] == run_name:
            return experiment
    raise ValueError(f"Experiment config not found for run: {run_name}")


def _fit_estimator(experiment: dict[str, Any]):
    return _build_estimator(experiment["algorithm"], experiment.get("hyperparams", {}))


def _cross_validated_fold_maes(
    experiment: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
) -> list[float]:
    estimator = _fit_estimator(experiment)
    pipeline = Pipeline([("model", estimator)])
    log_scale = experiment["log_target"]

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring=_dollar_mae_scorer(log_scale),
        n_jobs=-1,
    )
    return (-cv_results["test_score"]).tolist()


def _select_top_models(summary: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
    excluded = {"baseline_naive", "baseline_engineered"}
    candidates = [
        run for run in summary["runs"] if run["algorithm"] not in excluded
    ]
    candidates.sort(key=lambda item: item["metrics"]["mae"])
    return candidates[:top_n]


def _run_significance_tests(
    fold_maes_by_run: dict[str, list[float]],
    alpha: float,
) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []

    for model_a, model_b in combinations(fold_maes_by_run.keys(), 2):
        scores_a = fold_maes_by_run[model_a]
        scores_b = fold_maes_by_run[model_b]
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        mean_diff = float(np.mean(scores_a) - np.mean(scores_b))
        tests.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "mean_mae_a": float(np.mean(scores_a)),
                "mean_mae_b": float(np.mean(scores_b)),
                "mean_difference": mean_diff,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_at_alpha": bool(p_value < alpha),
                "interpretation": (
                    f"Diferença estatisticamente significativa (p={p_value:.4f} < {alpha})"
                    if p_value < alpha
                    else f"Diferença não significativa (p={p_value:.4f} >= {alpha})"
                ),
            }
        )

    return tests


def _render_comparison_markdown(
    top_models: list[dict[str, Any]],
    significance_tests: list[dict[str, Any]],
    registry_info: dict[str, Any],
    alpha: float,
) -> str:
    lines = [
        "# Model Comparison — 6 Algorithms + Significance Tests",
        "",
        "Comparação dos melhores modelos com teste t pareado sobre MAE de cross-validation.",
        "",
        "## Top modelos (holdout)",
        "",
        "| Rank | Run | Algoritmo | MAE ($) | RMSE ($) | R² | CV MAE ($) |",
        "|------|-----|-----------|---------|----------|-----|------------|",
    ]

    for index, run in enumerate(top_models, start=1):
        metrics = run["metrics"]
        lines.append(
            f"| {index} | `{run['run_name']}` | {run['algorithm']} | "
            f"${metrics['mae']:,.0f} | ${metrics['rmse']:,.0f} | "
            f"{metrics['r2']:.4f} | ${metrics['cv_mae']:,.0f} |"
        )

    lines.extend(
        [
            "",
            f"## Testes de significância (paired t-test, α={alpha})",
            "",
            "| Modelo A | Modelo B | Δ MAE ($) | p-value | Significativo? |",
            "|----------|----------|-----------|---------|----------------|",
        ]
    )

    for test in significance_tests:
        significant = "Sim" if test["significant_at_alpha"] else "Não"
        lines.append(
            f"| `{test['model_a']}` | `{test['model_b']}` | "
            f"${test['mean_difference']:,.0f} | {test['p_value']:.4f} | {significant} |"
        )

    lines.extend(
        [
            "",
            "## Model Registry",
            "",
            f"- **Modelo registrado:** `{registry_info['registered_model_name']}`",
            f"- **Versão:** {registry_info['model_version']}",
            f"- **Stage:** {registry_info['stage']}",
            f"- **Run de origem:** `{registry_info['source_run_name']}`",
            f"- **MAE:** ${registry_info['metrics']['mae']:,.0f}",
            "",
            "## Conclusão",
            "",
            _build_conclusion(top_models, significance_tests, registry_info),
            "",
        ]
    )

    return "\n".join(lines)


def _build_conclusion(
    top_models: list[dict[str, Any]],
    significance_tests: list[dict[str, Any]],
    registry_info: dict[str, Any],
) -> str:
    winner = top_models[0]["run_name"]
    runner_up = top_models[1]["run_name"] if len(top_models) > 1 else None

    comparison = next(
        (test for test in significance_tests if {test["model_a"], test["model_b"]} == {winner, runner_up}),
        None,
    )

    if comparison and not comparison["significant_at_alpha"]:
        return (
            f"O modelo `{winner}` tem menor MAE no holdout, mas a diferença para "
            f"`{runner_up}` **não é estatisticamente significativa** no CV pareado. "
            f"Ambos são candidatos sólidos; `{registry_info['source_run_name']}` foi "
            f"promovido por critério de menor MAE no holdout."
        )

    return (
        f"O modelo `{registry_info['source_run_name']}` apresentou o menor MAE no holdout "
        f"(${registry_info['metrics']['mae']:,.0f}) e foi promovido para **Production** "
        f"no Model Registry como `{registry_info['registered_model_name']}`."
    )


def run_evaluation(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or load_params()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    evaluation = params.get("evaluation", {})
    top_n = int(evaluation.get("top_n_models", 3))
    alpha = float(evaluation.get("significance_level", 0.05))
    cv_folds = int(params["training"]["cv_folds"])

    summary = _load_training_summary()
    top_models = _select_top_models(summary, top_n)

    X_train, y_train = _load_training_data()

    fold_maes_by_run: dict[str, list[float]] = {}
    for run in top_models:
        experiment = _get_experiment_by_run_name(run["run_name"], params)
        fold_maes_by_run[run["run_name"]] = _cross_validated_fold_maes(
            experiment,
            X_train,
            y_train,
            cv_folds,
        )

    significance_tests = _run_significance_tests(fold_maes_by_run, alpha)
    registry_info = register_best_model(params)

    evaluation_payload = {
        "top_models": top_models,
        "fold_maes_by_run": fold_maes_by_run,
        "significance_tests": significance_tests,
        "significance_level": alpha,
        "registry": registry_info,
    }

    SIGNIFICANCE_PATH.write_text(
        json.dumps(
            {
                "significance_level": alpha,
                "fold_maes_by_run": fold_maes_by_run,
                "tests": significance_tests,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    EVALUATION_METRICS_PATH.write_text(json.dumps(evaluation_payload, indent=2), encoding="utf-8")
    COMPARISON_PATH.write_text(
        _render_comparison_markdown(top_models, significance_tests, registry_info, alpha),
        encoding="utf-8",
    )

    print(f"Compared top {len(top_models)} models with {len(significance_tests)} significance tests")
    print(f"Saved {SIGNIFICANCE_PATH}")
    print(f"Saved {COMPARISON_PATH}")
    print(f"Saved {EVALUATION_METRICS_PATH}")

    return evaluation_payload


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
