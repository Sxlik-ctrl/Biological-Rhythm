"""
PHASE 8 - FINAL INTEGRATION & PROJECT RESULTS

Automated Biological Rhythm Analysis & Health State Classification

Purpose:
    Consolidate and validate the outputs of Phases 2-7
    into a final reproducible project-results package.
"""

from pathlib import Path
import json
import warnings
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_ROOT = PROJECT_ROOT / "results"

PHASE3_DIR = RESULTS_ROOT / "phase3"
PHASE4_DIR = RESULTS_ROOT / "phase4"
PHASE5_DIR = RESULTS_ROOT / "phase5"
PHASE6_DIR = RESULTS_ROOT / "phase6_final"
PHASE7_DIR = RESULTS_ROOT / "phase7"

OUTPUT_DIR = RESULTS_ROOT / "phase8"
PLOTS_DIR = OUTPUT_DIR / "plots"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# SAFE CSV LOADER
# ============================================================

def load_csv(path):

    if not path.exists():

        print(
            f"WARNING: File not found: {path}"
        )

        return None

    return pd.read_csv(path)


# ============================================================
# PHASE 3 - ACTIVITY
# ============================================================

def integrate_activity():

    path = (
        PHASE3_DIR
        / "activity_features.csv"
    )

    df = load_csv(path)

    if df is None:
        return None

    numeric_columns = [
        "mean_activity",
        "median_activity",
        "activity_std",
        "activity_min",
        "activity_max",
        "daytime_mean_activity",
        "nighttime_mean_activity",
        "day_night_ratio",
        "activity_cv"
    ]

    available = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    summary = (
        df.groupby("label")[available]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        OUTPUT_DIR
        / "final_activity_summary.csv"
    )

    return df


# ============================================================
# PHASE 3 - SLEEP
# ============================================================

def integrate_sleep():

    path = (
        PHASE3_DIR
        / "sleep_features.csv"
    )

    df = load_csv(path)

    if df is None:
        return None

    numeric_columns = [
        "estimated_sleep_hours",
        "estimated_wake_hours",
        "estimated_sleep_periods",
        "mean_night_activity",
        "sleep_activity_threshold",
        "sleep_regularity"
    ]

    available = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    summary = (
        df.groupby("label")[available]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        OUTPUT_DIR
        / "final_sleep_summary.csv"
    )

    return df


# ============================================================
# PHASE 4 - COSINOR
# ============================================================

def integrate_cosinor():

    path = (
        PHASE4_DIR
        / "cosinor_subject_features.csv"
    )

    df = load_csv(path)

    if df is None:
        return None

    numeric_columns = [
        "mesor",
        "amplitude",
        "acrophase_hours",
        "trough_hours",
        "fitted_max",
        "fitted_min",
        "r_squared",
        "rmse",
        "relative_amplitude"
    ]

    available = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    summary = (
        df.groupby("label")[available]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        OUTPUT_DIR
        / "final_cosinor_summary.csv"
    )

    return df


# ============================================================
# PHASE 5 - STATISTICS
# ============================================================

def integrate_statistics():

    path = (
        PHASE5_DIR
        / "statistical_comparison.csv"
    )

    df = load_csv(path)

    if df is None:
        return None

    # IMPORTANT:
    # The actual Phase 5 column is "fdr_p_value",
    # not "fdr".

    selected_columns = [
        column
        for column in [
            "feature",
            "condition_mean",
            "control_mean",
            "condition_median",
            "control_median",
            "p_value",
            "fdr_p_value",
            "cohens_d",
            "rank_biserial",
            "significant_fdr",
            "effect_size_notable"
        ]
        if column in df.columns
    ]

    final = df[
        selected_columns
    ].copy()

    final.to_csv(
        OUTPUT_DIR
        / "final_statistical_summary.csv",
        index=False
    )

    return final


# ============================================================
# PHASE 6 - MACHINE LEARNING
# ============================================================

def integrate_ml():

    path = (
        PHASE6_DIR
        / "out_of_fold_results.csv"
    )

    df = load_csv(path)

    if df is None:
        return None

    df.to_csv(
        OUTPUT_DIR
        / "final_ml_summary.csv",
        index=False
    )

    return df


# ============================================================
# PHASE 7 - FEATURE IMPORTANCE
# ============================================================

def integrate_feature_importance():

    importance_path = (
        PHASE7_DIR
        / "permutation_importance.csv"
    )

    group_path = (
        PHASE7_DIR
        / "feature_group_importance.csv"
    )

    importance = load_csv(
        importance_path
    )

    groups = load_csv(
        group_path
    )

    if importance is not None:

        importance.to_csv(
            OUTPUT_DIR
            / "final_feature_importance.csv",
            index=False
        )

    if groups is not None:

        groups.to_csv(
            OUTPUT_DIR
            / "final_feature_group_importance.csv",
            index=False
        )

    return (
        importance,
        groups
    )


# ============================================================
# BUILD PROJECT SUMMARY
# ============================================================

def build_project_summary(
    activity,
    sleep,
    cosinor,
    statistics,
    ml,
    importance,
    groups
):

    summary = {}

    # --------------------------------------------------------
    # Subjects
    # --------------------------------------------------------

    if activity is not None:

        summary["subjects"] = int(
            activity["subject_id"].nunique()
        )

        summary["condition_subjects"] = int(
            (
                activity["label"]
                == "condition"
            ).sum()
        )

        summary["control_subjects"] = int(
            (
                activity["label"]
                == "control"
            ).sum()
        )

    # --------------------------------------------------------
    # Phase 3
    # --------------------------------------------------------

    if activity is not None:

        summary[
            "activity_feature_rows"
        ] = int(len(activity))

    if sleep is not None:

        summary[
            "sleep_feature_rows"
        ] = int(len(sleep))

    # --------------------------------------------------------
    # Phase 4
    # --------------------------------------------------------

    if cosinor is not None:

        summary[
            "cosinor_subjects"
        ] = int(len(cosinor))

        summary[
            "cosinor_successful_subjects"
        ] = int(
            cosinor[
                "r_squared"
            ].notna().sum()
        )

    # --------------------------------------------------------
    # Phase 5
    # --------------------------------------------------------

    if statistics is not None:

        summary[
            "statistical_features"
        ] = int(len(statistics))

        # Correct column name:
        # fdr_p_value

        if "fdr_p_value" in statistics.columns:

            summary[
                "fdr_significant_features"
            ] = int(
                (
                    statistics[
                        "fdr_p_value"
                    ] < 0.05
                ).sum()
            )

    # --------------------------------------------------------
    # Phase 6
    # --------------------------------------------------------

    if ml is not None:

        summary[
            "ml_models"
        ] = int(len(ml))

        if "roc_auc" in ml.columns:

            gradient = ml[
                ml["model"]
                == "Gradient Boosting"
            ]

            if len(gradient) > 0:

                summary[
                    "gradient_boosting_roc_auc"
                ] = float(
                    gradient.iloc[0]["roc_auc"]
                )

                summary[
                    "gradient_boosting_accuracy"
                ] = float(
                    gradient.iloc[0]["accuracy"]
                )

                summary[
                    "gradient_boosting_f1"
                ] = float(
                    gradient.iloc[0]["f1"]
                )

    # --------------------------------------------------------
    # Phase 7
    # --------------------------------------------------------

    if importance is not None:

        if len(importance) > 0:

            top_feature = importance.iloc[0]

            summary[
                "top_explainability_feature"
            ] = str(
                top_feature["feature"]
            )

            summary[
                "top_feature_importance"
            ] = float(
                top_feature[
                    "importance_mean"
                ]
            )

    if groups is not None:

        if len(groups) > 0:

            top_group = groups.sort_values(
                "mean_absolute_importance",
                ascending=False
            ).iloc[0]

            summary[
                "highest_mean_importance_group"
            ] = str(
                top_group["feature_group"]
            )

    return summary


# ============================================================
# VALIDATION CHECKS
# ============================================================

def run_validation_checks(
    summary,
    activity,
    sleep,
    cosinor,
    ml,
    importance
):

    checks = []

    # --------------------------------------------------------
    # Subject checks
    # --------------------------------------------------------

    checks.append(
        {
            "check":
                "Total subjects = 55",
            "expected":
                55,
            "observed":
                summary.get("subjects"),
            "status":
                (
                    "PASS"
                    if summary.get("subjects") == 55
                    else "CHECK"
                )
        }
    )

    checks.append(
        {
            "check":
                "Condition subjects = 23",
            "expected":
                23,
            "observed":
                summary.get("condition_subjects"),
            "status":
                (
                    "PASS"
                    if summary.get(
                        "condition_subjects"
                    ) == 23
                    else "CHECK"
                )
        }
    )

    checks.append(
        {
            "check":
                "Control subjects = 32",
            "expected":
                32,
            "observed":
                summary.get("control_subjects"),
            "status":
                (
                    "PASS"
                    if summary.get(
                        "control_subjects"
                    ) == 32
                    else "CHECK"
                )
        }
    )

    # --------------------------------------------------------
    # Phase 3
    # --------------------------------------------------------

    checks.append(
        {
            "check":
                "Activity feature rows = 55",
            "expected":
                55,
            "observed":
                summary.get(
                    "activity_feature_rows"
                ),
            "status":
                (
                    "PASS"
                    if summary.get(
                        "activity_feature_rows"
                    ) == 55
                    else "CHECK"
                )
        }
    )

    checks.append(
        {
            "check":
                "Sleep feature rows = 55",
            "expected":
                55,
            "observed":
                summary.get(
                    "sleep_feature_rows"
                ),
            "status":
                (
                    "PASS"
                    if summary.get(
                        "sleep_feature_rows"
                    ) == 55
                    else "CHECK"
                )
        }
    )

    # --------------------------------------------------------
    # Phase 4
    # --------------------------------------------------------

    checks.append(
        {
            "check":
                "Cosinor subjects = 55",
            "expected":
                55,
            "observed":
                summary.get(
                    "cosinor_subjects"
                ),
            "status":
                (
                    "PASS"
                    if summary.get(
                        "cosinor_subjects"
                    ) == 55
                    else "CHECK"
                )
        }
    )

    # --------------------------------------------------------
    # Phase 5
    # --------------------------------------------------------

    checks.append(
        {
            "check":
                "FDR-significant features = 7",
            "expected":
                7,
            "observed":
                summary.get(
                    "fdr_significant_features"
                ),
            "status":
                (
                    "PASS"
                    if summary.get(
                        "fdr_significant_features"
                    ) == 7
                    else "CHECK"
                )
        }
    )

    # --------------------------------------------------------
    # Phase 6
    # --------------------------------------------------------

    if ml is not None:

        gradient = ml[
            ml["model"]
            == "Gradient Boosting"
        ]

        if len(gradient) > 0:

            auc = float(
                gradient.iloc[0]["roc_auc"]
            )

            accuracy = float(
                gradient.iloc[0]["accuracy"]
            )

            checks.append(
                {
                    "check":
                        "Gradient Boosting ROC-AUC",
                    "expected":
                        "Approximately 0.895",
                    "observed":
                        auc,
                    "status":
                        (
                            "PASS"
                            if abs(
                                auc - 0.89538
                            ) < 0.01
                            else "CHECK"
                        )
                }
            )

            checks.append(
                {
                    "check":
                        "Gradient Boosting accuracy",
                    "expected":
                        "Approximately 0.818",
                    "observed":
                        accuracy,
                    "status":
                        (
                            "PASS"
                            if abs(
                                accuracy - 0.81818
                            ) < 0.01
                            else "CHECK"
                        )
                }
            )

    # --------------------------------------------------------
    # Phase 7
    # --------------------------------------------------------

    if importance is not None:

        top_feature = (
            importance.iloc[0]["feature"]
            if len(importance) > 0
            else None
        )

        checks.append(
            {
                "check":
                    "Top explainability feature",
                "expected":
                    "estimated_sleep_hours",
                "observed":
                    top_feature,
                "status":
                    (
                        "PASS"
                        if top_feature
                        == "estimated_sleep_hours"
                        else "CHECK"
                    )
            }
        )

    return pd.DataFrame(checks)


# ============================================================
# FINAL PROJECT SUMMARY TABLE
# ============================================================

def create_final_summary_table(
    summary
):

    rows = [

        [
            "Dataset subjects",
            summary.get("subjects")
        ],

        [
            "Condition subjects",
            summary.get("condition_subjects")
        ],

        [
            "Control subjects",
            summary.get("control_subjects")
        ],

        [
            "Activity feature rows",
            summary.get(
                "activity_feature_rows"
            )
        ],

        [
            "Sleep feature rows",
            summary.get(
                "sleep_feature_rows"
            )
        ],

        [
            "Cosinor subjects",
            summary.get(
                "cosinor_subjects"
            )
        ],

        [
            "FDR-significant features",
            summary.get(
                "fdr_significant_features"
            )
        ],

        [
            "Gradient Boosting accuracy",
            summary.get(
                "gradient_boosting_accuracy"
            )
        ],

        [
            "Gradient Boosting F1",
            summary.get(
                "gradient_boosting_f1"
            )
        ],

        [
            "Gradient Boosting ROC-AUC",
            summary.get(
                "gradient_boosting_roc_auc"
            )
        ],

        [
            "Top explainability feature",
            summary.get(
                "top_explainability_feature"
            )
        ],

        [
            "Highest feature group",
            summary.get(
                "highest_mean_importance_group"
            )
        ]
    ]

    df = pd.DataFrame(
        rows,
        columns=[
            "metric",
            "value"
        ]
    )

    df.to_csv(
        OUTPUT_DIR
        / "final_project_summary.csv",
        index=False
    )

    return df


# ============================================================
# FIGURES
# ============================================================

def plot_activity_comparison(activity):

    if activity is None:
        return

    if "mean_activity" not in activity.columns:
        return

    groups = (
        activity
        .groupby("label")["mean_activity"]
        .mean()
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        groups.index,
        groups.values
    )

    plt.ylabel("Mean activity")

    plt.title(
        "Average Activity by Group"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "01_activity_group_comparison.png",
        dpi=300
    )

    plt.close()


def plot_sleep_comparison(sleep):

    if sleep is None:
        return

    if (
        "estimated_sleep_hours"
        not in sleep.columns
    ):
        return

    groups = (
        sleep
        .groupby("label")[
            "estimated_sleep_hours"
        ]
        .mean()
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        groups.index,
        groups.values
    )

    plt.ylabel(
        "Estimated sleep hours"
    )

    plt.title(
        "Estimated Sleep Duration by Group"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "02_sleep_group_comparison.png",
        dpi=300
    )

    plt.close()


def plot_cosinor_comparison(cosinor):

    if cosinor is None:
        return

    if "mesor" not in cosinor.columns:
        return

    groups = (
        cosinor
        .groupby("label")["mesor"]
        .mean()
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        groups.index,
        groups.values
    )

    plt.ylabel("Mesor")

    plt.title(
        "Average Circadian Mesor by Group"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "03_cosinor_group_comparison.png",
        dpi=300
    )

    plt.close()


def plot_model_performance(ml):

    if ml is None:
        return

    required = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc"
    ]

    if not all(
        column in ml.columns
        for column in required
    ):
        return

    x = np.arange(len(ml))
    width = 0.18

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        x - 1.5 * width,
        ml["accuracy"],
        width,
        label="Accuracy"
    )

    plt.bar(
        x - 0.5 * width,
        ml["balanced_accuracy"],
        width,
        label="Balanced Accuracy"
    )

    plt.bar(
        x + 0.5 * width,
        ml["f1"],
        width,
        label="F1"
    )

    plt.bar(
        x + 1.5 * width,
        ml["roc_auc"],
        width,
        label="ROC-AUC"
    )

    plt.xticks(
        x,
        ml["model"],
        rotation=20
    )

    plt.ylabel("Score")

    plt.ylim(0, 1)

    plt.title(
        "Classification Model Performance"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "04_final_model_comparison.png",
        dpi=300
    )

    plt.close()


def plot_final_feature_importance(importance):

    if importance is None:
        return

    data = (
        importance
        .head(10)
        .sort_values(
            "absolute_importance"
        )
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.barh(
        data["feature"],
        data["absolute_importance"]
    )

    plt.xlabel(
        "Absolute permutation importance"
    )

    plt.ylabel("Feature")

    plt.title(
        "Top Predictive Biological-Rhythm Features"
    )

    plt.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "05_final_feature_importance.png",
        dpi=300
    )

    plt.close()


def copy_final_roc_curve():

    source = (
        PHASE7_DIR
        / "plots"
        / "05_roc_curve_final_model.png"
    )

    destination = (
        PLOTS_DIR
        / "06_final_roc_curve.png"
    )

    if source.exists():

        shutil.copy2(
            source,
            destination
        )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    summary,
    checks
):

    report = {

        "project":
            "Automated Biological Rhythm Analysis "
            "& Health State Classification",

        "phase":
            "Phase 8 - Final Integration",

        "summary":
            summary,

        "validation_checks":
            checks
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            ),

        "methodological_notes": [

            "The project contains 55 subjects: "
            "23 condition and 32 control.",

            "Hourly visualization was used for "
            "the main activity analysis while "
            "fine-grained activity data were retained "
            "for detailed processing.",

            "The sleep analysis uses activity-derived "
            "heuristics and is not a clinical sleep "
            "measurement.",

            "Cosinor analysis used a fixed 24-hour "
            "period.",

            "The final ML evaluation used 5-fold "
            "stratified cross-validation.",

            "Recording-duration variables were excluded "
            "from the final ML evaluation.",

            "Phase 7 permutation importance was "
            "calculated on held-out validation folds.",

            "The classification results are exploratory "
            "and should not be interpreted as clinical "
            "diagnostic performance."
        ],

        "final_conclusion":
            "The integrated analysis demonstrates that "
            "activity, sleep-related and circadian "
            "features can provide predictive information "
            "for distinguishing condition and control "
            "subjects in the Depresjon dataset. The "
            "final Gradient Boosting evaluation achieved "
            "an out-of-fold ROC-AUC of approximately "
            "0.895 and accuracy of approximately 0.818. "
            "Estimated sleep hours showed the highest "
            "individual held-out permutation importance "
            "in the final explainability analysis. "
            "These findings are dataset-specific and "
            "exploratory rather than clinical validation."
    }

    path = (
        OUTPUT_DIR
        / "phase8_report.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            default=str
        )

    return path


# ============================================================
# MAIN
# ============================================================

def run_phase8():

    print("=" * 70)
    print(
        "PHASE 8 - FINAL INTEGRATION & "
        "PROJECT RESULTS"
    )
    print("=" * 70)

    create_directories()

    print()
    print("-" * 70)
    print(
        "INTEGRATING PHASE 3 ACTIVITY"
    )
    print("-" * 70)

    activity = integrate_activity()

    print()
    print(
        "INTEGRATING PHASE 3 SLEEP"
    )

    sleep = integrate_sleep()

    print()
    print(
        "INTEGRATING PHASE 4 COSINOR"
    )

    cosinor = integrate_cosinor()

    print()
    print(
        "INTEGRATING PHASE 5 STATISTICS"
    )

    statistics = integrate_statistics()

    print()
    print(
        "INTEGRATING PHASE 6 MACHINE LEARNING"
    )

    ml = integrate_ml()

    print()
    print(
        "INTEGRATING PHASE 7 EXPLAINABILITY"
    )

    (
        importance,
        groups
    ) = integrate_feature_importance()

    print()
    print("-" * 70)
    print(
        "BUILDING FINAL PROJECT SUMMARY"
    )
    print("-" * 70)

    summary = build_project_summary(
        activity,
        sleep,
        cosinor,
        statistics,
        ml,
        importance,
        groups
    )

    final_summary = (
        create_final_summary_table(
            summary
        )
    )

    print(
        final_summary.to_string(
            index=False
        )
    )

    print()
    print("-" * 70)
    print(
        "RUNNING FINAL VALIDATION CHECKS"
    )
    print("-" * 70)

    checks = run_validation_checks(
        summary,
        activity,
        sleep,
        cosinor,
        ml,
        importance
    )

    checks.to_csv(
        OUTPUT_DIR
        / "phase8_validation_checks.csv",
        index=False
    )

    print(
        checks.to_string(
            index=False
        )
    )

    print()
    print(
        "Generating final figures..."
    )

    plot_activity_comparison(
        activity
    )

    plot_sleep_comparison(
        sleep
    )

    plot_cosinor_comparison(
        cosinor
    )

    plot_model_performance(
        ml
    )

    plot_final_feature_importance(
        importance
    )

    copy_final_roc_curve()

    report_path = save_report(
        summary,
        checks
    )

    print()
    print("=" * 70)
    print(
        "PHASE 8 COMPLETED"
    )
    print("=" * 70)

    print()
    print(
        "Final results directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Final project summary:"
    )

    print(
        OUTPUT_DIR
        / "final_project_summary.csv"
    )

    print()
    print(
        "Validation checks:"
    )

    print(
        OUTPUT_DIR
        / "phase8_validation_checks.csv"
    )

    print()
    print(
        "Final report:"
    )

    print(
        report_path
    )

    print()
    print(
        "Final figures:"
    )

    print(
        PLOTS_DIR
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    run_phase8()