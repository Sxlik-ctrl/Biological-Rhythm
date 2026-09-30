"""
PHASE 7 - EXPLAINABILITY & FINAL EVALUATION

Automated Biological Rhythm Analysis & Health State Classification

Purpose:
    Explain the final classification model using
    held-out permutation importance and final
    out-of-fold evaluation.

Dataset:
    results/phase5/ml_ready_dataset.csv

Final model:
    Gradient Boosting

Validation:
    5-fold Stratified Cross-Validation

Important:
    Feature importance is calculated on held-out
    validation folds rather than training data.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "phase5"
    / "ml_ready_dataset.csv"
)

PHASE6_OOF_FILE = (
    PROJECT_ROOT
    / "results"
    / "phase6_final"
    / "out_of_fold_predictions.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase7"
)

PLOTS_DIR = RESULTS_DIR / "plots"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

N_PERMUTATIONS = 30

EXCLUDED_FEATURES = {
    "hourly_records",
    "total_activity",
    "estimated_wake_hours"
}


# ============================================================
# FEATURE GROUPS
# ============================================================

FEATURE_GROUPS = {

    "Activity": [
        "median_activity",
        "activity_max",
        "day_night_ratio",
        "most_active_hour",
        "least_active_hour",
        "activity_cv"
    ],

    "Sleep": [
        "estimated_sleep_hours",
        "estimated_sleep_periods",
        "sleep_regularity"
    ],

    "Circadian / Cosinor": [
        "mesor",
        "relative_amplitude",
        "trough_hours",
        "fitted_min",
        "rmse"
    ]
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    X = df.drop(
        columns=[
            "subject_id",
            "label"
        ]
    )

    X = X.drop(
        columns=[
            column
            for column in EXCLUDED_FEATURES
            if column in X.columns
        ]
    )

    y = df[
        "label"
    ].map(
        {
            "control": 0,
            "condition": 1
        }
    )

    if y.isna().any():

        raise ValueError(
            "Unknown labels found."
        )

    y = y.astype(int)

    return df, X, y


# ============================================================
# CREATE FINAL MODEL
# ============================================================

def create_model():

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=2,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    return model


# ============================================================
# HELD-OUT PERMUTATION IMPORTANCE
# ============================================================

def calculate_permutation_importance(
    X,
    y,
    cv
):

    feature_names = list(
        X.columns
    )

    importance_records = []

    fold_predictions = []

    for fold_number, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print()
        print(
            f"Processing fold {fold_number}/{N_SPLITS}"
        )

        X_train = X.iloc[
            train_index
        ]

        X_test = X.iloc[
            test_index
        ]

        y_train = y.iloc[
            train_index
        ]

        y_test = y.iloc[
            test_index
        ]

        model = create_model()

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Predictions on unseen validation fold
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        fold_predictions.append(
            {
                "fold": fold_number,
                "test_index": test_index,
                "y_true": y_test.to_numpy(),
                "y_pred": predictions,
                "probability": probabilities
            }
        )

        # ----------------------------------------------------
        # Permutation importance
        # ----------------------------------------------------

        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=N_PERMUTATIONS,
            random_state=(
                RANDOM_STATE
                + fold_number
            ),
            n_jobs=-1
        )

        for index, feature in enumerate(
            feature_names
        ):

            importance_records.append(
                {
                    "fold": fold_number,
                    "feature": feature,
                    "importance_mean":
                        result.importances_mean[
                            index
                        ],
                    "importance_std":
                        result.importances_std[
                            index
                        ]
                }
            )

    importance_df = pd.DataFrame(
        importance_records
    )

    summary = (
        importance_df
        .groupby("feature")
        .agg(
            importance_mean=(
                "importance_mean",
                "mean"
            ),
            importance_std=(
                "importance_mean",
                "std"
            ),
            mean_permutation_std=(
                "importance_std",
                "mean"
            )
        )
        .reset_index()
    )

    summary[
        "absolute_importance"
    ] = summary[
        "importance_mean"
    ].abs()

    summary = summary.sort_values(
        "absolute_importance",
        ascending=False
    )

    return (
        importance_df,
        summary,
        fold_predictions
    )


# ============================================================
# OOF FINAL METRICS
# ============================================================

def calculate_final_oof_metrics(
    fold_predictions
):

    y_true = np.concatenate(
        [
            fold["y_true"]
            for fold in fold_predictions
        ]
    )

    y_pred = np.concatenate(
        [
            fold["y_pred"]
            for fold in fold_predictions
        ]
    )

    probabilities = np.concatenate(
        [
            fold["probability"]
            for fold in fold_predictions
        ]
    )

    metrics = {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities
            )
    }

    return (
        metrics,
        y_true,
        y_pred,
        probabilities
    )


# ============================================================
# FEATURE GROUP IMPORTANCE
# ============================================================

def calculate_group_importance(
    feature_summary
):

    records = []

    for group_name, features in (
        FEATURE_GROUPS.items()
    ):

        available = feature_summary[
            feature_summary["feature"]
            .isin(features)
        ]

        if len(available) == 0:

            continue

        records.append(
            {
                "feature_group":
                    group_name,

                "number_of_features":
                    len(available),

                "mean_absolute_importance":
                    available[
                        "absolute_importance"
                    ].mean(),

                "total_absolute_importance":
                    available[
                        "absolute_importance"
                    ].sum()
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# FEATURE IMPORTANCE PLOT
# ============================================================

def plot_feature_importance(
    feature_summary
):

    data = (
        feature_summary
        .head(15)
        .sort_values(
            "absolute_importance"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        data["feature"],
        data["importance_mean"]
    )

    plt.xlabel(
        "Mean held-out permutation importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Gradient Boosting Feature Importance"
    )

    plt.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "01_feature_importance.png",
        dpi=300
    )

    plt.close()


# ============================================================
# GROUP IMPORTANCE PLOT
# ============================================================

def plot_group_importance(
    group_summary
):

    data = (
        group_summary
        .sort_values(
            "mean_absolute_importance"
        )
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.barh(
        data["feature_group"],
        data["mean_absolute_importance"]
    )

    plt.xlabel(
        "Mean absolute permutation importance"
    )

    plt.ylabel(
        "Feature group"
    )

    plt.title(
        "Biological Feature-Group Importance"
    )

    plt.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "03_feature_group_importance.png",
        dpi=300
    )

    plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "Gradient Boosting - "
        "Out-of-Fold Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        [
            "Control",
            "Condition"
        ]
    )

    plt.yticks(
        [0, 1],
        [
            "Control",
            "Condition"
        ]
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    for row in range(2):

        for column in range(2):

            plt.text(
                column,
                row,
                str(
                    cm[
                        row,
                        column
                    ]
                ),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "04_confusion_matrix_final_model.png",
        dpi=300
    )

    plt.close()


# ============================================================
# ROC CURVE
# ============================================================

def plot_roc_curve(
    y_true,
    probabilities
):

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities
    )

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"Gradient Boosting (AUC={auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Gradient Boosting - Out-of-Fold ROC Curve"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "05_roc_curve_final_model.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

def plot_probability_distribution(
    y_true,
    probabilities
):

    control_probabilities = probabilities[
        y_true == 0
    ]

    condition_probabilities = probabilities[
        y_true == 1
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        control_probabilities,
        bins=10,
        alpha=0.6,
        label="Control"
    )

    plt.hist(
        condition_probabilities,
        bins=10,
        alpha=0.6,
        label="Condition"
    )

    plt.xlabel(
        "Predicted probability of condition"
    )

    plt.ylabel(
        "Number of subjects"
    )

    plt.title(
        "Out-of-Fold Prediction Probability Distribution"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "06_prediction_probability_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# SAVE OOF FINAL PREDICTIONS
# ============================================================

def save_predictions(
    df,
    fold_predictions
):

    prediction_table = pd.DataFrame(
        {
            "subject_id":
                df["subject_id"].values,

            "label":
                df["label"].values,

            "actual":
                np.concatenate(
                    [
                        fold["y_true"]
                        for fold in fold_predictions
                    ]
                ),

            "prediction":
                np.concatenate(
                    [
                        fold["y_pred"]
                        for fold in fold_predictions
                    ]
                ),

            "condition_probability":
                np.concatenate(
                    [
                        fold["probability"]
                        for fold in fold_predictions
                    ]
                )
        }
    )

    path = (
        RESULTS_DIR
        / "final_model_oof_predictions.csv"
    )

    prediction_table.to_csv(
        path,
        index=False
    )

    return path


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    df,
    X,
    metrics,
    feature_summary,
    group_summary
):

    report = {

        "phase":
            "Phase 7 - Explainability "
            "and Final Evaluation",

        "dataset": {

            "subjects":
                int(len(df)),

            "control":
                int(
                    (
                        df["label"]
                        == "control"
                    ).sum()
                ),

            "condition":
                int(
                    (
                        df["label"]
                        == "condition"
                    ).sum()
                ),

            "features":
                list(X.columns)
        },

        "model":
            "Gradient Boosting",

        "validation":
            "5-fold Stratified Cross-Validation",

        "permutation_importance":
            feature_summary
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            ),

        "feature_group_importance":
            group_summary
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            ),

        "final_oof_metrics":
            metrics,

        "interpretation_note":
            "Permutation importance was calculated "
            "on held-out validation folds. Positive "
            "importance indicates that permuting the "
            "feature reduced validation ROC-AUC on "
            "average. Importance values describe "
            "predictive contribution and do not "
            "establish causation or clinical "
            "diagnostic validity."
    }

    path = (
        RESULTS_DIR
        / "phase7_report.json"
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

def run_phase7():

    print("=" * 70)
    print(
        "PHASE 7 - EXPLAINABILITY & FINAL EVALUATION"
    )
    print("=" * 70)

    create_directories()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(
        "LOADING DATA"
    )
    print("-" * 70)

    df, X, y = load_data()

    print(
        f"Subjects: {len(df)}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print()
    print(
        "Features used:"
    )

    for feature in X.columns:

        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Permutation importance
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(
        "HELD-OUT PERMUTATION IMPORTANCE"
    )
    print("-" * 70)

    (
        importance_df,
        feature_summary,
        fold_predictions
    ) = calculate_permutation_importance(
        X,
        y,
        cv
    )

    importance_df.to_csv(
        RESULTS_DIR
        / "permutation_importance_by_fold.csv",
        index=False
    )

    feature_summary.to_csv(
        RESULTS_DIR
        / "permutation_importance.csv",
        index=False
    )

    # --------------------------------------------------------
    # Print importance
    # --------------------------------------------------------

    print()
    print(
        "Top features:"
    )

    print(
        feature_summary[
            [
                "feature",
                "importance_mean",
                "importance_std",
                "absolute_importance"
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Feature groups
    # --------------------------------------------------------

    group_summary = (
        calculate_group_importance(
            feature_summary
        )
    )

    group_summary.to_csv(
        RESULTS_DIR
        / "feature_group_importance.csv",
        index=False
    )

    print()
    print(
        "Feature-group importance:"
    )

    print(
        group_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Final OOF metrics
    # --------------------------------------------------------

    (
        metrics,
        y_true,
        y_pred,
        probabilities
    ) = calculate_final_oof_metrics(
        fold_predictions
    )

    metrics_df = pd.DataFrame(
        [
            metrics
        ]
    )

    metrics_df.to_csv(
        RESULTS_DIR
        / "final_evaluation_summary.csv",
        index=False
    )

    print()
    print("-" * 70)
    print(
        "FINAL GRADIENT BOOSTING "
        "OUT-OF-FOLD METRICS"
    )
    print("-" * 70)

    for metric, value in metrics.items():

        print(
            f"{metric:22s}: "
            f"{value:.4f}"
        )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    print()
    print(
        "Generating plots..."
    )

    plot_feature_importance(
        feature_summary
    )

    plot_group_importance(
        group_summary
    )

    plot_confusion_matrix(
        y_true,
        y_pred
    )

    plot_roc_curve(
        y_true,
        probabilities
    )

    plot_probability_distribution(
        y_true,
        probabilities
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    prediction_path = save_predictions(
        df,
        fold_predictions
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = save_report(
        df,
        X,
        metrics,
        feature_summary,
        group_summary
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PHASE 7 COMPLETED"
    )
    print("=" * 70)

    print()
    print(
        "Results:"
    )

    print(
        RESULTS_DIR
    )

    print()
    print(
        "Feature importance:"
    )

    print(
        RESULTS_DIR
        / "permutation_importance.csv"
    )

    print()
    print(
        "Final evaluation:"
    )

    print(
        RESULTS_DIR
        / "final_evaluation_summary.csv"
    )

    print()
    print(
        "Predictions:"
    )

    print(
        prediction_path
    )

    print()
    print(
        "Report:"
    )

    print(
        report_path
    )

    print()
    print(
        "Plots:"
    )

    print(
        PLOTS_DIR
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    run_phase7()