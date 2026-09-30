"""
PHASE 6 - FINAL LEAKAGE-AWARE HEALTH-STATE CLASSIFICATION

Automated Biological Rhythm Analysis & Health State Classification

Dataset:
    results/phase5/ml_ready_dataset.csv

Classification:
    condition vs control

Models:
    Logistic Regression
    Random Forest
    Support Vector Machine
    Gradient Boosting

Validation:
    5-fold Stratified Cross-Validation

Important methodological rules:
    1. Recording-duration variables are excluded.
    2. Preprocessing occurs inside each CV pipeline.
    3. Model performance is evaluated using out-of-fold predictions.
    4. No test-set performance is claimed.
    5. Results are exploratory because n = 55 subjects.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase6_final"
)

PLOTS_DIR = RESULTS_DIR / "plots"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5


# ============================================================
# FEATURES EXCLUDED FROM FINAL ML
# ============================================================

# These variables primarily describe recording quantity
# or duration rather than biological rhythm.

EXCLUDED_DURATION_FEATURES = {
    "hourly_records",
    "total_activity",
    "estimated_wake_hours"
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

def load_dataset():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "subject_id",
        "label"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    df
):

    excluded = {
        "subject_id",
        "label"
    }

    all_features = [
        column
        for column in df.columns
        if column not in excluded
    ]

    removed_features = [
        feature
        for feature in all_features
        if feature in EXCLUDED_DURATION_FEATURES
    ]

    feature_columns = [
        feature
        for feature in all_features
        if feature not in EXCLUDED_DURATION_FEATURES
    ]

    X = df[
        feature_columns
    ].copy()

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
            "Unknown labels detected."
        )

    y = y.astype(int)

    return (
        X,
        y,
        feature_columns,
        removed_features
    )


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    models = {

        "Logistic Regression":

            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    ),

                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE
                        )
                    )
                ]
            ),

        "Random Forest":

            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=500,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1
                        )
                    )
                ]
            ),

        "SVM":

            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    ),

                    (
                        "classifier",
                        SVC(
                            kernel="rbf",
                            probability=True,
                            class_weight="balanced",
                            random_state=RANDOM_STATE
                        )
                    )
                ]
            ),

        "Gradient Boosting":

            Pipeline(
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
    }

    return models


# ============================================================
# CROSS-VALIDATION
# ============================================================

def create_cv():

    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_models(
    models,
    X,
    y,
    cv
):

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    cv_records = []

    predictions = {}

    for model_name, model in models.items():

        print()
        print(
            f"Evaluating: {model_name}"
        )

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        record = {
            "model": model_name
        }

        for metric in scoring:

            values = scores[
                f"test_{metric}"
            ]

            record[
                f"{metric}_mean"
            ] = float(
                np.mean(values)
            )

            record[
                f"{metric}_std"
            ] = float(
                np.std(values)
            )

        cv_records.append(
            record
        )

        # ----------------------------------------------------
        # Out-of-fold class predictions.
        # ----------------------------------------------------

        predicted = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict",
            n_jobs=-1
        )

        # ----------------------------------------------------
        # Out-of-fold probabilities.
        # ----------------------------------------------------

        probabilities = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]

        predictions[
            model_name
        ] = {
            "predicted": predicted,
            "probability": probabilities
        }

    return (
        pd.DataFrame(cv_records),
        predictions
    )


# ============================================================
# OUT-OF-FOLD METRICS
# ============================================================

def calculate_oof_metrics(
    y,
    predictions
):

    records = []

    for model_name, result in predictions.items():

        predicted = result[
            "predicted"
        ]

        probability = result[
            "probability"
        ]

        records.append(
            {
                "model": model_name,

                "accuracy": accuracy_score(
                    y,
                    predicted
                ),

                "balanced_accuracy":
                    balanced_accuracy_score(
                        y,
                        predicted
                    ),

                "precision":
                    precision_score(
                        y,
                        predicted,
                        zero_division=0
                    ),

                "recall":
                    recall_score(
                        y,
                        predicted,
                        zero_division=0
                    ),

                "f1":
                    f1_score(
                        y,
                        predicted,
                        zero_division=0
                    ),

                "roc_auc":
                    roc_auc_score(
                        y,
                        probability
                    )
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# CONFUSION MATRICES
# ============================================================

def plot_confusion_matrices(
    y,
    predictions
):

    for model_name, result in predictions.items():

        cm = confusion_matrix(
            y,
            result["predicted"]
        )

        plt.figure(
            figsize=(6, 5)
        )

        plt.imshow(
            cm,
            interpolation="nearest"
        )

        plt.title(
            f"Confusion Matrix - {model_name}"
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

        filename = (
            model_name
            .lower()
            .replace(
                " ",
                "_"
            )
        )

        plt.savefig(
            PLOTS_DIR
            / f"confusion_matrix_{filename}.png",
            dpi=300
        )

        plt.close()


# ============================================================
# ROC CURVES
# ============================================================

def plot_roc_curves(
    y,
    predictions
):

    plt.figure(
        figsize=(9, 7)
    )

    for model_name, result in predictions.items():

        probabilities = result[
            "probability"
        ]

        fpr, tpr, _ = roc_curve(
            y,
            probabilities
        )

        auc = roc_auc_score(
            y,
            probabilities
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{model_name} "
                f"(AUC={auc:.3f})"
            )
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
        "ROC Curves - Final ML Evaluation"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "roc_curves.png",
        dpi=300
    )

    plt.close()


# ============================================================
# MODEL COMPARISON
# ============================================================

def plot_model_comparison(
    results
):

    metrics = [
        "accuracy_mean",
        "balanced_accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean"
    ]

    labels = [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]

    x = np.arange(
        len(metrics)
    )

    width = (
        0.8
        / len(results)
    )

    plt.figure(
        figsize=(14, 7)
    )

    for index, (_, row) in enumerate(
        results.iterrows()
    ):

        values = [
            row[metric]
            for metric in metrics
        ]

        plt.bar(
            x + index * width,
            values,
            width,
            label=row["model"]
        )

    plt.xticks(
        x
        + width * (
            len(results) - 1
        ) / 2,
        labels,
        rotation=20
    )

    plt.ylabel(
        "Score"
    )

    plt.ylim(
        0,
        1
    )

    plt.title(
        "Final Model Comparison"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "model_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# CLASSIFICATION REPORTS
# ============================================================

def generate_classification_reports(
    y,
    predictions
):

    reports = {}

    for model_name, result in predictions.items():

        reports[
            model_name
        ] = classification_report(
            y,
            result["predicted"],
            target_names=[
                "control",
                "condition"
            ],
            output_dict=True,
            zero_division=0
        )

    return reports


# ============================================================
# SAVE OOF PREDICTIONS
# ============================================================

def save_oof_predictions(
    df,
    predictions
):

    output = df[
        [
            "subject_id",
            "label"
        ]
    ].copy()

    for model_name, result in predictions.items():

        safe_name = (
            model_name
            .lower()
            .replace(
                " ",
                "_"
            )
        )

        output[
            f"{safe_name}_prediction"
        ] = result[
            "predicted"
        ]

        output[
            f"{safe_name}_probability"
        ] = result[
            "probability"
        ]

    output_path = (
        RESULTS_DIR
        / "out_of_fold_predictions.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    return output_path


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    df,
    feature_columns,
    removed_features,
    cv_results,
    oof_results,
    reports
):

    report = {

        "phase":
            "Phase 6 - Final Leakage-Aware "
            "Health-State Classification",

        "dataset": {

            "subjects":
                int(
                    len(df)
                ),

            "condition_subjects":
                int(
                    (
                        df["label"]
                        == "condition"
                    ).sum()
                ),

            "control_subjects":
                int(
                    (
                        df["label"]
                        == "control"
                    ).sum()
                ),

            "features_used":
                int(
                    len(feature_columns)
                ),

            "features_removed":
                removed_features
        },

        "validation": {

            "method":
                "5-fold Stratified Cross-Validation",

            "shuffle":
                True,

            "random_state":
                RANDOM_STATE,

            "preprocessing":
                "Imputation and scaling are fitted "
                "within each training fold."
        },

        "models":
            list(
                cv_results[
                    "model"
                ]
            ),

        "cross_validation_results":
            cv_results
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            ),

        "out_of_fold_results":
            oof_results
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            ),

        "classification_reports":
            reports,

        "interpretation_note":
            "These results are exploratory estimates "
            "from 55 subjects and should not be "
            "interpreted as clinical diagnostic "
            "performance."
    }

    report_path = (
        RESULTS_DIR
        / "phase6_final_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            default=str
        )

    return report_path


# ============================================================
# MAIN
# ============================================================

def run_phase6():

    print("=" * 70)
    print(
        "PHASE 6 - FINAL LEAKAGE-AWARE "
        "HEALTH-STATE CLASSIFICATION"
    )
    print("=" * 70)

    create_directories()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING PHASE 5 DATASET")
    print("-" * 70)

    df = load_dataset()

    print(
        f"Subjects: {len(df)}"
    )

    print()
    print(
        "Class distribution:"
    )

    print(
        df["label"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    (
        X,
        y,
        feature_columns,
        removed_features
    ) = prepare_data(
        df
    )

    print()
    print(
        "Features removed from final ML:"
    )

    for feature in removed_features:

        print(
            f"  - {feature}"
        )

    print()
    print(
        f"Features used: "
        f"{len(feature_columns)}"
    )

    print()

    for feature in feature_columns:

        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = create_models()

    print()
    print(
        "Models:"
    )

    for model in models:

        print(
            f"  - {model}"
        )

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    cv = create_cv()

    print()
    print("-" * 70)
    print(
        "5-FOLD STRATIFIED "
        "CROSS-VALIDATION"
    )
    print("-" * 70)

    (
        cv_results,
        predictions
    ) = evaluate_models(
        models,
        X,
        y,
        cv
    )

    # --------------------------------------------------------
    # Save CV results
    # --------------------------------------------------------

    cv_results.to_csv(
        RESULTS_DIR
        / "cross_validation_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # OOF results
    # --------------------------------------------------------

    oof_results = (
        calculate_oof_metrics(
            y,
            predictions
        )
    )

    oof_results.to_csv(
        RESULTS_DIR
        / "out_of_fold_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # Print CV results
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(
        "CROSS-VALIDATION RESULTS"
    )
    print("-" * 70)

    display_columns = [
        "model",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean"
    ]

    print(
        cv_results[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Print OOF results
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(
        "OUT-OF-FOLD RESULTS"
    )
    print("-" * 70)

    print(
        oof_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_path = (
        save_oof_predictions(
            df,
            predictions
        )
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    print()
    print(
        "Generating confusion matrices..."
    )

    plot_confusion_matrices(
        y,
        predictions
    )

    print(
        "Generating ROC curves..."
    )

    plot_roc_curves(
        y,
        predictions
    )

    print(
        "Generating model comparison..."
    )

    plot_model_comparison(
        cv_results
    )

    # --------------------------------------------------------
    # Classification reports
    # --------------------------------------------------------

    reports = (
        generate_classification_reports(
            y,
            predictions
        )
    )

    # --------------------------------------------------------
    # Save final report
    # --------------------------------------------------------

    report_path = save_report(
        df,
        feature_columns,
        removed_features,
        cv_results,
        oof_results,
        reports
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL PHASE 6 COMPLETED"
    )
    print("=" * 70)

    print()
    print(
        f"Subjects: {len(df)}"
    )

    print(
        f"Features used: "
        f"{len(feature_columns)}"
    )

    print()
    print(
        "Cross-validation results:"
    )

    print(
        RESULTS_DIR
        / "cross_validation_results.csv"
    )

    print()
    print(
        "Out-of-fold results:"
    )

    print(
        RESULTS_DIR
        / "out_of_fold_results.csv"
    )

    print()
    print(
        "Out-of-fold predictions:"
    )

    print(
        prediction_path
    )

    print()
    print(
        "Plots:"
    )

    print(
        PLOTS_DIR
    )

    print()
    print(
        "Report:"
    )

    print(
        report_path
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    run_phase6()