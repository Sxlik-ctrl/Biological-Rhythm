"""
PHASE 5 - STATISTICAL ANALYSIS & FEATURE SELECTION

Automated Biological Rhythm Analysis & Health State Classification

Inputs:
    results/phase3/activity_features.csv
    results/phase3/sleep_features.csv
    results/phase4/cosinor_subject_features.csv

Outputs:
    Combined feature dataset
    Statistical comparison
    Effect sizes
    Multiple-testing correction
    Correlation matrix
    Feature selection
    ML-ready dataset
    Visualizations
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    mannwhitneyu,
    shapiro,
    spearmanr
)

from statsmodels.stats.multitest import (
    multipletests
)


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHASE3_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase3"
)

PHASE4_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase4"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase5"
)

PLOTS_DIR = RESULTS_DIR / "plots"


# ============================================================
# INPUT FILES
# ============================================================

ACTIVITY_FILE = (
    PHASE3_DIR
    / "activity_features.csv"
)

SLEEP_FILE = (
    PHASE3_DIR
    / "sleep_features.csv"
)

COSINOR_FILE = (
    PHASE4_DIR
    / "cosinor_subject_features.csv"
)


# ============================================================
# STATISTICAL SETTINGS
# ============================================================

ALPHA = 0.05

# FDR threshold for multiple testing.
FDR_ALPHA = 0.05

# Correlation threshold used to identify highly redundant
# numerical features.
CORRELATION_THRESHOLD = 0.90

# Minimum absolute effect size considered notable for
# feature-selection reporting.
EFFECT_SIZE_THRESHOLD = 0.20


# ============================================================
# DIRECTORY SETUP
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
# LOAD FEATURE FILES
# ============================================================

def load_features():

    for path in [
        ACTIVITY_FILE,
        SLEEP_FILE,
        COSINOR_FILE
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    activity = pd.read_csv(
        ACTIVITY_FILE
    )

    sleep = pd.read_csv(
        SLEEP_FILE
    )

    cosinor = pd.read_csv(
        COSINOR_FILE
    )

    return (
        activity,
        sleep,
        cosinor
    )


# ============================================================
# PREPARE FEATURE DATASETS
# ============================================================

def prepare_features(
    activity,
    sleep,
    cosinor
):

    # --------------------------------------------------------
    # Remove duplicate label columns before merging.
    # --------------------------------------------------------

    activity = activity.copy()

    sleep = sleep.copy()

    cosinor = cosinor.copy()

    # --------------------------------------------------------
    # Cosinor columns to retain.
    # --------------------------------------------------------

    cosinor_keep = [
        "subject_id",
        "label",
        "mesor",
        "amplitude",
        "relative_amplitude",
        "acrophase_hours",
        "trough_hours",
        "fitted_max",
        "fitted_min",
        "r_squared",
        "rmse"
    ]

    cosinor = cosinor[
        [
            column
            for column in cosinor_keep
            if column in cosinor.columns
        ]
    ]

    # --------------------------------------------------------
    # Merge.
    # --------------------------------------------------------

    combined = activity.merge(
        sleep,
        on=[
            "subject_id",
            "label"
        ],
        how="inner",
        suffixes=(
            "",
            "_sleep"
        )
    )

    combined = combined.merge(
        cosinor,
        on=[
            "subject_id",
            "label"
        ],
        how="inner",
        suffixes=(
            "",
            "_cosinor"
        )
    )

    # --------------------------------------------------------
    # Remove accidental duplicate columns.
    # --------------------------------------------------------

    duplicate_columns = [
        column
        for column in combined.columns
        if column.endswith(
            "_sleep"
        )
        or column.endswith(
            "_cosinor"
        )
    ]

    # Keep original versions wherever possible.
    combined = combined.drop(
        columns=duplicate_columns,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Numeric conversion.
    # --------------------------------------------------------

    for column in combined.columns:

        if column not in [
            "subject_id",
            "label"
        ]:

            combined[column] = pd.to_numeric(
                combined[column],
                errors="coerce"
            )

    return combined


# ============================================================
# GET FEATURE COLUMNS
# ============================================================

def get_feature_columns(
    df
):

    excluded = {
        "subject_id",
        "label"
    }

    return [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]


# ============================================================
# DATA QUALITY
# ============================================================

def calculate_data_quality(
    df,
    features
):

    records = []

    for feature in features:

        values = df[feature]

        records.append(
            {
                "feature": feature,
                "n": int(
                    values.notna().sum()
                ),
                "missing": int(
                    values.isna().sum()
                ),
                "missing_percent": float(
                    values.isna().mean()
                    * 100
                ),
                "mean": float(
                    values.mean()
                ),
                "std": float(
                    values.std()
                ),
                "min": float(
                    values.min()
                ),
                "max": float(
                    values.max()
                )
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def calculate_descriptive_statistics(
    df,
    features
):

    records = []

    for feature in features:

        for label in [
            "condition",
            "control"
        ]:

            values = df.loc[
                df["label"] == label,
                feature
            ].dropna()

            if len(values) == 0:
                continue

            records.append(
                {
                    "feature": feature,
                    "label": label,
                    "n": int(
                        len(values)
                    ),
                    "mean": float(
                        values.mean()
                    ),
                    "median": float(
                        values.median()
                    ),
                    "std": float(
                        values.std()
                    ),
                    "min": float(
                        values.min()
                    ),
                    "max": float(
                        values.max()
                    ),
                    "q25": float(
                        values.quantile(
                            0.25
                        )
                    ),
                    "q75": float(
                        values.quantile(
                            0.75
                        )
                    )
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# COHEN'S D
# ============================================================

def cohens_d(
    group1,
    group2
):

    group1 = np.asarray(
        group1,
        dtype=float
    )

    group2 = np.asarray(
        group2,
        dtype=float
    )

    group1 = group1[
        np.isfinite(group1)
    ]

    group2 = group2[
        np.isfinite(group2)
    ]

    n1 = len(group1)
    n2 = len(group2)

    if n1 < 2 or n2 < 2:
        return np.nan

    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    var1 = np.var(
        group1,
        ddof=1
    )

    var2 = np.var(
        group2,
        ddof=1
    )

    pooled_sd = np.sqrt(
        (
            (n1 - 1) * var1
            + (n2 - 1) * var2
        )
        /
        (
            n1 + n2 - 2
        )
    )

    if pooled_sd == 0:
        return 0.0

    return (
        (mean1 - mean2)
        / pooled_sd
    )


# ============================================================
# RANK-BISERIAL EFFECT SIZE
# ============================================================

def rank_biserial_effect(
    condition,
    control
):

    condition = np.asarray(
        condition,
        dtype=float
    )

    control = np.asarray(
        control,
        dtype=float
    )

    condition = condition[
        np.isfinite(condition)
    ]

    control = control[
        np.isfinite(control)
    ]

    if (
        len(condition) == 0
        or len(control) == 0
    ):
        return np.nan

    try:

        u_stat, _ = mannwhitneyu(
            condition,
            control,
            alternative="two-sided"
        )

    except Exception:

        return np.nan

    n1 = len(condition)
    n2 = len(control)

    return (
        2 * u_stat
        / (n1 * n2)
        - 1
    )


# ============================================================
# STATISTICAL TESTING
# ============================================================

def perform_statistical_tests(
    df,
    features
):

    records = []

    for feature in features:

        condition = df.loc[
            df["label"] == "condition",
            feature
        ].dropna()

        control = df.loc[
            df["label"] == "control",
            feature
        ].dropna()

        if (
            len(condition) < 3
            or len(control) < 3
        ):
            continue

        # ----------------------------------------------------
        # Shapiro normality checks.
        # ----------------------------------------------------

        try:

            if len(condition) <= 5000:

                shapiro_condition = (
                    shapiro(
                        condition
                    ).pvalue
                )

            else:

                shapiro_condition = np.nan

        except Exception:

            shapiro_condition = np.nan

        try:

            if len(control) <= 5000:

                shapiro_control = (
                    shapiro(
                        control
                    ).pvalue
                )

            else:

                shapiro_control = np.nan

        except Exception:

            shapiro_control = np.nan

        # ----------------------------------------------------
        # Mann-Whitney U.
        #
        # Non-parametric test is used because sample sizes
        # are relatively small and distributions may not
        # satisfy normality assumptions.
        # ----------------------------------------------------

        try:

            u_statistic, p_value = (
                mannwhitneyu(
                    condition,
                    control,
                    alternative="two-sided"
                )
            )

        except Exception:

            u_statistic = np.nan
            p_value = np.nan

        # ----------------------------------------------------
        # Effect sizes.
        # ----------------------------------------------------

        d = cohens_d(
            condition,
            control
        )

        rank_biserial = (
            rank_biserial_effect(
                condition,
                control
            )
        )

        records.append(
            {
                "feature": feature,
                "condition_n": int(
                    len(condition)
                ),
                "control_n": int(
                    len(control)
                ),
                "condition_mean": float(
                    condition.mean()
                ),
                "control_mean": float(
                    control.mean()
                ),
                "condition_median": float(
                    condition.median()
                ),
                "control_median": float(
                    control.median()
                ),
                "shapiro_condition_p": float(
                    shapiro_condition
                ),
                "shapiro_control_p": float(
                    shapiro_control
                ),
                "mann_whitney_u": float(
                    u_statistic
                ),
                "p_value": float(
                    p_value
                ),
                "cohens_d": float(
                    d
                ),
                "rank_biserial": float(
                    rank_biserial
                )
            }
        )

    result = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Multiple testing correction.
    # --------------------------------------------------------

    if (
        not result.empty
        and result["p_value"]
        .notna()
        .any()
    ):

        valid = result[
            "p_value"
        ].notna()

        adjusted = multipletests(
            result.loc[
                valid,
                "p_value"
            ],
            alpha=FDR_ALPHA,
            method="fdr_bh"
        )

        result.loc[
            valid,
            "fdr_p_value"
        ] = adjusted[1]

        result.loc[
            valid,
            "significant_fdr"
        ] = adjusted[0]

    else:

        result["fdr_p_value"] = np.nan
        result["significant_fdr"] = False

    # --------------------------------------------------------
    # Effect-size flag.
    # --------------------------------------------------------

    result["effect_size_notable"] = (
        result["cohens_d"]
        .abs()
        >= EFFECT_SIZE_THRESHOLD
    )

    return result


# ============================================================
# CORRELATION MATRIX
# ============================================================

def calculate_correlations(
    df,
    features
):

    matrix = pd.DataFrame(
        np.nan,
        index=features,
        columns=features
    )

    for feature1 in features:

        for feature2 in features:

            valid = df[
                [
                    feature1,
                    feature2
                ]
            ].dropna()

            if len(valid) < 3:
                continue

            try:

                correlation, _ = (
                    spearmanr(
                        valid[feature1],
                        valid[feature2]
                    )
                )

                matrix.loc[
                    feature1,
                    feature2
                ] = correlation

            except Exception:

                pass

    return matrix


# ============================================================
# REDUNDANCY DETECTION
# ============================================================

def identify_redundant_features(
    correlation_matrix
):

    redundant = []

    features = list(
        correlation_matrix.columns
    )

    for i in range(
        len(features)
    ):

        for j in range(
            i + 1,
            len(features)
        ):

            feature1 = features[i]
            feature2 = features[j]

            correlation = (
                correlation_matrix.loc[
                    feature1,
                    feature2
                ]
            )

            if pd.isna(correlation):
                continue

            if abs(correlation) >= (
                CORRELATION_THRESHOLD
            ):

                redundant.append(
                    {
                        "feature_1": feature1,
                        "feature_2": feature2,
                        "spearman_correlation": float(
                            correlation
                        )
                    }
                )

    return pd.DataFrame(
        redundant
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features(
    statistical_results,
    correlation_matrix,
    features
):

    selected = []

    for feature in features:

        row = statistical_results[
            statistical_results["feature"]
            == feature
        ]

        if row.empty:
            continue

        row = row.iloc[0]

        p_value = row[
            "fdr_p_value"
        ]

        effect = row[
            "cohens_d"
        ]

        # ----------------------------------------------------
        # Primary selection criterion:
        #
        # FDR significance OR meaningful effect size.
        #
        # This prevents us from throwing away potentially
        # useful features simply because the sample is small.
        # ----------------------------------------------------

        fdr_significant = (
            pd.notna(p_value)
            and p_value < FDR_ALPHA
        )

        meaningful_effect = (
            pd.notna(effect)
            and abs(effect)
            >= EFFECT_SIZE_THRESHOLD
        )

        if (
            fdr_significant
            or meaningful_effect
        ):

            selected.append(
                feature
            )

    # --------------------------------------------------------
    # Remove highly correlated duplicates.
    #
    # When two features are highly correlated, retain the
    # feature with stronger statistical evidence.
    # --------------------------------------------------------

    final_selected = []

    for feature in selected:

        keep = True

        for existing in final_selected:

            if (
                feature not in
                correlation_matrix.index
                or existing not in
                correlation_matrix.columns
            ):
                continue

            correlation = (
                correlation_matrix.loc[
                    feature,
                    existing
                ]
            )

            if pd.isna(correlation):
                continue

            if abs(correlation) >= (
                CORRELATION_THRESHOLD
            ):

                current_row = (
                    statistical_results[
                        statistical_results[
                            "feature"
                        ] == feature
                    ]
                )

                existing_row = (
                    statistical_results[
                        statistical_results[
                            "feature"
                        ] == existing
                    ]
                )

                if (
                    current_row.empty
                    or existing_row.empty
                ):
                    continue

                current_score = (
                    abs(
                        current_row.iloc[0][
                            "cohens_d"
                        ]
                    )
                )

                existing_score = (
                    abs(
                        existing_row.iloc[0][
                            "cohens_d"
                        ]
                    )
                )

                if (
                    pd.notna(current_score)
                    and pd.notna(existing_score)
                    and current_score
                    > existing_score
                ):

                    final_selected.remove(
                        existing
                    )

                    break

                else:

                    keep = False
                    break

        if keep:
            final_selected.append(
                feature
            )

    return final_selected


# ============================================================
# ML DATASET
# ============================================================

def create_ml_dataset(
    df,
    selected_features
):

    columns = [
        "subject_id",
        "label"
    ] + selected_features

    ml_df = df[
        [
            column
            for column in columns
            if column in df.columns
        ]
    ].copy()

    return ml_df


# ============================================================
# PLOT FEATURE DISTRIBUTIONS
# ============================================================

def plot_feature_distributions(
    df,
    statistical_results
):

    # Select up to 8 features with largest
    # absolute effect sizes.

    ranked = (
        statistical_results
        .assign(
            abs_effect=lambda x:
            x["cohens_d"].abs()
        )
        .sort_values(
            "abs_effect",
            ascending=False
        )
    )

    selected = ranked[
        "feature"
    ].head(8).tolist()

    for feature in selected:

        plt.figure(
            figsize=(9, 6)
        )

        condition = df.loc[
            df["label"] == "condition",
            feature
        ].dropna()

        control = df.loc[
            df["label"] == "control",
            feature
        ].dropna()

        plt.boxplot(
            [
                condition,
                control
            ],
            labels=[
                "Condition",
                "Control"
            ]
        )

        plt.ylabel(
            feature
        )

        plt.title(
            f"Distribution: {feature}"
        )

        plt.grid(
            axis="y",
            alpha=0.3
        )

        plt.tight_layout()

        safe_name = (
            feature
            .replace(
                " ",
                "_"
            )
        )

        plt.savefig(
            PLOTS_DIR
            / f"feature_{safe_name}.png",
            dpi=300
        )

        plt.close()


# ============================================================
# PLOT CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(
    correlation_matrix
):

    plt.figure(
        figsize=(14, 12)
    )

    matrix = (
        correlation_matrix
        .values
    )

    image = plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest"
    )

    plt.colorbar(
        image,
        label="Spearman correlation"
    )

    plt.xticks(
        range(
            len(correlation_matrix.columns)
        ),
        correlation_matrix.columns,
        rotation=90
    )

    plt.yticks(
        range(
            len(correlation_matrix.index)
        ),
        correlation_matrix.index
    )

    plt.title(
        "Feature Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "02_correlation_heatmap.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT EFFECT SIZES
# ============================================================

def plot_effect_sizes(
    statistical_results
):

    ranked = (
        statistical_results
        .assign(
            abs_effect=lambda x:
            x["cohens_d"].abs()
        )
        .sort_values(
            "abs_effect",
            ascending=True
        )
    )

    plt.figure(
        figsize=(10, 12)
    )

    plt.barh(
        ranked["feature"],
        ranked["cohens_d"]
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Cohen's d"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Feature Effect Sizes"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "03_effect_sizes.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT SELECTED FEATURES
# ============================================================

def plot_selected_features(
    statistical_results,
    selected_features
):

    subset = statistical_results[
        statistical_results[
            "feature"
        ].isin(
            selected_features
        )
    ].copy()

    if subset.empty:
        return

    subset["abs_effect"] = (
        subset["cohens_d"]
        .abs()
    )

    subset = subset.sort_values(
        "abs_effect",
        ascending=True
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        subset["feature"],
        subset["cohens_d"]
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Cohen's d"
    )

    plt.ylabel(
        "Selected Feature"
    )

    plt.title(
        "Selected Features by Effect Size"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "04_selected_features.png",
        dpi=300
    )

    plt.close()


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    combined,
    features,
    statistical_results,
    redundant_features,
    selected_features
):

    significant_count = int(
        (
            statistical_results[
                "significant_fdr"
            ]
            .fillna(False)
        )
        .sum()
    )

    report = {

        "phase": (
            "Phase 5 - "
            "Statistical Analysis and "
            "Feature Selection"
        ),

        "subjects": int(
            combined["subject_id"]
            .nunique()
        ),

        "condition_subjects": int(
            (
                combined["label"]
                == "condition"
            ).sum()
        ),

        "control_subjects": int(
            (
                combined["label"]
                == "control"
            ).sum()
        ),

        "total_features_before_selection": int(
            len(features)
        ),

        "statistical_test": (
            "Mann-Whitney U "
            "two-sided"
        ),

        "multiple_testing_correction": (
            "Benjamini-Hochberg FDR"
        ),

        "alpha": ALPHA,

        "fdr_alpha": FDR_ALPHA,

        "features_significant_after_fdr": (
            significant_count
        ),

        "correlation_threshold": (
            CORRELATION_THRESHOLD
        ),

        "effect_size_threshold": (
            EFFECT_SIZE_THRESHOLD
        ),

        "highly_correlated_pairs": int(
            len(redundant_features)
        ),

        "selected_features": (
            selected_features
        ),

        "number_selected_features": int(
            len(selected_features)
        ),

        "sleep_feature_note": (
            "Sleep-related variables are "
            "activity-derived estimates and "
            "should not be interpreted as "
            "clinical sleep-stage measurements."
        )
    }

    report_path = (
        RESULTS_DIR
        / "phase5_analysis_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
            default=str
        )

    return report_path


# ============================================================
# MAIN
# ============================================================

def run_phase5():

    print("=" * 70)
    print(
        "PHASE 5 - STATISTICAL ANALYSIS "
        "& FEATURE SELECTION"
    )
    print("=" * 70)

    create_directories()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING PHASE 3 AND PHASE 4 FEATURES")
    print("-" * 70)

    activity, sleep, cosinor = (
        load_features()
    )

    print(
        f"Activity feature rows: "
        f"{len(activity)}"
    )

    print(
        f"Sleep feature rows: "
        f"{len(sleep)}"
    )

    print(
        f"Cosinor feature rows: "
        f"{len(cosinor)}"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("COMBINING FEATURES")
    print("-" * 70)

    combined = prepare_features(
        activity,
        sleep,
        cosinor
    )

    print(
        f"Combined subjects: "
        f"{len(combined)}"
    )

    print(
        f"Combined feature count: "
        f"{len(combined.columns) - 2}"
    )

    combined_output = (
        RESULTS_DIR
        / "combined_features.csv"
    )

    combined.to_csv(
        combined_output,
        index=False
    )

    # --------------------------------------------------------
    # Feature list
    # --------------------------------------------------------

    features = get_feature_columns(
        combined
    )

    print()
    print(
        f"Numeric features: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("DATA QUALITY ANALYSIS")
    print("-" * 70)

    quality = calculate_data_quality(
        combined,
        features
    )

    quality.to_csv(
        RESULTS_DIR
        / "feature_quality.csv",
        index=False
    )

    print(
        f"Total missing feature values: "
        f"{int(quality['missing'].sum())}"
    )

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("-" * 70)

    descriptive = (
        calculate_descriptive_statistics(
            combined,
            features
        )
    )

    descriptive.to_csv(
        RESULTS_DIR
        / "descriptive_statistics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Statistical tests
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STATISTICAL COMPARISON")
    print("-" * 70)

    statistical_results = (
        perform_statistical_tests(
            combined,
            features
        )
    )

    statistical_results.to_csv(
        RESULTS_DIR
        / "statistical_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CORRELATION ANALYSIS")
    print("-" * 70)

    correlation_matrix = (
        calculate_correlations(
            combined,
            features
        )
    )

    correlation_matrix.to_csv(
        RESULTS_DIR
        / "correlation_matrix.csv"
    )

    redundant_features = (
        identify_redundant_features(
            correlation_matrix
        )
    )

    redundant_features.to_csv(
        RESULTS_DIR
        / "redundant_feature_pairs.csv",
        index=False
    )

    print(
        f"Highly correlated pairs: "
        f"{len(redundant_features)}"
    )

    # --------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("FEATURE SELECTION")
    print("-" * 70)

    selected_features = select_features(
        statistical_results,
        correlation_matrix,
        features
    )

    selected_output = (
        RESULTS_DIR
        / "selected_features.csv"
    )

    pd.DataFrame(
        {
            "feature": selected_features
        }
    ).to_csv(
        selected_output,
        index=False
    )

    print(
        f"Selected features: "
        f"{len(selected_features)}"
    )

    for feature in selected_features:
        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # ML dataset
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATING ML-READY DATASET")
    print("-" * 70)

    ml_dataset = create_ml_dataset(
        combined,
        selected_features
    )

    ml_output = (
        RESULTS_DIR
        / "ml_ready_dataset.csv"
    )

    ml_dataset.to_csv(
        ml_output,
        index=False
    )

    print(
        f"ML dataset shape: "
        f"{ml_dataset.shape}"
    )

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("GENERATING STATISTICAL VISUALIZATIONS")
    print("-" * 70)

    plot_feature_distributions(
        combined,
        statistical_results
    )

    plot_correlation_heatmap(
        correlation_matrix
    )

    plot_effect_sizes(
        statistical_results
    )

    plot_selected_features(
        statistical_results,
        selected_features
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = save_report(
        combined,
        features,
        statistical_results,
        redundant_features,
        selected_features
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 5 COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Subjects: {len(combined)}"
    )

    print(
        f"Features before selection: "
        f"{len(features)}"
    )

    print(
        f"Features selected: "
        f"{len(selected_features)}"
    )

    print()
    print(
        "Combined features:"
    )

    print(
        combined_output
    )

    print()
    print(
        "Statistical comparison:"
    )

    print(
        RESULTS_DIR
        / "statistical_comparison.csv"
    )

    print()
    print(
        "Correlation matrix:"
    )

    print(
        RESULTS_DIR
        / "correlation_matrix.csv"
    )

    print()
    print(
        "Selected features:"
    )

    print(
        selected_output
    )

    print()
    print(
        "ML-ready dataset:"
    )

    print(
        ml_output
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
    run_phase5()
    