"""
PHASE 3 - ACTIVITY & SLEEP ANALYSIS

Automated Biological Rhythm Analysis & Health State Classification

Purpose:
1. Analyze hourly activity patterns.
2. Calculate subject-level activity features.
3. Estimate activity-derived sleep characteristics.
4. Compare condition and control groups descriptively.
5. Generate 1-hour-resolution visualizations.
6. Preserve the cleaned 1-minute data for later detailed analysis.

Input:
    data/processed/depresjon_hourly_visualization.csv

Detailed 1-minute data:
    data/processed/depresjon/<subject_id>/activity_cleaned.csv

Output:
    results/phase3/
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HOURLY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "depresjon_hourly_visualization.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "depresjon"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase3"
)

PLOTS_DIR = RESULTS_DIR / "plots"


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

# Hours considered daytime.
DAY_START = 7
DAY_END = 22

# Hours considered night.
# 22:00 through 06:59
NIGHT_START = 22
NIGHT_END = 7

# Threshold used for activity-derived sleep estimation.
#
# IMPORTANT:
# This is NOT a clinical sleep-stage classifier.
# It is an activity-based heuristic.
#
# The threshold is calculated relative to each subject's
# activity distribution rather than using one absolute value
# for every subject.
LOW_ACTIVITY_PERCENTILE = 20

# Minimum number of consecutive low-activity hourly periods
# required for an estimated sleep period.
MIN_SLEEP_HOURS = 3


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    """Create Phase 3 output directories."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOAD HOURLY DATA
# ============================================================

def load_hourly_data():
    """Load and validate the Phase 2 hourly dataset."""

    if not HOURLY_FILE.exists():
        raise FileNotFoundError(
            f"Hourly dataset not found:\n{HOURLY_FILE}"
        )

    df = pd.read_csv(HOURLY_FILE)

    required_columns = [
        "datetime",
        "activity_mean",
        "activity_median",
        "activity_std",
        "activity_min",
        "activity_max",
        "observations",
        "subject_id",
        "label",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # Parse datetime.
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    # Remove rows with invalid datetime.
    df = df.dropna(
        subset=["datetime"]
    ).copy()

    # Numeric conversion.
    numeric_columns = [
        "activity_mean",
        "activity_median",
        "activity_std",
        "activity_min",
        "activity_max",
        "observations",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove invalid subject IDs / labels.
    df = df.dropna(
        subset=["subject_id", "label"]
    ).copy()

    # Sort.
    df = df.sort_values(
        ["subject_id", "datetime"]
    ).reset_index(drop=True)

    return df


# ============================================================
# BASIC DATASET SUMMARY
# ============================================================

def create_dataset_summary(df):
    """Create a high-level Phase 3 dataset summary."""

    summary = {
        "total_hourly_records": int(len(df)),
        "unique_subjects": int(df["subject_id"].nunique()),
        "condition_subjects": int(
            df.loc[
                df["label"] == "condition",
                "subject_id"
            ].nunique()
        ),
        "control_subjects": int(
            df.loc[
                df["label"] == "control",
                "subject_id"
            ].nunique()
        ),
        "condition_records": int(
            (df["label"] == "condition").sum()
        ),
        "control_records": int(
            (df["label"] == "control").sum()
        ),
        "start_datetime": str(
            df["datetime"].min()
        ),
        "end_datetime": str(
            df["datetime"].max()
        ),
    }

    return summary


# ============================================================
# TIME FEATURES
# ============================================================

def add_time_features(df):
    """Add hour, date and day/night indicators."""

    data = df.copy()

    data["hour"] = data["datetime"].dt.hour

    data["date"] = (
        data["datetime"]
        .dt
        .date
    )

    data["day_of_week"] = (
        data["datetime"]
        .dt
        .dayofweek
    )

    data["is_daytime"] = (
        (data["hour"] >= DAY_START)
        & (data["hour"] < DAY_END)
    )

    data["is_night"] = ~data["is_daytime"]

    return data


# ============================================================
# SUBJECT ACTIVITY FEATURES
# ============================================================

def calculate_activity_features(subject_df):
    """Calculate subject-level activity features."""

    subject_df = subject_df.copy()

    subject_id = subject_df["subject_id"].iloc[0]
    label = subject_df["label"].iloc[0]

    activity = subject_df["activity_mean"].dropna()

    if len(activity) == 0:
        return {
            "subject_id": subject_id,
            "label": label,
            "hourly_records": 0,
            "mean_activity": np.nan,
            "median_activity": np.nan,
            "activity_std": np.nan,
            "activity_min": np.nan,
            "activity_max": np.nan,
            "total_activity": np.nan,
            "daytime_mean_activity": np.nan,
            "nighttime_mean_activity": np.nan,
            "day_night_ratio": np.nan,
            "most_active_hour": np.nan,
            "least_active_hour": np.nan,
            "activity_cv": np.nan,
        }

    mean_activity = activity.mean()
    std_activity = activity.std()

    # Daytime activity.
    daytime = subject_df.loc[
        subject_df["is_daytime"],
        "activity_mean"
    ].dropna()

    # Nighttime activity.
    nighttime = subject_df.loc[
        subject_df["is_night"],
        "activity_mean"
    ].dropna()

    daytime_mean = (
        daytime.mean()
        if len(daytime) > 0
        else np.nan
    )

    nighttime_mean = (
        nighttime.mean()
        if len(nighttime) > 0
        else np.nan
    )

    if (
        pd.notna(daytime_mean)
        and pd.notna(nighttime_mean)
        and nighttime_mean != 0
    ):
        day_night_ratio = (
            daytime_mean / nighttime_mean
        )
    else:
        day_night_ratio = np.nan

    # Mean activity by hour.
    hourly_profile = (
        subject_df
        .groupby("hour")["activity_mean"]
        .mean()
        .dropna()
    )

    if len(hourly_profile) > 0:
        most_active_hour = int(
            hourly_profile.idxmax()
        )

        least_active_hour = int(
            hourly_profile.idxmin()
        )
    else:
        most_active_hour = np.nan
        least_active_hour = np.nan

    # Coefficient of variation.
    if (
        pd.notna(mean_activity)
        and mean_activity != 0
    ):
        activity_cv = (
            std_activity / mean_activity
        )
    else:
        activity_cv = np.nan

    return {
        "subject_id": subject_id,
        "label": label,
        "hourly_records": int(len(activity)),
        "mean_activity": mean_activity,
        "median_activity": activity.median(),
        "activity_std": std_activity,
        "activity_min": activity.min(),
        "activity_max": activity.max(),
        "total_activity": activity.sum(),
        "daytime_mean_activity": daytime_mean,
        "nighttime_mean_activity": nighttime_mean,
        "day_night_ratio": day_night_ratio,
        "most_active_hour": most_active_hour,
        "least_active_hour": least_active_hour,
        "activity_cv": activity_cv,
    }


# ============================================================
# SLEEP ESTIMATION
# ============================================================

def estimate_sleep_features(subject_df):
    """
    Estimate sleep-related features from low activity.

    IMPORTANT:
    These are activity-derived estimates, not clinical
    polysomnography or medically validated sleep-stage labels.
    """

    subject_df = subject_df.copy()

    subject_id = subject_df["subject_id"].iloc[0]
    label = subject_df["label"].iloc[0]

    data = (
        subject_df[
            [
                "datetime",
                "activity_mean"
            ]
        ]
        .dropna()
        .sort_values("datetime")
        .copy()
    )

    if data.empty:
        return {
            "subject_id": subject_id,
            "label": label,
            "estimated_sleep_hours": np.nan,
            "estimated_wake_hours": np.nan,
            "estimated_sleep_periods": 0,
            "mean_night_activity": np.nan,
            "sleep_activity_threshold": np.nan,
            "sleep_regularity": np.nan,
        }

    # Subject-specific low-activity threshold.
    threshold = data["activity_mean"].quantile(
        LOW_ACTIVITY_PERCENTILE / 100.0
    )

    data["low_activity"] = (
        data["activity_mean"] <= threshold
    )

    # We focus on night hours.
    data["hour"] = data["datetime"].dt.hour

    data["night"] = (
        (data["hour"] >= NIGHT_START)
        | (data["hour"] < NIGHT_END)
    )

    night_data = data[data["night"]].copy()

    # Count low-activity night hours.
    low_activity_hours = int(
        night_data["low_activity"].sum()
    )

    # Approximate total sleep hours.
    estimated_sleep_hours = low_activity_hours

    # Approximate wake hours in observed period.
    observed_hours = len(data)

    estimated_wake_hours = max(
        observed_hours - estimated_sleep_hours,
        0
    )

    # Count consecutive low-activity periods.
    low = data["low_activity"].astype(int)

    groups = (
        low.ne(low.shift())
        .cumsum()
    )

    run_lengths = (
        data.groupby(groups)["low_activity"]
        .agg(
            value="first",
            length="size"
        )
    )

    sleep_periods = run_lengths[
        (run_lengths["value"] == True)
        & (run_lengths["length"] >= MIN_SLEEP_HOURS)
    ]

    estimated_sleep_periods = int(
        len(sleep_periods)
    )

    # Mean night activity.
    if len(night_data) > 0:
        mean_night_activity = (
            night_data["activity_mean"].mean()
        )
    else:
        mean_night_activity = np.nan

    # Sleep regularity:
    #
    # Fraction of observed night hours classified
    # as low activity.
    if len(night_data) > 0:
        sleep_regularity = (
            night_data["low_activity"].mean()
        )
    else:
        sleep_regularity = np.nan

    return {
        "subject_id": subject_id,
        "label": label,
        "estimated_sleep_hours": estimated_sleep_hours,
        "estimated_wake_hours": estimated_wake_hours,
        "estimated_sleep_periods": estimated_sleep_periods,
        "mean_night_activity": mean_night_activity,
        "sleep_activity_threshold": threshold,
        "sleep_regularity": sleep_regularity,
    }


# ============================================================
# BUILD SUBJECT FEATURE TABLES
# ============================================================

def build_subject_features(df):
    """Build activity and sleep feature tables."""

    activity_results = []
    sleep_results = []

    for subject_id, subject_df in df.groupby(
        "subject_id"
    ):

        activity_features = calculate_activity_features(
            subject_df
        )

        sleep_features = estimate_sleep_features(
            subject_df
        )

        activity_results.append(
            activity_features
        )

        sleep_results.append(
            sleep_features
        )

    activity_features_df = pd.DataFrame(
        activity_results
    )

    sleep_features_df = pd.DataFrame(
        sleep_results
    )

    return (
        activity_features_df,
        sleep_features_df
    )


# ============================================================
# GROUP COMPARISON
# ============================================================

def create_group_comparison(
    activity_features,
    sleep_features
):
    """Create descriptive group-level statistics."""

    activity_numeric = [
        "mean_activity",
        "median_activity",
        "activity_std",
        "activity_min",
        "activity_max",
        "total_activity",
        "daytime_mean_activity",
        "nighttime_mean_activity",
        "day_night_ratio",
        "activity_cv",
    ]

    sleep_numeric = [
        "estimated_sleep_hours",
        "estimated_wake_hours",
        "estimated_sleep_periods",
        "mean_night_activity",
        "sleep_activity_threshold",
        "sleep_regularity",
    ]

    combined = activity_features.merge(
        sleep_features,
        on=["subject_id", "label"],
        how="inner"
    )

    records = []

    for label, group in combined.groupby("label"):

        record = {
            "label": label,
            "subjects": int(len(group)),
        }

        for column in (
            activity_numeric
            + sleep_numeric
        ):

            if column in group.columns:

                record[
                    f"{column}_mean"
                ] = group[column].mean()

                record[
                    f"{column}_median"
                ] = group[column].median()

                record[
                    f"{column}_std"
                ] = group[column].std()

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 24-HOUR GROUP ACTIVITY PROFILE
# ============================================================

def create_hourly_group_profile(df):
    """Calculate mean hourly activity for each group."""

    profile = (
        df.groupby(
            ["label", "hour"]
        )["activity_mean"]
        .agg(
            mean="mean",
            median="median",
            std="std",
            count="count"
        )
        .reset_index()
    )

    return profile


# ============================================================
# PLOT 1
# ============================================================

def plot_24_hour_activity(df):
    """Plot overall 24-hour activity profile."""

    profile = (
        df.groupby("hour")["activity_mean"]
        .mean()
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        profile.index,
        profile.values,
        marker="o"
    )

    plt.xticks(
        range(24)
    )

    plt.xlabel(
        "Hour of Day"
    )

    plt.ylabel(
        "Mean Activity"
    )

    plt.title(
        "24-Hour Average Activity Profile"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "01_24_hour_average_activity.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 2
# ============================================================

def plot_condition_control_profile(df):
    """Compare condition and control 24-hour profiles."""

    profile = (
        df.groupby(
            ["label", "hour"]
        )["activity_mean"]
        .mean()
        .reset_index()
    )

    plt.figure(
        figsize=(12, 6)
    )

    for label in [
        "condition",
        "control"
    ]:

        subset = profile[
            profile["label"] == label
        ]

        plt.plot(
            subset["hour"],
            subset["activity_mean"],
            marker="o",
            label=label
        )

    plt.xticks(
        range(24)
    )

    plt.xlabel(
        "Hour of Day"
    )

    plt.ylabel(
        "Mean Activity"
    )

    plt.title(
        "24-Hour Activity Profile: Condition vs Control"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "02_condition_vs_control_24h.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 3
# ============================================================

def plot_day_night_activity(activity_features):
    """Plot daytime and nighttime activity."""

    plot_df = activity_features[
        [
            "label",
            "daytime_mean_activity",
            "nighttime_mean_activity"
        ]
    ].copy()

    grouped = (
        plot_df.groupby("label")
        [
            [
                "daytime_mean_activity",
                "nighttime_mean_activity"
            ]
        ]
        .mean()
    )

    ax = grouped.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_xlabel(
        "Group"
    )

    ax.set_ylabel(
        "Mean Activity"
    )

    ax.set_title(
        "Daytime vs Nighttime Activity"
    )

    ax.legend(
        [
            "Daytime",
            "Nighttime"
        ]
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "03_day_vs_night_activity.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 4
# ============================================================

def plot_activity_distribution(activity_features):
    """Plot subject-level mean activity distribution."""

    groups = []

    for label in [
        "condition",
        "control"
    ]:

        values = activity_features.loc[
            activity_features["label"] == label,
            "mean_activity"
        ].dropna()

        groups.append(values)

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        groups,
        labels=[
            "Condition",
            "Control"
        ]
    )

    plt.ylabel(
        "Mean Activity"
    )

    plt.title(
        "Subject-Level Mean Activity Distribution"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "04_mean_activity_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 5
# ============================================================

def plot_activity_variability(activity_features):
    """Plot subject-level activity variability."""

    groups = []

    for label in [
        "condition",
        "control"
    ]:

        values = activity_features.loc[
            activity_features["label"] == label,
            "activity_std"
        ].dropna()

        groups.append(values)

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        groups,
        labels=[
            "Condition",
            "Control"
        ]
    )

    plt.ylabel(
        "Activity Standard Deviation"
    )

    plt.title(
        "Activity Variability by Group"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "05_activity_variability.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 6
# ============================================================

def plot_activity_heatmap(df):
    """
    Plot average activity by group and hour.

    This is a simple matplotlib heatmap implementation.
    """

    pivot = (
        df.groupby(
            ["label", "hour"]
        )["activity_mean"]
        .mean()
        .unstack(
            level=0
        )
    )

    pivot = pivot.reindex(
        range(24)
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        pivot.T,
        aspect="auto",
        interpolation="nearest"
    )

    plt.xticks(
        range(24),
        range(24)
    )

    plt.yticks(
        range(len(pivot.columns)),
        pivot.columns
    )

    plt.xlabel(
        "Hour of Day"
    )

    plt.ylabel(
        "Group"
    )

    plt.title(
        "Hourly Activity Heatmap"
    )

    plt.colorbar(
        label="Mean Activity"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "06_activity_heatmap.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 7
# ============================================================

def plot_sleep_duration(sleep_features):
    """Plot estimated sleep duration."""

    groups = []

    for label in [
        "condition",
        "control"
    ]:

        values = sleep_features.loc[
            sleep_features["label"] == label,
            "estimated_sleep_hours"
        ].dropna()

        groups.append(values)

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        groups,
        labels=[
            "Condition",
            "Control"
        ]
    )

    plt.ylabel(
        "Estimated Sleep Hours"
    )

    plt.title(
        "Activity-Derived Estimated Sleep Duration"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "07_estimated_sleep_duration.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 8
# ============================================================

def plot_sleep_regularity(sleep_features):
    """Plot activity-derived sleep regularity."""

    groups = []

    for label in [
        "condition",
        "control"
    ]:

        values = sleep_features.loc[
            sleep_features["label"] == label,
            "sleep_regularity"
        ].dropna()

        groups.append(values)

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        groups,
        labels=[
            "Condition",
            "Control"
        ]
    )

    plt.ylabel(
        "Low-Activity Night Fraction"
    )

    plt.title(
        "Activity-Derived Sleep Regularity"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "08_sleep_regularity.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 9
# ============================================================

def plot_sample_subjects(df):
    """Plot representative subjects individually."""

    subjects = []

    # Select a few subjects from each group.
    for label in [
        "condition",
        "control"
    ]:

        label_subjects = (
            df.loc[
                df["label"] == label,
                "subject_id"
            ]
            .drop_duplicates()
            .head(3)
            .tolist()
        )

        subjects.extend(
            label_subjects
        )

    plt.figure(
        figsize=(14, 8)
    )

    for subject_id in subjects:

        subject = df[
            df["subject_id"] == subject_id
        ]

        profile = (
            subject.groupby("hour")[
                "activity_mean"
            ]
            .mean()
        )

        plt.plot(
            profile.index,
            profile.values,
            marker="o",
            label=subject_id
        )

    plt.xticks(
        range(24)
    )

    plt.xlabel(
        "Hour of Day"
    )

    plt.ylabel(
        "Mean Activity"
    )

    plt.title(
        "Representative Subject Activity Profiles"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "09_representative_subject_profiles.png",
        dpi=300
    )

    plt.close()


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    dataset_summary,
    activity_features,
    sleep_features,
    group_comparison
):
    """Save Phase 3 analysis report."""

    report = {
        "phase": "Phase 3 - Activity & Sleep Analysis",
        "dataset": dataset_summary,

        "analysis_settings": {
            "visualization_resolution": "1 hour",
            "day_start_hour": DAY_START,
            "day_end_hour": DAY_END,
            "night_start_hour": NIGHT_START,
            "night_end_hour": NIGHT_END,
            "low_activity_percentile": LOW_ACTIVITY_PERCENTILE,
            "minimum_sleep_period_hours": MIN_SLEEP_HOURS,
        },

        "activity_features": {
            "subjects": int(
                len(activity_features)
            ),
            "features": list(
                activity_features.columns
            ),
        },

        "sleep_features": {
            "subjects": int(
                len(sleep_features)
            ),
            "features": list(
                sleep_features.columns
            ),
            "method": (
                "Activity-derived heuristic based on "
                "subject-specific low activity thresholds. "
                "Not a clinical sleep-stage classifier."
            ),
        },

        "group_comparison": (
            group_comparison
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            )
        ),
    }

    report_path = (
        RESULTS_DIR
        / "phase3_analysis_report.json"
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

def run_phase3():

    print("=" * 70)
    print("PHASE 3 - ACTIVITY & SLEEP ANALYSIS")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Hourly input:")
    print(HOURLY_FILE)

    print()
    print("Visualization resolution: 1 HOUR")

    create_directories()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING HOURLY DATA")
    print("-" * 70)

    df = load_hourly_data()

    print(
        f"Hourly records: {len(df)}"
    )

    print(
        f"Subjects: {df['subject_id'].nunique()}"
    )

    print(
        f"Condition records: "
        f"{(df['label'] == 'condition').sum()}"
    )

    print(
        f"Control records: "
        f"{(df['label'] == 'control').sum()}"
    )

    # --------------------------------------------------------
    # Add time features
    # --------------------------------------------------------

    df = add_time_features(
        df
    )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    dataset_summary = create_dataset_summary(
        df
    )

    # --------------------------------------------------------
    # Subject features
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CALCULATING SUBJECT-LEVEL FEATURES")
    print("-" * 70)

    (
        activity_features,
        sleep_features
    ) = build_subject_features(
        df
    )

    print(
        f"Activity feature rows: "
        f"{len(activity_features)}"
    )

    print(
        f"Sleep feature rows: "
        f"{len(sleep_features)}"
    )

    # --------------------------------------------------------
    # Save subject features
    # --------------------------------------------------------

    activity_path = (
        RESULTS_DIR
        / "activity_features.csv"
    )

    sleep_path = (
        RESULTS_DIR
        / "sleep_features.csv"
    )

    activity_features.to_csv(
        activity_path,
        index=False
    )

    sleep_features.to_csv(
        sleep_path,
        index=False
    )

    # --------------------------------------------------------
    # Group comparison
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATING GROUP COMPARISON")
    print("-" * 70)

    group_comparison = create_group_comparison(
        activity_features,
        sleep_features
    )

    group_path = (
        RESULTS_DIR
        / "group_comparison.csv"
    )

    group_comparison.to_csv(
        group_path,
        index=False
    )

    # --------------------------------------------------------
    # Hourly group profile
    # --------------------------------------------------------

    hourly_profile = create_hourly_group_profile(
        df
    )

    hourly_profile_path = (
        RESULTS_DIR
        / "hourly_group_activity_profile.csv"
    )

    hourly_profile.to_csv(
        hourly_profile_path,
        index=False
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("GENERATING VISUALIZATIONS")
    print("-" * 70)

    plot_24_hour_activity(
        df
    )

    plot_condition_control_profile(
        df
    )

    plot_day_night_activity(
        activity_features
    )

    plot_activity_distribution(
        activity_features
    )

    plot_activity_variability(
        activity_features
    )

    plot_activity_heatmap(
        df
    )

    plot_sleep_duration(
        sleep_features
    )

    plot_sleep_regularity(
        sleep_features
    )

    plot_sample_subjects(
        df
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = save_report(
        dataset_summary,
        activity_features,
        sleep_features,
        group_comparison
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 3 COMPLETED")
    print("=" * 70)

    print()
    print("Activity features:")
    print(activity_path)

    print()
    print("Sleep features:")
    print(sleep_path)

    print()
    print("Group comparison:")
    print(group_path)

    print()
    print("Hourly group profile:")
    print(hourly_profile_path)

    print()
    print("Plots:")
    print(PLOTS_DIR)

    print()
    print("Analysis report:")
    print(report_path)

    print()
    print("=" * 70)


if __name__ == "__main__":
    run_phase3()