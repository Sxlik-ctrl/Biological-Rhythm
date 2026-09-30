"""
PHASE 4 - 24-HOUR COSINOR / BIOLOGICAL RHYTHM ANALYSIS

Automated Biological Rhythm Analysis & Health State Classification

Input:
    data/processed/depresjon_hourly_visualization.csv

Main outputs:
    Mesor
    Amplitude
    Acrophase
    R2
    Rhythm-adjusted maximum
    Rhythm-adjusted minimum
    Peak time
    Trough time

The primary rhythm period is fixed at 24 hours.
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

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "depresjon_hourly_visualization.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase4"
)

PLOTS_DIR = RESULTS_DIR / "plots"


# ============================================================
# COSINOR SETTINGS
# ============================================================

PERIOD_HOURS = 24.0

MIN_OBSERVATIONS = 12

# Number of points used to create smooth fitted curves.
FIT_POINTS = 240


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
# LOAD DATA
# ============================================================

def load_data():
    """Load Phase 2 hourly activity data."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "datetime",
        "activity_mean",
        "subject_id",
        "label",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df["activity_mean"] = pd.to_numeric(
        df["activity_mean"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "datetime",
            "activity_mean",
            "subject_id",
            "label",
        ]
    ).copy()

    df["hour"] = (
        df["datetime"].dt.hour
        + df["datetime"].dt.minute / 60.0
    )

    df = df.sort_values(
        ["subject_id", "datetime"]
    )

    return df


# ============================================================
# COSINOR FIT
# ============================================================

def fit_cosinor(
    time_hours,
    values,
    period=24.0
):
    """
    Fit a single-component fixed-period Cosinor model.

    Model:

        y(t) = M + beta_cos*cos(wt) + beta_sin*sin(wt)

    Amplitude:

        A = sqrt(beta_cos^2 + beta_sin^2)

    Acrophase:

        phi = atan2(beta_sin, beta_cos)

    The final model is:

        y(t) = M + A*cos(wt - phi)
    """

    time_hours = np.asarray(
        time_hours,
        dtype=float
    )

    values = np.asarray(
        values,
        dtype=float
    )

    valid = (
        np.isfinite(time_hours)
        & np.isfinite(values)
    )

    time_hours = time_hours[valid]
    values = values[valid]

    n = len(values)

    if n < MIN_OBSERVATIONS:
        return None

    omega = (
        2.0
        * np.pi
        / period
    )

    radians = omega * time_hours

    X = np.column_stack(
        [
            np.ones(n),
            np.cos(radians),
            np.sin(radians),
        ]
    )

    try:
        coefficients, _, _, _ = np.linalg.lstsq(
            X,
            values,
            rcond=None
        )
    except np.linalg.LinAlgError:
        return None

    mesor = coefficients[0]

    beta_cos = coefficients[1]
    beta_sin = coefficients[2]

    amplitude = np.sqrt(
        beta_cos ** 2
        + beta_sin ** 2
    )

    # Acrophase in radians.
    phase_radians = np.arctan2(
        beta_sin,
        beta_cos
    )

    # Convert to hours.
    acrophase_hours = (
        phase_radians
        / omega
    )

    acrophase_hours %= period

    # Predictions.
    predicted = X @ coefficients

    residuals = (
        values - predicted
    )

    ss_res = np.sum(
        residuals ** 2
    )

    ss_tot = np.sum(
        (values - np.mean(values)) ** 2
    )

    if ss_tot > 0:
        r_squared = (
            1.0
            - ss_res / ss_tot
        )
    else:
        r_squared = np.nan

    rmse = np.sqrt(
        np.mean(
            residuals ** 2
        )
    )

    # Rhythm-adjusted maximum/minimum.
    fitted_max = (
        mesor + amplitude
    )

    fitted_min = (
        mesor - amplitude
    )

    # Peak and trough.
    peak_time = acrophase_hours

    trough_time = (
        acrophase_hours
        + period / 2.0
    ) % period

    return {
        "mesor": float(mesor),
        "amplitude": float(amplitude),
        "acrophase_hours": float(acrophase_hours),
        "trough_hours": float(trough_time),
        "fitted_max": float(fitted_max),
        "fitted_min": float(fitted_min),
        "r_squared": float(r_squared),
        "rmse": float(rmse),
        "n_observations": int(n),
        "beta_cos": float(beta_cos),
        "beta_sin": float(beta_sin),
    }


# ============================================================
# PREDICT COSINOR CURVE
# ============================================================

def predict_cosinor(
    hours,
    mesor,
    amplitude,
    acrophase,
    period=24.0
):
    """Generate fitted Cosinor values."""

    omega = (
        2.0
        * np.pi
        / period
    )

    return (
        mesor
        + amplitude
        * np.cos(
            omega
            * (
                hours
                - acrophase
            )
        )
    )


# ============================================================
# SUBJECT-LEVEL ANALYSIS
# ============================================================

def analyze_subject(
    subject_df
):
    """Fit a 24-hour Cosinor model for one subject."""

    subject_id = (
        subject_df["subject_id"]
        .iloc[0]
    )

    label = (
        subject_df["label"]
        .iloc[0]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Aggregate by hour so the rhythm is represented by a
    # 24-hour activity profile.
    # --------------------------------------------------------

    hourly_profile = (
        subject_df
        .groupby("hour")["activity_mean"]
        .mean()
        .reset_index()
    )

    result = fit_cosinor(
        hourly_profile["hour"].values,
        hourly_profile["activity_mean"].values,
        PERIOD_HOURS
    )

    if result is None:
        return None

    result["subject_id"] = subject_id
    result["label"] = label
    result["hours_available"] = int(
        len(hourly_profile)
    )

    # Reorder fields.
    result = {
        "subject_id": result["subject_id"],
        "label": result["label"],
        "hours_available": result["hours_available"],
        "n_observations": result["n_observations"],
        "mesor": result["mesor"],
        "amplitude": result["amplitude"],
        "acrophase_hours": result["acrophase_hours"],
        "trough_hours": result["trough_hours"],
        "fitted_max": result["fitted_max"],
        "fitted_min": result["fitted_min"],
        "r_squared": result["r_squared"],
        "rmse": result["rmse"],
        "beta_cos": result["beta_cos"],
        "beta_sin": result["beta_sin"],
    }

    return result


# ============================================================
# ALL SUBJECTS
# ============================================================

def analyze_all_subjects(df):

    results = []

    print()
    print("-" * 70)
    print("FITTING 24-HOUR COSINOR MODELS")
    print("-" * 70)

    subjects = list(
        df.groupby("subject_id")
    )

    total = len(subjects)

    for index, (
        subject_id,
        subject_df
    ) in enumerate(
        subjects,
        start=1
    ):

        print(
            f"[{index}/{total}] "
            f"{subject_id}"
        )

        result = analyze_subject(
            subject_df
        )

        if result is not None:
            results.append(
                result
            )

    return pd.DataFrame(
        results
    )


# ============================================================
# FORMAT TIME
# ============================================================

def hours_to_clock(hours):
    """Convert decimal hours to HH:MM."""

    if pd.isna(hours):
        return ""

    total_minutes = int(
        round(hours * 60)
    ) % (24 * 60)

    hour = (
        total_minutes // 60
    )

    minute = (
        total_minutes % 60
    )

    return f"{hour:02d}:{minute:02d}"


# ============================================================
# ADD INTERPRETABLE COLUMNS
# ============================================================

def add_interpretable_columns(
    results
):

    results = results.copy()

    results["acrophase_time"] = (
        results["acrophase_hours"]
        .apply(hours_to_clock)
    )

    results["trough_time"] = (
        results["trough_hours"]
        .apply(hours_to_clock)
    )

    # Relative amplitude.
    results["relative_amplitude"] = (
        results["amplitude"]
        / results["mesor"].abs().replace(
            0,
            np.nan
        )
    )

    return results


# ============================================================
# GROUP SUMMARY
# ============================================================

def create_group_summary(
    results
):

    numeric_features = [
        "mesor",
        "amplitude",
        "relative_amplitude",
        "acrophase_hours",
        "trough_hours",
        "fitted_max",
        "fitted_min",
        "r_squared",
        "rmse",
    ]

    records = []

    for label, group in results.groupby(
        "label"
    ):

        record = {
            "label": label,
            "subjects": int(
                len(group)
            ),
        }

        for feature in numeric_features:

            record[
                f"{feature}_mean"
            ] = group[feature].mean()

            record[
                f"{feature}_median"
            ] = group[feature].median()

            record[
                f"{feature}_std"
            ] = group[feature].std()

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# GROUP RHYTHM PROFILE
# ============================================================

def create_group_rhythm_profiles(
    df
):

    profiles = []

    for label, group in df.groupby(
        "label"
    ):

        profile = (
            group
            .groupby("hour")["activity_mean"]
            .mean()
            .reset_index()
        )

        profile["label"] = label

        profiles.append(
            profile
        )

    return pd.concat(
        profiles,
        ignore_index=True
    )


# ============================================================
# PLOT 1 — OVERALL COSINOR
# ============================================================

def plot_overall_cosinor(
    df,
    results
):

    observed = (
        df.groupby("hour")[
            "activity_mean"
        ]
        .mean()
    )

    fit = fit_cosinor(
        observed.index.values,
        observed.values,
        PERIOD_HOURS
    )

    if fit is None:
        return

    smooth_hours = np.linspace(
        0,
        24,
        FIT_POINTS
    )

    fitted = predict_cosinor(
        smooth_hours,
        fit["mesor"],
        fit["amplitude"],
        fit["acrophase_hours"],
        PERIOD_HOURS
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        observed.index,
        observed.values,
        marker="o",
        label="Observed"
    )

    plt.plot(
        smooth_hours,
        fitted,
        label="24-hour Cosinor fit"
    )

    plt.axhline(
        fit["mesor"],
        linestyle="--",
        label="Mesor"
    )

    plt.axvline(
        fit["acrophase_hours"],
        linestyle=":",
        label="Acrophase"
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
        "Overall 24-Hour Biological Rhythm"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "01_overall_cosinor.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 2 — CONDITION VS CONTROL
# ============================================================

def plot_group_cosinor(
    df,
    results
):

    plt.figure(
        figsize=(12, 6)
    )

    for label in [
        "condition",
        "control"
    ]:

        subset = df[
            df["label"] == label
        ]

        profile = (
            subset
            .groupby("hour")[
                "activity_mean"
            ]
            .mean()
        )

        fit = fit_cosinor(
            profile.index.values,
            profile.values,
            PERIOD_HOURS
        )

        if fit is None:
            continue

        smooth_hours = np.linspace(
            0,
            24,
            FIT_POINTS
        )

        fitted = predict_cosinor(
            smooth_hours,
            fit["mesor"],
            fit["amplitude"],
            fit["acrophase_hours"],
            PERIOD_HOURS
        )

        plt.plot(
            profile.index,
            profile.values,
            marker="o",
            linestyle="--",
            label=f"{label} observed"
        )

        plt.plot(
            smooth_hours,
            fitted,
            label=f"{label} Cosinor"
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
        "24-Hour Cosinor Profiles: Condition vs Control"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "02_condition_control_cosinor.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 3 — MESOR
# ============================================================

def plot_mesor(
    results
):

    groups = []

    labels = []

    for label in [
        "condition",
        "control"
    ]:

        values = results.loc[
            results["label"] == label,
            "mesor"
        ].dropna()

        groups.append(
            values
        )

        labels.append(
            label.capitalize()
        )

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        groups,
        labels=labels
    )

    plt.ylabel(
        "Mesor"
    )

    plt.title(
        "Cosinor Mesor by Group"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "03_mesor_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 4 — AMPLITUDE
# ============================================================

def plot_amplitude(
    results
):

    groups = []
    labels = []

    for label in [
        "condition",
        "control"
    ]:

        values = results.loc[
            results["label"] == label,
            "amplitude"
        ].dropna()

        groups.append(
            values
        )

        labels.append(
            label.capitalize()
        )

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        groups,
        labels=labels
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        "Cosinor Amplitude by Group"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "04_amplitude_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 5 — ACROPHASE
# ============================================================

def plot_acrophase(
    results
):

    groups = []
    labels = []

    for label in [
        "condition",
        "control"
    ]:

        values = results.loc[
            results["label"] == label,
            "acrophase_hours"
        ].dropna()

        groups.append(
            values
        )

        labels.append(
            label.capitalize()
        )

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        groups,
        labels=labels
    )

    plt.ylabel(
        "Acrophase (Hour)"
    )

    plt.title(
        "Cosinor Acrophase by Group"
    )

    plt.ylim(
        0,
        24
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "05_acrophase_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 6 — R2
# ============================================================

def plot_r_squared(
    results
):

    groups = []
    labels = []

    for label in [
        "condition",
        "control"
    ]:

        values = results.loc[
            results["label"] == label,
            "r_squared"
        ].dropna()

        groups.append(
            values
        )

        labels.append(
            label.capitalize()
        )

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        groups,
        labels=labels
    )

    plt.ylabel(
        "R²"
    )

    plt.title(
        "Cosinor Model R² by Group"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "06_r_squared_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 7 — ACROPHASE HISTOGRAM
# ============================================================

def plot_acrophase_histogram(
    results
):

    plt.figure(
        figsize=(12, 6)
    )

    for label in [
        "condition",
        "control"
    ]:

        values = results.loc[
            results["label"] == label,
            "acrophase_hours"
        ].dropna()

        plt.hist(
            values,
            bins=12,
            alpha=0.5,
            label=label
        )

    plt.xlabel(
        "Acrophase (Hour)"
    )

    plt.ylabel(
        "Number of Subjects"
    )

    plt.title(
        "Distribution of Circadian Acrophase"
    )

    plt.xticks(
        range(0, 25, 2)
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "07_acrophase_histogram.png",
        dpi=300
    )

    plt.close()


# ============================================================
# PLOT 8 — SUBJECT FIT EXAMPLES
# ============================================================

def plot_subject_examples(
    df,
    results
):

    selected = []

    for label in [
        "condition",
        "control"
    ]:

        candidates = results[
            results["label"] == label
        ]

        if len(candidates) > 0:
            selected.append(
                candidates.iloc[0]
            )

    for row in selected:

        subject_id = row["subject_id"]

        subject = df[
            df["subject_id"] == subject_id
        ]

        profile = (
            subject
            .groupby("hour")[
                "activity_mean"
            ]
            .mean()
        )

        smooth_hours = np.linspace(
            0,
            24,
            FIT_POINTS
        )

        fitted = predict_cosinor(
            smooth_hours,
            row["mesor"],
            row["amplitude"],
            row["acrophase_hours"],
            PERIOD_HOURS
        )

        plt.figure(
            figsize=(12, 6)
        )

        plt.plot(
            profile.index,
            profile.values,
            marker="o",
            label="Observed"
        )

        plt.plot(
            smooth_hours,
            fitted,
            label="24-hour Cosinor"
        )

        plt.axhline(
            row["mesor"],
            linestyle="--",
            label="Mesor"
        )

        plt.axvline(
            row["acrophase_hours"],
            linestyle=":",
            label="Acrophase"
        )

        plt.xticks(
            range(24)
        )

        plt.xlabel(
            "Hour of Day"
        )

        plt.ylabel(
            "Activity"
        )

        plt.title(
            f"Cosinor Fit - {subject_id}"
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        safe_name = (
            subject_id
            .replace(
                " ",
                "_"
            )
        )

        plt.savefig(
            PLOTS_DIR
            / f"08_cosinor_{safe_name}.png",
            dpi=300
        )

        plt.close()


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    df,
    results,
    group_summary
):

    report = {
        "phase": (
            "Phase 4 - "
            "24-Hour Cosinor Analysis"
        ),

        "input_file": str(
            INPUT_FILE
        ),

        "dataset": {
            "hourly_records": int(
                len(df)
            ),
            "subjects": int(
                df["subject_id"].nunique()
            ),
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
        },

        "cosinor_model": {
            "period_hours": PERIOD_HOURS,
            "minimum_observations": MIN_OBSERVATIONS,
            "model": (
                "Y(t) = M + A*cos(2*pi*t/24 - phi)"
            ),
            "features": [
                "mesor",
                "amplitude",
                "acrophase_hours",
                "trough_hours",
                "fitted_max",
                "fitted_min",
                "r_squared",
                "rmse",
            ],
        },

        "successful_subject_fits": int(
            len(results)
        ),

        "group_summary": (
            group_summary
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            )
        ),
    }

    path = (
        RESULTS_DIR
        / "phase4_cosinor_report.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
            default=str
        )

    return path


# ============================================================
# MAIN
# ============================================================

def run_phase4():

    print("=" * 70)
    print("PHASE 4 - 24-HOUR COSINOR ANALYSIS")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Input:")
    print(INPUT_FILE)

    print()
    print(
        f"Fixed rhythm period: "
        f"{PERIOD_HOURS} hours"
    )

    create_directories()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LOADING DATA")
    print("-" * 70)

    df = load_data()

    print(
        f"Hourly records: {len(df)}"
    )

    print(
        f"Subjects: "
        f"{df['subject_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Fit models
    # --------------------------------------------------------

    results = analyze_all_subjects(
        df
    )

    if results.empty:
        raise RuntimeError(
            "No subjects could be fitted."
        )

    # --------------------------------------------------------
    # Add interpretable fields
    # --------------------------------------------------------

    results = add_interpretable_columns(
        results
    )

    # --------------------------------------------------------
    # Save subject features
    # --------------------------------------------------------

    subject_output = (
        RESULTS_DIR
        / "cosinor_subject_features.csv"
    )

    results.to_csv(
        subject_output,
        index=False
    )

    # --------------------------------------------------------
    # Group summary
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CREATING GROUP SUMMARY")
    print("-" * 70)

    group_summary = create_group_summary(
        results
    )

    group_output = (
        RESULTS_DIR
        / "cosinor_group_summary.csv"
    )

    group_summary.to_csv(
        group_output,
        index=False
    )

    # --------------------------------------------------------
    # Group profiles
    # --------------------------------------------------------

    profiles = create_group_rhythm_profiles(
        df
    )

    profile_output = (
        RESULTS_DIR
        / "group_rhythm_profiles.csv"
    )

    profiles.to_csv(
        profile_output,
        index=False
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("GENERATING COSINOR VISUALIZATIONS")
    print("-" * 70)

    plot_overall_cosinor(
        df,
        results
    )

    plot_group_cosinor(
        df,
        results
    )

    plot_mesor(
        results
    )

    plot_amplitude(
        results
    )

    plot_acrophase(
        results
    )

    plot_r_squared(
        results
    )

    plot_acrophase_histogram(
        results
    )

    plot_subject_examples(
        df,
        results
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = save_report(
        df,
        results,
        group_summary
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 4 COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Subjects fitted: "
        f"{len(results)}"
    )

    print()
    print("Subject Cosinor features:")
    print(subject_output)

    print()
    print("Group summary:")
    print(group_output)

    print()
    print("Group rhythm profiles:")
    print(profile_output)

    print()
    print("Plots:")
    print(PLOTS_DIR)

    print()
    print("Report:")
    print(report_path)

    print()
    print("=" * 70)


if __name__ == "__main__":
    run_phase4()