"""
PHASE 2 - DEPRESJON DATA PREPROCESSING

Dataset structure:

data/
└── extracted/
    └── depresjon/
        └── data/
            ├── scores.csv
            ├── condition/
            │   ├── condition_1.csv
            │   └── ...
            └── control/
                ├── control_1.csv
                └── ...

Outputs:

data/
└── processed/
    ├── depresjon/
    │   ├── condition_1/
    │   │   ├── activity_cleaned.csv
    │   │   └── activity_hourly.csv
    │   └── ...
    │
    └── depresjon_hourly_visualization.csv

Rules:
- Original data is never modified.
- Duplicate timestamps are removed.
- Invalid timestamps are removed.
- Invalid activity values are handled.
- Only complete gaps <= 3 minutes are interpolated.
- Longer gaps remain missing.
- Original cleaned resolution remains 1 minute.
- Graph/visualization resolution is 1 hour.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

EXTRACTED_DIR = DATA_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed"

DEPRESJON_INPUT_DIR = (
    EXTRACTED_DIR
    / "depresjon"
    / "data"
)

DEPRESJON_OUTPUT_DIR = (
    PROCESSED_DIR
    / "depresjon"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase2"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "preprocessing_summary.csv"
)

DETAIL_FILE = (
    RESULTS_DIR
    / "preprocessing_detailed_report.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_INTERPOLATION_GAP_MINUTES = 3

HOURLY_FREQUENCY = "1h"


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():

    DEPRESJON_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FIND SUBJECT CSV FILES
# ============================================================

def find_subject_files():

    files = []

    condition_dir = (
        DEPRESJON_INPUT_DIR
        / "condition"
    )

    control_dir = (
        DEPRESJON_INPUT_DIR
        / "control"
    )

    # --------------------------------------------------------
    # Condition subjects
    # --------------------------------------------------------

    if condition_dir.exists():

        files.extend(
            sorted(
                condition_dir.glob(
                    "condition_*.csv"
                )
            )
        )

    # --------------------------------------------------------
    # Control subjects
    # --------------------------------------------------------

    if control_dir.exists():

        files.extend(
            sorted(
                control_dir.glob(
                    "control_*.csv"
                )
            )
        )

    return files


# ============================================================
# DETERMINE SUBJECT ID
# ============================================================

def get_subject_id(file_path):

    return file_path.stem


# ============================================================
# DETERMINE LABEL
# ============================================================

def get_label(file_path):

    name = file_path.stem.lower()

    if name.startswith("condition_"):
        return "condition"

    if name.startswith("control_"):
        return "control"

    return "unknown"


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path):

    df = pd.read_csv(
        file_path
    )

    if df.empty:

        raise ValueError(
            f"CSV is empty: {file_path}"
        )

    return df


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_timestamp_column(df):

    candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
        "date_time",
        "time_stamp"
    ]

    column_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    # Exact matches
    for candidate in candidates:

        if candidate in column_map:

            return column_map[
                candidate
            ]

    # Keyword search
    for col in df.columns:

        name = str(col).lower()

        if (
            "timestamp" in name
            or "datetime" in name
        ):

            return col

    return None


def find_activity_column(df):

    candidates = [
        "activity",
        "activity_count",
        "activity_counts",
        "activity_value",
        "value",
        "counts"
    ]

    column_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    # Exact matches
    for candidate in candidates:

        if candidate in column_map:

            return column_map[
                candidate
            ]

    # Keyword search
    for col in df.columns:

        name = str(col).lower()

        if "activity" in name:

            return col

    # Numeric fallback
    numeric_columns = (
        df
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    if numeric_columns:

        return numeric_columns[0]

    return None


# ============================================================
# TIMESTAMP PROCESSING
# ============================================================

def process_timestamps(
    df,
    timestamp_column
):

    df = df.copy()

    df["datetime"] = (
        pd.to_datetime(
            df[timestamp_column],
            errors="coerce"
        )
    )

    invalid_timestamps = int(
        df["datetime"]
        .isna()
        .sum()
    )

    df = df.dropna(
        subset=[
            "datetime"
        ]
    )

    df = df.sort_values(
        "datetime"
    )

    return (
        df,
        invalid_timestamps
    )


# ============================================================
# ACTIVITY PROCESSING
# ============================================================

def process_activity(
    df,
    activity_column
):

    df = df.copy()

    df["activity"] = (
        pd.to_numeric(
            df[activity_column],
            errors="coerce"
        )
    )

    invalid_activity = int(
        df["activity"]
        .isna()
        .sum()
    )

    # Negative activity values
    negative_mask = (
        df["activity"] < 0
    )

    negative_count = int(
        negative_mask.sum()
    )

    if negative_count > 0:

        df.loc[
            negative_mask,
            "activity"
        ] = np.nan

    invalid_activity += (
        negative_count
    )

    return (
        df,
        invalid_activity
    )


# ============================================================
# DUPLICATE TIMESTAMPS
# ============================================================

def remove_duplicates(df):

    duplicate_count = int(
        df["datetime"]
        .duplicated()
        .sum()
    )

    df = (
        df.drop_duplicates(
            subset=[
                "datetime"
            ],
            keep="first"
        )
    )

    return (
        df,
        duplicate_count
    )


# ============================================================
# GAP STATISTICS
# ============================================================

def calculate_gap_statistics(df):

    if len(df) < 2:

        return {
            "largest_gap_minutes": 0,
            "gaps_over_5_minutes": 0,
            "gaps_over_30_minutes": 0,
            "gaps_over_60_minutes": 0
        }

    timestamps = (
        pd.to_datetime(
            df["datetime"]
        )
        .sort_values()
    )

    differences = (
        timestamps
        .diff()
        .dt.total_seconds()
        / 60
    )

    gaps = differences[
        differences > 1
    ]

    if gaps.empty:

        largest_gap = 1

    else:

        largest_gap = float(
            gaps.max()
        )

    return {
        "largest_gap_minutes":
            largest_gap,

        "gaps_over_5_minutes":
            int(
                (gaps > 5).sum()
            ),

        "gaps_over_30_minutes":
            int(
                (gaps > 30).sum()
            ),

        "gaps_over_60_minutes":
            int(
                (gaps > 60).sum()
            )
    }


# ============================================================
# SHORT GAP INTERPOLATION
# ============================================================

def interpolate_short_gaps(df):

    df = df.copy()

    if df.empty:

        return (
            df,
            0,
            0,
            0
        )

    df = df.sort_values(
        "datetime"
    )

    df = df[
        [
            "datetime",
            "activity"
        ]
    ]

    df = df.set_index(
        "datetime"
    )

    # --------------------------------------------------------
    # Create 1-minute timeline
    # --------------------------------------------------------

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="1min"
    )

    df = df.reindex(
        full_index
    )

    df.index.name = "datetime"

    # --------------------------------------------------------
    # Missing before interpolation
    # --------------------------------------------------------

    missing_before = int(
        df["activity"]
        .isna()
        .sum()
    )

    # --------------------------------------------------------
    # Identify consecutive missing groups
    # --------------------------------------------------------

    missing_mask = (
        df["activity"]
        .isna()
    )

    group_id = (
        missing_mask
        .ne(
            missing_mask.shift()
        )
        .cumsum()
    )

    missing_groups = (
        df.loc[
            missing_mask
        ]
        .groupby(
            group_id[
                missing_mask
            ]
        )
        .size()
    )

    # --------------------------------------------------------
    # Only gaps <= 3 minutes
    # --------------------------------------------------------

    allowed_groups = (
        missing_groups[
            missing_groups
            <= MAX_INTERPOLATION_GAP_MINUTES
        ]
        .index
    )

    allowed_mask = (
        missing_mask
        &
        group_id.isin(
            allowed_groups
        )
    )

    # --------------------------------------------------------
    # Candidate interpolation
    # --------------------------------------------------------

    interpolated = (
        df["activity"]
        .interpolate(
            method="time"
        )
    )

    # --------------------------------------------------------
    # Apply ONLY approved gaps
    # --------------------------------------------------------

    df.loc[
        allowed_mask,
        "activity"
    ] = interpolated.loc[
        allowed_mask
    ]

    # --------------------------------------------------------
    # Missing after
    # --------------------------------------------------------

    missing_after = int(
        df["activity"]
        .isna()
        .sum()
    )

    interpolated_values = (
        missing_before
        - missing_after
    )

    df = df.reset_index()

    return (
        df,
        missing_before,
        missing_after,
        interpolated_values
    )


# ============================================================
# HOURLY AGGREGATION
# ============================================================

def create_hourly_data(df):
    """
    Aggregate cleaned 1-minute activity data into 1-hour resolution.

    The original cleaned 1-minute data is retained separately.
    Only the visualization/summary data is reduced to hourly resolution.
    """

    if df.empty:
        return pd.DataFrame()

    data = df.copy()

    # Make sure datetime is the index
    if "datetime" in data.columns:
        data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
        data = data.dropna(subset=["datetime"])
        data = data.set_index("datetime")

    elif not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Could not find a valid datetime column/index.")

    # Make sure activity column exists
    if "activity" not in data.columns:
        raise ValueError(
            f"Activity column not found. Available columns: {list(data.columns)}"
        )

    # Convert activity to numeric
    data["activity"] = pd.to_numeric(
        data["activity"],
        errors="coerce"
    )

    # Explicit hourly aggregation
    hourly = data["activity"].resample("1h").agg(
        activity_mean="mean",
        activity_median="median",
        activity_std="std",
        activity_min="min",
        activity_max="max",
        observations="count"
    )

    # Reset index
    hourly = hourly.reset_index()

    return hourly    


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(
    file_path
):

    subject_id = (
        get_subject_id(
            file_path
        )
    )

    label = (
        get_label(
            file_path
        )
    )

    print(
        f"\nProcessing "
        f"{subject_id} "
        f"({label})"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data(
        file_path
    )

    rows_before = len(df)

    # --------------------------------------------------------
    # Detect columns
    # --------------------------------------------------------

    timestamp_column = (
        find_timestamp_column(
            df
        )
    )

    activity_column = (
        find_activity_column(
            df
        )
    )

    if timestamp_column is None:

        raise ValueError(
            "Could not detect timestamp "
            f"column.\n"
            f"Available columns: "
            f"{df.columns.tolist()}"
        )

    if activity_column is None:

        raise ValueError(
            "Could not detect activity "
            f"column.\n"
            f"Available columns: "
            f"{df.columns.tolist()}"
        )

    # --------------------------------------------------------
    # Process timestamps
    # --------------------------------------------------------

    (
        df,
        invalid_timestamps
    ) = process_timestamps(
        df,
        timestamp_column
    )

    # --------------------------------------------------------
    # Process activity
    # --------------------------------------------------------

    (
        df,
        invalid_activity
    ) = process_activity(
        df,
        activity_column
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    (
        df,
        duplicate_count
    ) = remove_duplicates(
        df
    )

    # --------------------------------------------------------
    # Gap statistics
    # --------------------------------------------------------

    gap_stats = (
        calculate_gap_statistics(
            df
        )
    )

    # --------------------------------------------------------
    # Regular 1-minute timeline
    # --------------------------------------------------------

    (
        cleaned,
        missing_before,
        missing_after,
        interpolated_values
    ) = interpolate_short_gaps(
        df[
            [
                "datetime",
                "activity"
            ]
        ]
    )

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    cleaned["subject_id"] = (
        subject_id
    )

    cleaned["label"] = (
        label
    )

    # --------------------------------------------------------
    # Create hourly dataset
    # --------------------------------------------------------

    hourly = create_hourly_data(
        cleaned[
            [
                "datetime",
                "activity"
            ]
        ]
    )

    hourly["subject_id"] = (
        subject_id
    )

    hourly["label"] = (
        label
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    subject_output_dir = (
        DEPRESJON_OUTPUT_DIR
        / subject_id
    )

    subject_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save cleaned 1-minute data
    # --------------------------------------------------------

    cleaned.to_csv(
        subject_output_dir
        / "activity_cleaned.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save hourly data
    # --------------------------------------------------------

    hourly.to_csv(
        subject_output_dir
        / "activity_hourly.csv",
        index=False
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {

        "subject_id":
            subject_id,

        "label":
            label,

        "source_file":
            str(file_path),

        "timestamp_column":
            str(timestamp_column),

        "activity_column":
            str(activity_column),

        "rows_before":
            int(rows_before),

        "rows_after_cleaning":
            int(len(cleaned)),

        "hourly_rows":
            int(len(hourly)),

        "invalid_timestamps":
            int(invalid_timestamps),

        "invalid_activity_values":
            int(invalid_activity),

        "duplicate_timestamps_removed":
            int(duplicate_count),

        "missing_before_interpolation":
            int(missing_before),

        "missing_after_interpolation":
            int(missing_after),

        "interpolated_values":
            int(interpolated_values),

        "largest_gap_minutes":
            float(
                gap_stats[
                    "largest_gap_minutes"
                ]
            ),

        "gaps_over_5_minutes":
            int(
                gap_stats[
                    "gaps_over_5_minutes"
                ]
            ),

        "gaps_over_30_minutes":
            int(
                gap_stats[
                    "gaps_over_30_minutes"
                ]
            ),

        "gaps_over_60_minutes":
            int(
                gap_stats[
                    "gaps_over_60_minutes"
                ]
            )
    }

    return (
        cleaned,
        hourly,
        report
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_preprocessing():

    print(
        "=" * 70
    )

    print(
        "PHASE 2 - DATA PREPROCESSING"
    )

    print(
        "=" * 70
    )

    print(
        "\nProject root:"
    )

    print(
        PROJECT_ROOT
    )

    print(
        "\nInput directory:"
    )

    print(
        DEPRESJON_INPUT_DIR
    )

    print(
        "\nOutput directory:"
    )

    print(
        DEPRESJON_OUTPUT_DIR
    )

    print(
        "\nMaximum interpolation gap: "
        f"{MAX_INTERPOLATION_GAP_MINUTES} minutes"
    )

    print(
        "\nVisualization resolution: "
        "1 HOUR"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    subject_files = (
        find_subject_files()
    )

    print(
        f"\nSubject files found: "
        f"{len(subject_files)}"
    )

    if len(subject_files) == 0:

        raise RuntimeError(
            "No Depresjon subject CSV files "
            "were found."
        )

    # --------------------------------------------------------
    # Lists
    # --------------------------------------------------------

    reports = []

    hourly_datasets = []

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process every subject
    # --------------------------------------------------------

    for index, file_path in enumerate(
        subject_files,
        start=1
    ):

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"[{index}/{len(subject_files)}]"
        )

        try:

            (
                cleaned,
                hourly,
                report
            ) = process_subject(
                file_path
            )

            reports.append(
                report
            )

            hourly_datasets.append(
                hourly
            )

            successful += 1

            print(
                f"    Original rows: "
                f"{report['rows_before']}"
            )

            print(
                f"    Cleaned 1-minute rows: "
                f"{report['rows_after_cleaning']}"
            )

            print(
                f"    Hourly rows: "
                f"{report['hourly_rows']}"
            )

            print(
                f"    Interpolated values: "
                f"{report['interpolated_values']}"
            )

            print(
                f"    Remaining missing: "
                f"{report['missing_after_interpolation']}"
            )

            print(
                f"    Largest gap: "
                f"{report['largest_gap_minutes']} minutes"
            )

        except Exception as error:

            failed += 1

            print(
                f"    ERROR: {error}"
            )

            reports.append(
                {
                    "subject_id":
                        file_path.stem,

                    "label":
                        get_label(
                            file_path
                        ),

                    "error":
                        str(error)
                }
            )

    # ========================================================
    # COMBINE HOURLY DATA
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CREATING COMBINED HOURLY DATA"
    )

    print(
        "=" * 70
    )

    if hourly_datasets:

        combined_hourly = pd.concat(
            hourly_datasets,
            ignore_index=True
        )

        combined_hourly = (
            combined_hourly
            .sort_values(
                [
                    "subject_id",
                    "datetime"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        visualization_file = (
            PROCESSED_DIR
            / "depresjon_hourly_visualization.csv"
        )

        combined_hourly.to_csv(
            visualization_file,
            index=False
        )

        print(
            "\nHourly visualization file:"
        )

        print(
            visualization_file
        )

        print(
            "\nTotal hourly records:"
        )

        print(
            len(combined_hourly)
        )

        print(
            "\nColumns:"
        )

        print(
            combined_hourly.columns.tolist()
        )

    else:

        combined_hourly = (
            pd.DataFrame()
        )

        print(
            "\nNo hourly datasets generated."
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df = (
        pd.DataFrame(
            reports
        )
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # ========================================================
    # SAVE DETAILED JSON
    # ========================================================

    detailed_report = {

        "project":
            "Automated Biological Rhythm "
            "Analysis & Health State Classification",

        "phase":
            "Phase 2 - Data Preprocessing",

        "subjects_found":
            len(subject_files),

        "subjects_successful":
            successful,

        "subjects_failed":
            failed,

        "cleaned_resolution":
            "1 minute",

        "visualization_resolution":
            "1 hour",

        "interpolation_rule":
            "Only complete gaps <= 3 minutes "
            "are interpolated.",

        "long_gap_policy":
            "Longer gaps remain missing "
            "and are not artificially reconstructed.",

        "hourly_visualization_file":
            str(
                PROCESSED_DIR
                / "depresjon_hourly_visualization.csv"
            ),

        "subjects":
            reports
    }

    with open(
        DETAIL_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            detailed_report,
            file,
            indent=4,
            default=str
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHASE 2 COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSubject files found: "
        f"{len(subject_files)}"
    )

    print(
        f"Successfully processed: "
        f"{successful}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_FILE
    )

    print(
        "\nDetailed report:"
    )

    print(
        DETAIL_FILE
    )

    if not combined_hourly.empty:

        print(
            "\nHourly visualization:"
        )

        print(
            PROCESSED_DIR
            / "depresjon_hourly_visualization.csv"
        )

        print(
            "\nTotal hourly records:"
        )

        print(
            len(combined_hourly)
        )

    print(
        "\nGraphs will use 1-hour data."
    )

    print(
        "Cleaned 1-minute data is retained "
        "for detailed analysis."
    )

    print(
        "\n"
        + "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning
    )

    run_preprocessing()