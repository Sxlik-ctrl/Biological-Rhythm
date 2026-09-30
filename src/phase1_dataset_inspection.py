"""
Phase 1 - Dataset Inspection
Automated Biological Rhythm Analysis & Health State Classification

Purpose:
    Inspect the raw Depresjon and MMASH datasets without modifying
    the original data.

Important:
    - Raw data is never overwritten.
    - Original sampling resolution is preserved.
    - 1-hour aggregation is created ONLY for visualization.
"""

from pathlib import Path
import zipfile
import shutil
import json
import re

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase1"

DEPRESJON_ZIP = RAW_DIR / "depresjon.zip"
MMASH_ZIP = RAW_DIR / "mmash.zip"


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    """Create required project directories."""

    directories = [
        RAW_DIR,
        EXTRACTED_DIR,
        PROCESSED_DIR,
        RESULTS_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# ZIP EXTRACTION
# ============================================================

def extract_zip(zip_path: Path, output_dir: Path):
    """
    Extract ZIP archive if it has not already been extracted.
    """

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {zip_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"Extracted to: {output_dir}")


# ============================================================
# FILE DISCOVERY
# ============================================================

def discover_files(directory: Path):
    """Find supported data files recursively."""

    supported_extensions = {
        ".csv",
        ".tsv",
        ".xlsx",
        ".xls"
    }

    files = []

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported_extensions:
            files.append(path)

    return sorted(files)


# ============================================================
# FILE READER
# ============================================================

def read_dataset(file_path: Path):
    """Read CSV, TSV, or Excel file."""

    extension = file_path.suffix.lower()

    try:

        if extension == ".csv":

            # First try normal comma-separated CSV
            df = pd.read_csv(file_path)

            # Some datasets may actually be tab-separated
            if len(df.columns) == 1:
                df = pd.read_csv(file_path, sep="\t")

        elif extension == ".tsv":

            df = pd.read_csv(file_path, sep="\t")

        elif extension in {".xlsx", ".xls"}:

            df = pd.read_excel(file_path)

        else:
            return None

        return df

    except Exception as error:

        print(
            f"Could not read {file_path}: {error}"
        )

        return None


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_column(columns, keywords):
    """
    Detect a likely column based on keywords.

    This is only a candidate detector.
    Final decisions must be based on actual dataset inspection.
    """

    for column in columns:

        column_lower = str(column).lower()

        for keyword in keywords:

            if keyword in column_lower:
                return column

    return None


def detect_subject_column(df):
    """Detect likely subject/user identifier."""

    keywords = [
        "subject",
        "subject_id",
        "participant",
        "participant_id",
        "user",
        "user_id",
        "id"
    ]

    return detect_column(df.columns, keywords)


def detect_timestamp_column(df):
    """Detect likely timestamp column."""

    keywords = [
        "timestamp",
        "datetime",
        "date_time",
        "time_stamp"
    ]

    return detect_column(df.columns, keywords)


def detect_date_column(df):
    """Detect likely date column."""

    keywords = [
        "date",
        "day"
    ]

    return detect_column(df.columns, keywords)


def detect_activity_column(df):
    """Detect likely activity column."""

    keywords = [
        "activity",
        "activity_value",
        "vector magnitude",
        "vector_magnitude"
    ]

    return detect_column(df.columns, keywords)


# ============================================================
# TIMESTAMP ANALYSIS
# ============================================================

def analyze_timestamp(df, timestamp_column):
    """
    Analyze timestamp intervals.

    Returns:
        timestamp statistics dictionary
    """

    if timestamp_column is None:

        return {
            "timestamp_detected": False
        }

    timestamps = pd.to_datetime(
        df[timestamp_column],
        errors="coerce"
    )

    valid = timestamps.dropna()

    if len(valid) < 2:

        return {
            "timestamp_detected": True,
            "valid_timestamps": len(valid),
            "sampling_interval_available": False
        }

    valid = valid.sort_values()

    differences = valid.diff().dropna()

    difference_seconds = (
        differences.dt.total_seconds()
    )

    difference_seconds = difference_seconds[
        difference_seconds > 0
    ]

    if len(difference_seconds) == 0:

        return {
            "timestamp_detected": True,
            "valid_timestamps": len(valid),
            "sampling_interval_available": False
        }

    median_interval = difference_seconds.median()

    mean_interval = difference_seconds.mean()

    minimum_interval = difference_seconds.min()

    maximum_interval = difference_seconds.max()

    return {
        "timestamp_detected": True,
        "valid_timestamps": int(len(valid)),
        "start_time": str(valid.min()),
        "end_time": str(valid.max()),
        "duration_days": float(
            (valid.max() - valid.min()).total_seconds()
            / 86400
        ),
        "median_interval_seconds": float(
            median_interval
        ),
        "mean_interval_seconds": float(
            mean_interval
        ),
        "minimum_interval_seconds": float(
            minimum_interval
        ),
        "maximum_interval_seconds": float(
            maximum_interval
        ),
        "interval_1_second_count": int(
            (difference_seconds == 1).sum()
        ),
        "interval_1_minute_count": int(
            (difference_seconds == 60).sum()
        ),
        "interval_1_hour_count": int(
            (difference_seconds == 3600).sum()
        )
    }


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

def analyze_missing_values(df):
    """Calculate missing-value statistics."""

    missing = df.isna().sum()

    missing_percent = (
        missing / len(df) * 100
        if len(df) > 0
        else missing
    )

    result = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_percent": missing_percent.values
    })

    return result


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def analyze_duplicates(df):
    """Analyze duplicate rows."""

    return {
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "duplicate_percentage": float(
            df.duplicated().mean() * 100
        ) if len(df) > 0 else 0.0
    }


# ============================================================
# DATASET INSPECTION
# ============================================================

def inspect_file(file_path: Path):
    """Perform complete inspection of one dataset file."""

    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    df = read_dataset(file_path)

    if df is None:
        return None

    subject_column = detect_subject_column(df)
    timestamp_column = detect_timestamp_column(df)
    date_column = detect_date_column(df)
    activity_column = detect_activity_column(df)

    timestamp_info = analyze_timestamp(
        df,
        timestamp_column
    )

    missing_info = analyze_missing_values(df)

    duplicate_info = analyze_duplicates(df)

    number_of_subjects = None

    if subject_column is not None:

        number_of_subjects = int(
            df[subject_column]
            .dropna()
            .nunique()
        )

    summary = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_extension": file_path.suffix,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(
            map(str, df.columns)
        ),
        "data_types": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "subject_column_candidate": subject_column,
        "timestamp_column_candidate": timestamp_column,
        "date_column_candidate": date_column,
        "activity_column_candidate": activity_column,
        "number_of_subjects": number_of_subjects,
        "duplicate_information": duplicate_info,
        "timestamp_information": timestamp_info
    }

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print(
        f"\nSubject column candidate: "
        f"{subject_column}"
    )

    print(
        f"Timestamp column candidate: "
        f"{timestamp_column}"
    )

    print(
        f"Activity column candidate: "
        f"{activity_column}"
    )

    print(
        f"Duplicate rows: "
        f"{duplicate_info['duplicate_rows']:,}"
    )

    if number_of_subjects is not None:

        print(
            f"Number of subjects: "
            f"{number_of_subjects}"
        )

    if timestamp_info.get(
        "timestamp_detected",
        False
    ):

        print(
            "\nMedian sampling interval: "
            f"{timestamp_info.get('median_interval_seconds')} "
            "seconds"
        )

        print(
            "Recording duration: "
            f"{timestamp_info.get('duration_days', 0):.2f} days"
        )

    # Save missing-value report
    missing_file_name = (
        file_path.stem +
        "_missing_values.csv"
    )

    missing_info.to_csv(
        RESULTS_DIR / missing_file_name,
        index=False
    )

    return summary


# ============================================================
# DEPRESJON LABEL DETECTION
# ============================================================

def determine_depresjon_label(file_path):
    """
    Determine condition/control based on the actual
    Depresjon directory/file naming convention.
    """

    path_text = str(
        file_path
    ).lower()

    if "condition" in path_text:
        return "condition"

    if "control" in path_text:
        return "control"

    return None


def inspect_depresjon_subjects(depresjon_dir):
    """
    Inspect subject-level Depresjon activity files.
    """

    records = []

    csv_files = list(
        depresjon_dir.rglob("*.csv")
    )

    for file_path in csv_files:

        # Skip metadata files for subject activity inspection
        if file_path.name.lower() == "scores.csv":
            continue

        label = determine_depresjon_label(
            file_path
        )

        df = read_dataset(file_path)

        if df is None:
            continue

        timestamp_column = detect_timestamp_column(df)

        if timestamp_column is None:

            # Depresjon may use timestamp/date separately.
            timestamp_column = detect_date_column(df)

        activity_column = detect_activity_column(df)

        timestamp_info = analyze_timestamp(
            df,
            timestamp_column
        )

        records.append({

            "file": file_path.name,

            "label": label,

            "rows": len(df),

            "columns": len(df.columns),

            "timestamp_column": timestamp_column,

            "activity_column": activity_column,

            "missing_values": int(
                df.isna().sum().sum()
            ),

            "duplicate_rows": int(
                df.duplicated().sum()
            ),

            "start_time":
                timestamp_info.get(
                    "start_time"
                ),

            "end_time":
                timestamp_info.get(
                    "end_time"
                ),

            "duration_days":
                timestamp_info.get(
                    "duration_days"
                ),

            "median_interval_seconds":
                timestamp_info.get(
                    "median_interval_seconds"
                )
        })

    result = pd.DataFrame(records)

    output_file = (
        RESULTS_DIR /
        "depresjon_subject_inventory.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    return result


# ============================================================
# HOURLY AGGREGATION
# ============================================================

def create_hourly_activity(
    df,
    timestamp_column,
    activity_column,
    subject_id=None
):
    """
    Create hourly activity data.

    IMPORTANT:
        This does NOT replace the original data.

    It creates a separate visualization dataset.
    """

    if timestamp_column is None:
        raise ValueError(
            "Timestamp column not available."
        )

    if activity_column is None:
        raise ValueError(
            "Activity column not available."
        )

    working = df.copy()

    working[timestamp_column] = pd.to_datetime(
        working[timestamp_column],
        errors="coerce"
    )

    working[activity_column] = pd.to_numeric(
        working[activity_column],
        errors="coerce"
    )

    working = working.dropna(
        subset=[
            timestamp_column,
            activity_column
        ]
    )

    working = working.sort_values(
        timestamp_column
    )

    working = working.set_index(
        timestamp_column
    )

    hourly = working[
        activity_column
    ].resample("1h").agg([
        "mean",
        "median",
        "std",
        "min",
        "max",
        "count"
    ])

    hourly = hourly.reset_index()

    if subject_id is not None:

        hourly["subject_id"] = subject_id

    return hourly


# ============================================================
# DEPRESJON HOURLY DATASET
# ============================================================

def create_depresjon_hourly_dataset(
    depresjon_dir
):
    """
    Create an hourly visualization dataset
    from Depresjon activity files.
    """

    all_hourly = []

    csv_files = list(
        depresjon_dir.rglob("*.csv")
    )

    for file_path in csv_files:

        if file_path.name.lower() == "scores.csv":
            continue

        df = read_dataset(file_path)

        if df is None:
            continue

        timestamp_column = detect_timestamp_column(df)

        if timestamp_column is None:

            timestamp_column = detect_date_column(df)

        activity_column = detect_activity_column(df)

        if (
            timestamp_column is None
            or activity_column is None
        ):
            continue

        # Derive subject identifier from filename.
        subject_id = file_path.stem

        hourly = create_hourly_activity(
            df,
            timestamp_column,
            activity_column,
            subject_id
        )

        hourly["label"] = determine_depresjon_label(
            file_path
        )

        all_hourly.append(hourly)

    if not all_hourly:

        print(
            "No Depresjon activity files "
            "could be converted to hourly data."
        )

        return None

    result = pd.concat(
        all_hourly,
        ignore_index=True
    )

    output_file = (
        PROCESSED_DIR /
        "depresjon_activity_hourly.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        "\nHourly visualization dataset created:"
    )

    print(output_file)

    print(
        f"Rows: {len(result):,}"
    )

    return result


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def create_class_distribution(
    subject_inventory
):
    """Create Depresjon class distribution."""

    if subject_inventory is None:
        return

    distribution = (
        subject_inventory[
            "label"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis("class")
        .reset_index(
            name="subject_count"
        )
    )

    output_file = (
        RESULTS_DIR /
        "depresjon_class_distribution.csv"
    )

    distribution.to_csv(
        output_file,
        index=False
    )

    print(
        "\nClass distribution:"
    )

    print(distribution.to_string(
        index=False
    ))


# ============================================================
# SAVE COMPLETE SUMMARY
# ============================================================

def save_summary(all_summaries):
    """Save all inspection results as JSON."""

    output_file = (
        RESULTS_DIR /
        "dataset_inspection_summary.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_summaries,
            file,
            indent=4,
            default=str
        )

    print(
        f"\nComplete summary saved to:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "AUTOMATED BIOLOGICAL RHYTHM ANALYSIS"
    )
    print(
        "PHASE 1 - DATASET INSPECTION"
    )
    print("=" * 70)

    create_directories()

    # --------------------------------------------------------
    # Extract datasets
    # --------------------------------------------------------

    depresjon_extract = (
        EXTRACTED_DIR /
        "depresjon"
    )

    mmash_extract = (
        EXTRACTED_DIR /
        "mmash"
    )

    extract_zip(
        DEPRESJON_ZIP,
        depresjon_extract
    )

    extract_zip(
        MMASH_ZIP,
        mmash_extract
    )

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    depresjon_files = discover_files(
        depresjon_extract
    )

    mmash_files = discover_files(
        mmash_extract
    )

    print(
        f"\nDepresjon data files found: "
        f"{len(depresjon_files)}"
    )

    print(
        f"MMASH data files found: "
        f"{len(mmash_files)}"
    )

    # --------------------------------------------------------
    # Inspect every file
    # --------------------------------------------------------

    summaries = []

    for file_path in depresjon_files:

        summary = inspect_file(
            file_path
        )

        if summary:
            summary["dataset"] = "Depresjon"
            summaries.append(summary)

    for file_path in mmash_files:

        summary = inspect_file(
            file_path
        )

        if summary:
            summary["dataset"] = "MMASH"
            summaries.append(summary)

    # --------------------------------------------------------
    # Depresjon subject-level analysis
    # --------------------------------------------------------

    print(
        "\n\nInspecting Depresjon "
        "subject-level files..."
    )

    subject_inventory = (
        inspect_depresjon_subjects(
            depresjon_extract
        )
    )

    if subject_inventory is not None:

        print(
            f"Subjects/files inspected: "
            f"{len(subject_inventory)}"
        )

        create_class_distribution(
            subject_inventory
        )

    # --------------------------------------------------------
    # Create hourly visualization data
    # --------------------------------------------------------

    print(
        "\n\nCreating hourly "
        "visualization dataset..."
    )

    create_depresjon_hourly_dataset(
        depresjon_extract
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    save_summary(
        summaries
    )

    print("\n" + "=" * 70)
    print(
        "PHASE 1 INSPECTION COMPLETE"
    )
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Original raw data was not modified."
    )

    print(
        "Hourly data was created separately "
        "for visualization."
    )


if __name__ == "__main__":
    main()