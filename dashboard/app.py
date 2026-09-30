from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Biological Rhythm Research Analytics",
    page_icon="◷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"

DEPRESJON_ROOT = (
    DATA_ROOT
    / "processed"
    / "depresjon"
)

PHASE3 = RESULTS_ROOT / "phase3"
PHASE4 = RESULTS_ROOT / "phase4"
PHASE5 = RESULTS_ROOT / "phase5"
PHASE6 = RESULTS_ROOT / "phase6_final"
PHASE7 = RESULTS_ROOT / "phase7"
PHASE8 = RESULTS_ROOT / "phase8"


# ============================================================
# LIGHT SCIENTIFIC UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        border: 1px solid #d9dde3;
        border-radius: 4px;
        padding: 0.8rem;
        background: #ffffff;
    }

    .research-note {
        border-left: 3px solid #536b82;
        background: #f5f7f9;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0;
        color: #3e4852;
        font-size: 0.92rem;
    }

    div[data-baseweb="select"] > div {
        border-radius: 4px !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def read_csv(path):
    """Safely read a CSV file."""

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def first_column(df, candidates):
    """Return the first matching column."""

    for col in candidates:
        if col in df.columns:
            return col

    return None


def fmt_num(value, digits=2):
    """Format numeric value safely."""

    if pd.isna(value):
        return "—"

    return f"{float(value):.{digits}f}"


def fmt_pct(value, digits=1):
    """Format fraction as percentage."""

    if pd.isna(value):
        return "—"

    return f"{float(value) * 100:.{digits}f}%"


def subject_sort_key(value):
    """Natural sort for condition_1, condition_2, etc."""

    text = str(value)

    prefix = "".join(
        c for c in text
        if not c.isdigit()
    )

    digits = "".join(
        c for c in text
        if c.isdigit()
    )

    return (
        prefix,
        int(digits) if digits else 0
    )


# ============================================================
# LOAD ANALYTICAL RESULTS
# ============================================================

@st.cache_data
def load_results():

    activity = read_csv(
        PHASE3 / "activity_features.csv"
    )

    sleep = read_csv(
        PHASE3 / "sleep_features.csv"
    )

    cosinor = read_csv(
        PHASE4 / "cosinor_subject_features.csv"
    )

    hourly_profile = read_csv(
        PHASE3 / "hourly_group_activity_profile.csv"
    )

    statistics = read_csv(
        PHASE5 / "statistical_comparison.csv"
    )

    final_ml = read_csv(
        PHASE8 / "final_ml_summary.csv"
    )

    importance = read_csv(
        PHASE7 / "permutation_importance.csv"
    )

    final_importance = read_csv(
        PHASE8 / "final_feature_importance.csv"
    )

    group_importance = read_csv(
        PHASE7 / "feature_group_importance.csv"
    )

    final_group_importance = read_csv(
        PHASE8 / "final_feature_group_importance.csv"
    )

    return {
        "activity": activity,
        "sleep": sleep,
        "cosinor": cosinor,
        "hourly_profile": hourly_profile,
        "statistics": statistics,
        "final_ml": final_ml,
        "importance": importance,
        "final_importance": final_importance,
        "group_importance": group_importance,
        "final_group_importance": final_group_importance,
    }


DATA = load_results()


# ============================================================
# ORIGINAL 1-MINUTE DATA
# ============================================================

@st.cache_data
def load_minute_data(subject_id):
    """
    Load original cleaned minute-level activity data.

    Example:
    data/processed/depresjon/condition_1/activity_cleaned.csv
    """

    path = (
        DEPRESJON_ROOT
        / str(subject_id)
        / "activity_cleaned.csv"
    )

    if not path.exists():
        return pd.DataFrame()

    df = read_csv(path)

    required = {
        "datetime",
        "activity",
    }

    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    df["activity"] = pd.to_numeric(
        df["activity"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "datetime",
            "activity",
        ]
    )

    df = df.sort_values(
        "datetime"
    )

    return df


# ============================================================
# REAL RESOLUTION AGGREGATION
# ============================================================

@st.cache_data
def create_rhythm_profile(
    subject_id,
    resolution_minutes,
):
    """
    Create a 24-hour average activity profile
    from the original 1-minute data.

    IMPORTANT:
    This changes the actual observed data resolution.

    15 min  -> approximately 96 bins
    30 min  -> approximately 48 bins
    45 min  -> approximately 32 bins
    60 min  -> approximately 24 bins
    90 min  -> approximately 16 bins
    120 min -> approximately 12 bins
    """

    df = load_minute_data(
        subject_id
    )

    if df.empty:
        return pd.DataFrame()

    work = df.copy()

    # --------------------------------------------------------
    # Minutes since midnight
    # --------------------------------------------------------

    work["minute_of_day"] = (
        work["datetime"].dt.hour * 60
        + work["datetime"].dt.minute
        + work["datetime"].dt.second / 60
    )

    # --------------------------------------------------------
    # Assign each observation to a time-of-day bin
    # --------------------------------------------------------

    work["time_bin"] = (
        np.floor(
            work["minute_of_day"]
            / resolution_minutes
        )
        * resolution_minutes
    )

    # --------------------------------------------------------
    # Average all recording days for each time bin
    # --------------------------------------------------------

    profile = (
        work.groupby(
            "time_bin",
            as_index=False
        )["activity"]
        .mean()
        .rename(
            columns={
                "time_bin": "minute_of_day",
                "activity": "activity",
            }
        )
        .sort_values(
            "minute_of_day"
        )
        .reset_index(drop=True)
    )

    # Decimal hour for Plotly
    profile["hour"] = (
        profile["minute_of_day"]
        / 60
    )

    # Human-readable time
    def make_time_label(minutes):

        minutes = int(
            round(minutes)
        )

        if minutes >= 1440:
            return "24:00"

        hours = minutes // 60
        mins = minutes % 60

        return f"{hours:02d}:{mins:02d}"

    profile["time_label"] = (
        profile["minute_of_day"]
        .apply(make_time_label)
    )

    return profile


# ============================================================
# X-AXIS TICKS
# ============================================================

def create_time_ticks(
    interval_minutes
):
    """
    Create x-axis labels according
    to selected resolution.
    """

    values = list(
        np.arange(
            0,
            1440 + interval_minutes,
            interval_minutes,
        )
    )

    labels = []

    for value in values:

        value = int(
            round(value)
        )

        if value >= 1440:

            labels.append(
                "24:00"
            )

        else:

            hour = value // 60
            minute = value % 60

            labels.append(
                f"{hour:02d}:{minute:02d}"
            )

    return (
        [v / 60 for v in values],
        labels,
    )


# ============================================================
# COSINOR
# ============================================================

def get_cosinor_subject(
    subject_id
):

    df = DATA["cosinor"]

    if (
        df.empty
        or "subject_id"
        not in df.columns
    ):
        return None

    rows = df[
        df["subject_id"]
        .astype(str)
        == str(subject_id)
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def create_cosinor_curve(row):

    if row is None:
        return None, None

    mesor = pd.to_numeric(
        row.get("mesor"),
        errors="coerce",
    )

    beta_cos = pd.to_numeric(
        row.get("beta_cos"),
        errors="coerce",
    )

    beta_sin = pd.to_numeric(
        row.get("beta_sin"),
        errors="coerce",
    )

    if (
        pd.isna(mesor)
        or pd.isna(beta_cos)
        or pd.isna(beta_sin)
    ):
        return None, None

    x = np.linspace(
        0,
        24,
        481,
    )

    omega = (
        2 * np.pi / 24
    )

    y = (
        float(mesor)
        + float(beta_cos)
        * np.cos(
            omega * x
        )
        + float(beta_sin)
        * np.sin(
            omega * x
        )
    )

    return x, y


# ============================================================
# PLOT STYLE
# ============================================================

def style_plot(
    fig,
    height=480,
):

    fig.update_layout(
        height=height,
        margin=dict(
            l=60,
            r=30,
            t=70,
            b=55,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        font=dict(
            family="Arial, sans-serif",
            size=13,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e3e6e8",
        zeroline=False,
        linecolor="#cbd1d6",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e3e6e8",
        zeroline=False,
        linecolor="#cbd1d6",
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.title(
    "Automated Biological Rhythm Analysis"
)

st.caption(
    "Clinical research analytics dashboard · Depresjon dataset"
)

st.markdown(
    """
    <div class="research-note">
    Research analytics interface for activity, activity-derived
    sleep estimates, 24-hour Cosinor rhythm analysis, statistical
    comparison, classification and model explainability.
    This dashboard is not a clinical diagnostic system.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

page = st.sidebar.radio(
    "Analysis module",
    [
        "Overview",
        "Subject Explorer",
        "Activity Analysis",
        "Sleep Analysis",
        "Circadian Rhythm",
        "Statistical Analysis",
        "Classification",
        "Explainability",
    ],
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Raw data → cleaning → activity/sleep → "
    "Cosinor → statistics → ML → explainability"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    activity = DATA["activity"]

    subjects = (
        activity["subject_id"]
        .nunique()
        if (
            not activity.empty
            and "subject_id"
            in activity.columns
        )
        else 55
    )

    condition = (
        activity.loc[
            activity["label"]
            .astype(str)
            .str.lower()
            == "condition",
            "subject_id",
        ].nunique()
        if (
            not activity.empty
            and {
                "label",
                "subject_id",
            }.issubset(
                activity.columns
            )
        )
        else 23
    )

    control = (
        activity.loc[
            activity["label"]
            .astype(str)
            .str.lower()
            == "control",
            "subject_id",
        ].nunique()
        if (
            not activity.empty
            and {
                "label",
                "subject_id",
            }.issubset(
                activity.columns
            )
        )
        else 32
    )

    st.subheader(
        "Dataset and model status"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Subjects",
        subjects,
    )

    c2.metric(
        "Condition",
        condition,
    )

    c3.metric(
        "Control",
        control,
    )

    c4.metric(
        "GB accuracy",
        "81.8%",
    )

    c5.metric(
        "GB ROC-AUC",
        "0.895",
    )

    st.markdown(
        "### Research workflow"
    )

    workflow = pd.DataFrame(
        {
            "Stage": [
                "Data preparation",
                "Activity analysis",
                "Sleep estimation",
                "Circadian analysis",
                "Statistical analysis",
                "Classification",
                "Explainability",
                "Validation",
            ],
            "Output": [
                "Cleaned minute-level activity",
                "Activity features",
                "Activity-derived sleep features",
                "24-hour Cosinor parameters",
                "Group comparison and selection",
                "Four classifier comparison",
                "Held-out permutation importance",
                "Integrated validation",
            ],
            "Status": [
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "Complete",
                "10/10 passed",
            ],
        }
    )

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The classification results are exploratory estimates "
        "from 55 subjects and should not be interpreted as "
        "clinical diagnostic performance."
    )


# ============================================================
# SUBJECT EXPLORER
# ============================================================

elif page == "Subject Explorer":

    st.subheader(
        "Subject Explorer"
    )

    activity = DATA["activity"]

    subjects = sorted(
        activity["subject_id"]
        .astype(str)
        .unique(),
        key=subject_sort_key,
    )

    if not subjects:

        st.error(
            "No subjects were found."
        )

        st.stop()

    selected_subject = (
        st.selectbox(
            "Subject",
            subjects,
        )
    )

    subject_features = activity[
        activity["subject_id"]
        .astype(str)
        == selected_subject
    ]

    sleep = DATA["sleep"]

    subject_sleep = sleep[
        sleep["subject_id"]
        .astype(str)
        == selected_subject
    ] if (
        not sleep.empty
        and "subject_id"
        in sleep.columns
    ) else pd.DataFrame()

    cosinor = get_cosinor_subject(
        selected_subject
    )

    label = (
        str(
            subject_features.iloc[0]["label"]
        )
        if (
            not subject_features.empty
            and "label"
            in subject_features.columns
        )
        else "—"
    )

    st.markdown(
        "### Subject summary"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Subject",
        selected_subject,
    )

    c2.metric(
        "Group",
        label.title(),
    )

    c3.metric(
        "Mean activity",
        fmt_num(
            subject_features.iloc[0].get(
                "mean_activity"
            ),
            1,
        )
        if not subject_features.empty
        else "—",
    )

    c4.metric(
        "Activity CV",
        fmt_num(
            subject_features.iloc[0].get(
                "activity_cv"
            ),
            3,
        )
        if not subject_features.empty
        else "—",
    )

    c5.metric(
        "Acrophase",
        (
            f"{fmt_num(cosinor.get('acrophase_hours'), 2)} h"
            if cosinor is not None
            else "—"
        ),
    )

    # ========================================================
    # REAL RESOLUTION CONTROL
    # ========================================================

    st.markdown(
        "### 24-hour activity rhythm"
    )

    resolution = st.selectbox(
        "Activity resolution",
        options=[
            15,
            30,
            45,
            60,
            90,
            120,
        ],
        index=2,
        format_func=lambda x:
            f"{x} minutes",
        help=(
            "This changes the actual aggregation "
            "of the original 1-minute data."
        ),
    )

    st.caption(
        f"Source: original cleaned 1-minute activity data. "
        f"Current visualization resolution: "
        f"{resolution} minutes. "
        "Phase 3–8 analytical results remain based "
        "on the established 1-hour dataset."
    )

    profile = create_rhythm_profile(
        selected_subject,
        resolution,
    )

    if profile.empty:

        st.warning(
            "Minute-level activity data could not "
            f"be loaded for {selected_subject}."
        )

    else:

        fig = go.Figure()

        # ----------------------------------------------------
        # ACTUAL OBSERVED DATA
        # ----------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=profile["hour"],
                y=profile["activity"],
                mode="lines+markers",
                name=(
                    "Observed activity "
                    f"({resolution}-minute average)"
                ),
                line=dict(
                    width=2
                ),
                marker=dict(
                    size=5
                ),
                customdata=profile[
                    "time_label"
                ],
                hovertemplate=(
                    "<b>Time:</b> %{customdata}<br>"
                    "<b>Mean activity:</b> %{y:.1f}"
                    "<extra></extra>"
                ),
            )
        )

        # ----------------------------------------------------
        # COSINOR FIT
        # ----------------------------------------------------

        x_cos, y_cos = (
            create_cosinor_curve(
                cosinor
            )
        )

        if x_cos is not None:

            fig.add_trace(
                go.Scatter(
                    x=x_cos,
                    y=y_cos,
                    mode="lines",
                    name="24-hour Cosinor fit",
                    line=dict(
                        width=3
                    ),
                    hovertemplate=(
                        "<b>Time:</b> %{x:.2f} h<br>"
                        "<b>Fitted activity:</b> %{y:.1f}"
                        "<extra></extra>"
                    ),
                )
            )

        # ----------------------------------------------------
        # MESOR
        # ----------------------------------------------------

        if cosinor is not None:

            mesor = pd.to_numeric(
                cosinor.get(
                    "mesor"
                ),
                errors="coerce",
            )

            acrophase = pd.to_numeric(
                cosinor.get(
                    "acrophase_hours"
                ),
                errors="coerce",
            )

            if not pd.isna(mesor):

                fig.add_hline(
                    y=float(mesor),
                    line_dash="dash",
                    line_width=2,
                    annotation_text=(
                        f"Mesor = "
                        f"{float(mesor):.1f}"
                    ),
                    annotation_position=(
                        "top left"
                    ),
                )

            if not pd.isna(acrophase):

                fig.add_vline(
                    x=float(acrophase),
                    line_dash="dot",
                    line_width=2,
                    annotation_text=(
                        f"Acrophase = "
                        f"{float(acrophase):.2f} h"
                    ),
                    annotation_position="top",
                )

        # ----------------------------------------------------
        # X AXIS
        # ----------------------------------------------------

        tick_values, tick_labels = (
            create_time_ticks(
                resolution
            )
        )

        fig.update_xaxes(
            title="Time of day",
            range=[
                0,
                24,
            ],
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_labels,
        )

        fig.update_yaxes(
            title="Mean activity"
        )

        fig.update_layout(
            title=(
                f"{selected_subject} — "
                "24-Hour Activity Rhythm"
            )
        )

        style_plot(
            fig,
            height=520,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
            },
        )

        st.success(
            f"The observed blue line now contains "
            f"{len(profile)} actual "
            f"{resolution}-minute time-of-day bins."
        )

    # ========================================================
    # SUBJECT FEATURES
    # ========================================================

    st.markdown(
        "### Subject features"
    )

    tables = []

    if not subject_features.empty:

        table = (
            subject_features
            .drop(
                columns=[
                    "subject_id"
                ],
                errors="ignore",
            )
            .T
        )

        table.columns = [
            "Activity"
        ]

        tables.append(
            table
        )

    if not subject_sleep.empty:

        table = (
            subject_sleep
            .drop(
                columns=[
                    "subject_id"
                ],
                errors="ignore",
            )
            .T
        )

        table.columns = [
            "Sleep"
        ]

        tables.append(
            table
        )

    if cosinor is not None:

        table = pd.DataFrame(
            [cosinor.to_dict()]
        ).drop(
            columns=[
                "subject_id",
                "label",
            ],
            errors="ignore",
        ).T

        table.columns = [
            "Cosinor"
        ]

        tables.append(
            table
        )

    if tables:

        st.dataframe(
            pd.concat(
                tables,
                axis=1,
            ),
            use_container_width=True,
        )


# ============================================================
# ACTIVITY ANALYSIS
# ============================================================

elif page == "Activity Analysis":

    st.subheader(
        "Activity Analysis"
    )

    activity = DATA["activity"]

    if activity.empty:
        st.warning(
            "Activity results unavailable."
        )
        st.stop()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Subjects",
        len(activity),
    )

    c2.metric(
        "Mean activity",
        fmt_num(
            pd.to_numeric(
                activity["mean_activity"],
                errors="coerce",
            ).mean(),
            1,
        ),
    )

    c3.metric(
        "Mean activity SD",
        fmt_num(
            pd.to_numeric(
                activity["activity_std"],
                errors="coerce",
            ).mean(),
            1,
        ),
    )

    c4.metric(
        "Mean day/night ratio",
        fmt_num(
            pd.to_numeric(
                activity["day_night_ratio"],
                errors="coerce",
            ).mean(),
            2,
        ),
    )

    st.markdown(
        "### Mean activity distribution"
    )

    fig = go.Figure()

    for label in [
        "condition",
        "control",
    ]:

        values = pd.to_numeric(
            activity.loc[
                activity["label"]
                .astype(str)
                .str.lower()
                == label,
                "mean_activity",
            ],
            errors="coerce",
        ).dropna()

        fig.add_trace(
            go.Box(
                y=values,
                name=label.title(),
                boxmean=True,
                boxpoints="outliers",
            )
        )

    fig.update_layout(
        title="Mean activity by group",
        xaxis_title="Group",
        yaxis_title="Mean activity",
    )

    style_plot(
        fig
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown(
        "### Activity features"
    )

    st.dataframe(
        activity,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SLEEP ANALYSIS
# ============================================================

elif page == "Sleep Analysis":

    st.subheader(
        "Activity-Derived Sleep Analysis"
    )

    sleep = DATA["sleep"]

    if sleep.empty:
        st.warning(
            "Sleep results unavailable."
        )
        st.stop()

    sleep_hours = pd.to_numeric(
        sleep[
            "estimated_sleep_hours"
        ],
        errors="coerce",
    )

    regularity = pd.to_numeric(
        sleep[
            "sleep_regularity"
        ],
        errors="coerce",
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Subjects",
        len(sleep),
    )

    c2.metric(
        "Mean accumulated sleep estimate",
        fmt_num(
            sleep_hours.mean(),
            1,
        ),
    )

    c3.metric(
        "Mean sleep regularity",
        fmt_pct(
            regularity.mean()
        ),
    )

    st.warning(
        "These are activity-derived sleep estimates, not clinical "
        "sleep measurements. In the current Phase 3 implementation, "
        "estimated_sleep_hours is accumulated low-activity nighttime "
        "hours across each subject's recording period."
    )

    st.markdown(
        "### Estimated sleep duration"
    )

    fig = go.Figure()

    for label in [
        "condition",
        "control",
    ]:

        values = pd.to_numeric(
            sleep.loc[
                sleep["label"]
                .astype(str)
                .str.lower()
                == label,
                "estimated_sleep_hours",
            ],
            errors="coerce",
        ).dropna()

        fig.add_trace(
            go.Box(
                y=values,
                name=label.title(),
                boxmean=True,
                boxpoints="outliers",
            )
        )

    fig.update_layout(
        title=(
            "Accumulated activity-derived "
            "sleep hours"
        ),
        xaxis_title="Group",
        yaxis_title=(
            "Accumulated estimated sleep hours"
        ),
    )

    style_plot(fig)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown(
        "### Sleep regularity"
    )

    fig = go.Figure()

    for label in [
        "condition",
        "control",
    ]:

        values = pd.to_numeric(
            sleep.loc[
                sleep["label"]
                .astype(str)
                .str.lower()
                == label,
                "sleep_regularity",
            ],
            errors="coerce",
        ).dropna()

        fig.add_trace(
            go.Box(
                y=values,
                name=label.title(),
                boxmean=True,
                boxpoints="outliers",
            )
        )

    fig.update_layout(
        title=(
            "Activity-derived sleep regularity"
        ),
        xaxis_title="Group",
        yaxis_title=(
            "Fraction of observed night hours "
            "classified as low activity"
        ),
    )

    style_plot(fig)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown(
        "### Sleep features"
    )

    st.dataframe(
        sleep,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CIRCADIAN RHYTHM
# ============================================================

elif page == "Circadian Rhythm":

    st.subheader(
        "24-Hour Cosinor Rhythm Analysis"
    )

    cosinor = DATA["cosinor"]

    if cosinor.empty:
        st.warning(
            "Cosinor results unavailable."
        )
        st.stop()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Mean mesor",
        fmt_num(
            pd.to_numeric(
                cosinor["mesor"],
                errors="coerce",
            ).mean(),
            1,
        ),
    )

    c2.metric(
        "Mean amplitude",
        fmt_num(
            pd.to_numeric(
                cosinor["amplitude"],
                errors="coerce",
            ).mean(),
            1,
        ),
    )

    c3.metric(
        "Mean acrophase",
        (
            f"{fmt_num(pd.to_numeric(cosinor['acrophase_hours'], errors='coerce').mean(), 2)} h"
        ),
    )

    c4.metric(
        "Mean R²",
        fmt_num(
            pd.to_numeric(
                cosinor["r_squared"],
                errors="coerce",
            ).mean(),
            3,
        ),
    )

    st.markdown(
        "### Group Cosinor summary"
    )

    summary = pd.DataFrame(
        {
            "Group": [
                "Condition",
                "Control",
            ],
            "Mesor": [
                162.898494,
                207.958909,
            ],
            "Amplitude": [
                141.139773,
                154.195414,
            ],
            "Relative amplitude": [
                0.850060,
                0.744242,
            ],
            "Acrophase (h)": [
                15.054921,
                14.970276,
            ],
            "R²": [
                0.786764,
                0.791044,
            ],
        }
    )

    st.dataframe(
        summary.style.format(
            {
                "Mesor": "{:.2f}",
                "Amplitude": "{:.2f}",
                "Relative amplitude": "{:.3f}",
                "Acrophase (h)": "{:.2f}",
                "R²": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Subject-level Cosinor features"
    )

    cols = [
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
        "rmse",
        "acrophase_time",
        "trough_time",
    ]

    cols = [
        c for c in cols
        if c in cosinor.columns
    ]

    st.dataframe(
        cosinor[cols],
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Cosinor is a fixed-period 24-hour harmonic model. "
        "A fitted mathematical minimum can be below zero even "
        "when observed activity itself is non-negative."
    )


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

elif page == "Statistical Analysis":

    st.subheader(
        "Statistical Comparison"
    )

    stats = DATA["statistics"]

    if stats.empty:
        st.warning(
            "Statistical results unavailable."
        )
        st.stop()

    fdr_col = first_column(
        stats,
        [
            "fdr_p_value",
            "fdr",
            "adjusted_p_value",
        ],
    )

    effect_col = first_column(
        stats,
        [
            "cohens_d",
            "effect_size",
        ],
    )

    if fdr_col:

        fdr = pd.to_numeric(
            stats[fdr_col],
            errors="coerce",
        )

        significant = stats[
            fdr < 0.05
        ]

    else:

        significant = pd.DataFrame()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Features tested",
        len(stats),
    )

    c2.metric(
        "FDR-significant",
        len(significant),
    )

    c3.metric(
        "Largest |Cohen's d|",
        fmt_num(
            pd.to_numeric(
                stats[effect_col],
                errors="coerce",
            ).abs().max(),
            3,
        )
        if effect_col
        else "—",
    )

    st.markdown(
        "### FDR-significant features"
    )

    if not significant.empty:

        cols = [
            "feature",
            "condition_mean",
            "control_mean",
            "mann_whitney_u",
            "p_value",
            "cohens_d",
            "rank_biserial",
            fdr_col,
        ]

        cols = [
            c for c in cols
            if c in significant.columns
        ]

        st.dataframe(
            significant[
                cols
            ].sort_values(
                fdr_col
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "### Effect sizes"
    )

    if (
        effect_col
        and "feature"
        in stats.columns
    ):

        effect_df = stats[
            [
                "feature",
                effect_col,
            ]
        ].copy()

        effect_df[
            effect_col
        ] = pd.to_numeric(
            effect_df[
                effect_col
            ],
            errors="coerce",
        )

        effect_df = (
            effect_df
            .dropna()
            .sort_values(
                effect_col,
                key=lambda x: x.abs(),
                ascending=False,
            )
            .head(20)
            .sort_values(
                effect_col
            )
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=effect_df[
                    effect_col
                ],
                y=effect_df[
                    "feature"
                ],
                orientation="h",
                name="Cohen's d",
            )
        )

        fig.update_layout(
            title="Cohen's d by feature",
            xaxis_title="Cohen's d",
            yaxis_title="Feature",
        )

        style_plot(
            fig,
            height=max(
                500,
                len(effect_df) * 30,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        "### Statistical table"
    )

    st.dataframe(
        stats,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Phase 5 feature selection was exploratory and was not "
        "performed fully within each cross-validation fold. "
        "Therefore the final model should not be described as "
        "having completely leakage-free feature selection."
    )


# ============================================================
# CLASSIFICATION
# ============================================================

elif page == "Classification":

    st.subheader(
        "Health-State Classification"
    )

    st.write(
        "The final pipeline compares Logistic Regression, "
        "Random Forest, SVM and Gradient Boosting using "
        "stratified 5-fold cross-validation."
    )

    ml = DATA["final_ml"]

    if ml.empty:

        ml = pd.DataFrame(
            {
                "Model": [
                    "Logistic Regression",
                    "Random Forest",
                    "SVM",
                    "Gradient Boosting",
                ],
                "Accuracy": [
                    0.781818,
                    0.709091,
                    0.709091,
                    0.818182,
                ],
                "Balanced accuracy": [
                    0.774762,
                    0.692143,
                    0.700476,
                    0.808095,
                ],
                "Precision": [
                    0.763333,
                    0.686667,
                    0.673333,
                    0.810000,
                ],
                "Recall": [
                    0.740000,
                    0.570000,
                    0.620000,
                    0.740000,
                ],
                "F1": [
                    0.743550,
                    0.614286,
                    0.636667,
                    0.771111,
                ],
                "ROC-AUC": [
                    0.843333,
                    0.847143,
                    0.786190,
                    0.917143,
                ],
            }
        )

    st.dataframe(
        ml,
        use_container_width=True,
        hide_index=True,
    )

    model_col = first_column(
        ml,
        [
            "Model",
            "model",
            "classifier",
        ],
    )

    accuracy_col = first_column(
        ml,
        [
            "Accuracy",
            "accuracy_mean",
            "accuracy",
        ],
    )

    auc_col = first_column(
        ml,
        [
            "ROC-AUC",
            "roc_auc_mean",
            "roc_auc",
        ],
    )

    if (
        model_col
        and accuracy_col
    ):

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=ml[
                    model_col
                ],
                y=pd.to_numeric(
                    ml[
                        accuracy_col
                    ],
                    errors="coerce",
                ),
                name="Accuracy",
            )
        )

        if auc_col:

            fig.add_trace(
                go.Bar(
                    x=ml[
                        model_col
                    ],
                    y=pd.to_numeric(
                        ml[
                            auc_col
                        ],
                        errors="coerce",
                    ),
                    name="ROC-AUC",
                )
            )

        fig.update_layout(
            title=(
                "Cross-validation model comparison"
            ),
            xaxis_title="Model",
            yaxis_title="Score",
            yaxis=dict(
                range=[
                    0,
                    1,
                ]
            ),
            barmode="group",
        )

        style_plot(
            fig
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        "### Final Gradient Boosting evaluation"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        "81.8%",
    )

    c2.metric(
        "Balanced accuracy",
        "80.7%",
    )

    c3.metric(
        "Precision",
        "81.0%",
    )

    c4.metric(
        "Recall",
        "73.9%",
    )

    c5.metric(
        "F1",
        "77.3%",
    )

    st.metric(
        "OOF ROC-AUC",
        "0.895",
    )

    st.info(
        "These are exploratory out-of-fold estimates from 55 "
        "subjects, not clinical diagnostic performance."
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":

    st.subheader(
        "Model Explainability"
    )

    importance = DATA[
        "final_importance"
    ]

    if importance.empty:

        importance = DATA[
            "importance"
        ]

    if importance.empty:

        st.warning(
            "Permutation importance results unavailable."
        )

        st.stop()

    feature_col = first_column(
        importance,
        [
            "feature",
            "Feature",
        ],
    )

    value_col = first_column(
        importance,
        [
            "importance_mean",
            "mean_importance",
            "importance",
            "mean",
        ],
    )

    if (
        feature_col is None
        or value_col is None
    ):

        st.error(
            "Could not identify feature "
            "importance columns."
        )

        st.stop()

    importance = importance.copy()

    importance[value_col] = pd.to_numeric(
        importance[value_col],
        errors="coerce",
    )

    importance = (
        importance
        .dropna(
            subset=[
                value_col
            ]
        )
        .sort_values(
            value_col,
            ascending=False,
        )
    )

    top_feature = (
        importance.iloc[0][
            feature_col
        ]
        if not importance.empty
        else "—"
    )

    top_value = (
        importance.iloc[0][
            value_col
        ]
        if not importance.empty
        else np.nan
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Top feature",
        str(top_feature),
    )

    c2.metric(
        "Importance",
        fmt_num(
            top_value,
            3,
        ),
    )

    c3.metric(
        "Final model",
        "Gradient Boosting",
    )

    st.markdown(
        "### Permutation importance"
    )

    plot_df = (
        importance
        .head(15)
        .sort_values(
            value_col,
            ascending=True,
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_df[
                value_col
            ],
            y=plot_df[
                feature_col
            ],
            orientation="h",
            name="Permutation importance",
        )
    )

    fig.update_layout(
        title=(
            "Held-out feature importance"
        ),
        xaxis_title=(
            "Mean permutation importance"
        ),
        yaxis_title="Feature",
    )

    style_plot(
        fig,
        height=max(
            500,
            len(plot_df) * 32,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown(
        "### Feature-group importance"
    )

    groups = DATA[
        "final_group_importance"
    ]

    if groups.empty:

        groups = DATA[
            "group_importance"
        ]

    if not groups.empty:

        group_col = first_column(
            groups,
            [
                "group",
                "feature_group",
                "Feature group",
            ],
        )

        group_value = first_column(
            groups,
            [
                "mean_abs_importance",
                "mean_importance",
                "importance",
            ],
        )

        if (
            group_col
            and group_value
        ):

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=groups[
                        group_col
                    ],
                    y=pd.to_numeric(
                        groups[
                            group_value
                        ],
                        errors="coerce",
                    ),
                    name=(
                        "Mean absolute importance"
                    ),
                )
            )

            fig.update_layout(
                title=(
                    "Feature-group contribution"
                ),
                xaxis_title="Feature group",
                yaxis_title=(
                    "Mean absolute permutation importance"
                ),
            )

            style_plot(
                fig,
                height=430,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    st.markdown(
        "### Interpretation"
    )

    st.write(
        "`estimated_sleep_hours` had the highest individual "
        "held-out permutation importance in the final analysis."
    )

    st.write(
        "Permutation importance describes predictive contribution "
        "within the evaluated model and dataset. It does not "
        "demonstrate causation."
    )

    st.warning(
        "Sleep features are activity-derived heuristic estimates "
        "and should not be interpreted as validated clinical "
        "sleep measurements."
    )


# ============================================================
# SIDEBAR FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Automated Biological Rhythm Analysis & "
    "Health State Classification"
)

st.sidebar.caption(
    "B.Tech Information Technology research project"
)