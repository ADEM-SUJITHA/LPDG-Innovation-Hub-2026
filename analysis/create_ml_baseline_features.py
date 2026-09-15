import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
]

weeks = pd.date_range(
    "2025-08-25",
    "2026-01-26",
    freq="7D"
)

print("Loading telemetry...")

files = sorted(
    (DATA_DIR / "telemetry").glob("month=*/*.parquet")
)

frames = []

for file in files:

    df = pd.read_parquet(
        file,
        columns=[
            "gateway_id",
            "ts_utc",
            *METRICS
        ]
    )

    df["ts_utc"] = pd.to_datetime(
        df["ts_utc"],
        errors="coerce",
        utc=True
    )

    df["gateway_id"] = (
        df["gateway_id"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
    )

    frames.append(df)

telemetry = pd.concat(
    frames,
    ignore_index=True
)

print("Telemetry rows:", len(telemetry))
print(
    "Gateways:",
    telemetry["gateway_id"].nunique()
)

# ---------------------------------------------------------
# Build baseline-style features
# ---------------------------------------------------------

rows = []

for week in weeks:

    end = pd.Timestamp(
        week,
        tz="UTC"
    )

    baseline_start = (
        end - pd.Timedelta(days=28)
    )

    recent_start = (
        end - pd.Timedelta(days=7)
    )

    baseline = telemetry[
        (telemetry["ts_utc"] >= baseline_start)
        & (telemetry["ts_utc"] < end)
    ].copy()

    recent = telemetry[
        (telemetry["ts_utc"] >= recent_start)
        & (telemetry["ts_utc"] < end)
    ].copy()

    if baseline.empty or recent.empty:
        continue

    # 28-day gateway statistics
    stats = (
        baseline
        .groupby("gateway_id")[METRICS]
        .agg(["mean", "std"])
    )

    # Recent 7-day observations
    for metric in METRICS:

        mean = recent["gateway_id"].map(
            stats[(metric, "mean")]
        )

        std = recent["gateway_id"].map(
            stats[(metric, "std")]
        )

        std = std.replace(0, np.nan)

        exceeded = (
            recent[metric] - mean
        ) > 3.0 * std

        recent.loc[
            exceeded.fillna(False),
            f"{metric}_flag"
        ] = 1

    flag_columns = [
        f"{metric}_flag"
        for metric in METRICS
    ]

    for column in flag_columns:
        recent[column] = (
            recent[column]
            .fillna(0)
        )

    grouped = (
        recent
        .groupby("gateway_id")
        .agg(
            offline_flag_hours=(
                "offline_duration_sec_flag",
                "sum"
            ),
            disconnect_flag_hours=(
                "disconnection_cnt_flag",
                "sum"
            ),
            reboot_flag_hours=(
                "reboot_cnt_flag",
                "sum"
            ),
            recent_offline_mean=(
                "offline_duration_sec",
                "mean"
            ),
            recent_disconnect_mean=(
                "disconnection_cnt",
                "mean"
            ),
            recent_reboot_mean=(
                "reboot_cnt",
                "mean"
            ),
        )
        .reset_index()
    )

    grouped["total_flagged_hours"] = (
        grouped["offline_flag_hours"]
        + grouped["disconnect_flag_hours"]
        + grouped["reboot_flag_hours"]
    )

    grouped["week_start"] = week.strftime(
        "%Y-%m-%d"
    )

    rows.append(grouped)


features = pd.concat(
    rows,
    ignore_index=True
)

# ---------------------------------------------------------
# Add labels
# ---------------------------------------------------------

print("Loading field visits...")

visits = pd.read_csv(
    DATA_DIR / "field_visits.csv",
    encoding="cp1252"
)

visits["requested_on"] = pd.to_datetime(
    visits["requested_on"],
    errors="coerce"
)

visits["gateway_id"] = (
    visits["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.upper()
)

visits["week_start"] = (
    visits["requested_on"]
    - pd.to_timedelta(
        visits["requested_on"].dt.weekday,
        unit="D"
    )
).dt.normalize()

failure_labels = (
    visits[
        visits["outcome"]
        .astype(str)
        .str.strip()
        .eq("Fehler behoben")
    ][
        ["week_start", "gateway_id"]
    ]
    .drop_duplicates()
)

failure_labels["needs_visit"] = 1

failure_labels["week_start"] = (
    failure_labels["week_start"]
    .dt.strftime("%Y-%m-%d")
)

features = features.merge(
    failure_labels,
    on=["week_start", "gateway_id"],
    how="left"
)

features["needs_visit"] = (
    features["needs_visit"]
    .fillna(0)
    .astype(int)
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output = (
    RESULTS_DIR /
    "ml_baseline_features.csv"
)

features.to_csv(
    output,
    index=False
)

print()
print("Saved:", output)
print("Rows:", len(features))
print(
    "Positive labels:",
    int(features["needs_visit"].sum())
)
print(
    "Weeks:",
    features["week_start"].nunique()
)