from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_FILE = ROOT / "results" / "historical_baseline_predictions.csv"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
START_WEEK = pd.Timestamp("2025-12-01")
END_WEEK = pd.Timestamp("2026-01-26")

METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
]


# ---------------------------------------------------------
# Load telemetry
# ---------------------------------------------------------
print("Loading telemetry...")

telemetry = pd.read_parquet(
    DATA_DIR / "telemetry",
    columns=["gateway_id", "ts_utc"] + METRICS
)

telemetry["ts_utc"] = pd.to_datetime(
    telemetry["ts_utc"],
    utc=True,
    errors="coerce"
)

telemetry["gateway_id"] = (
    telemetry["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.upper()
)

telemetry = telemetry.dropna(
    subset=["ts_utc", "gateway_id"]
)

print(f"Telemetry rows: {len(telemetry):,}")


# ---------------------------------------------------------
# Load field visits
# ---------------------------------------------------------
print("Loading field visits...")

visits = pd.read_csv(
    DATA_DIR / "field_visits.csv"
)

visits["gateway_id"] = (
    visits["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.upper()
)

visits["requested_on"] = pd.to_datetime(
    visits["requested_on"],
    errors="coerce"
)

visits["needs_visit"] = (
    visits["outcome"]
    .astype(str)
    .str.strip()
    == "Fehler behoben"
).astype(int)

visits = visits.dropna(
    subset=["gateway_id", "requested_on"]
)


# ---------------------------------------------------------
# Create weekly labels
# ---------------------------------------------------------
visits["week_start"] = (
    visits["requested_on"]
    - pd.to_timedelta(
        visits["requested_on"].dt.weekday,
        unit="D"
    )
).dt.normalize()

labels = (
    visits[
        (visits["week_start"] >= START_WEEK)
        & (visits["week_start"] <= END_WEEK)
    ]
    .groupby(
        ["week_start", "gateway_id"],
        as_index=False
    )["needs_visit"]
    .max()
)

print(
    f"Historical labels: {len(labels):,}"
)

print(
    f"Confirmed failures: "
    f"{labels['needs_visit'].sum():,}"
)


# ---------------------------------------------------------
# Prepare telemetry dates
# ---------------------------------------------------------
print("\nGenerating historical baseline predictions...")

telemetry["date"] = (
    telemetry["ts_utc"]
    .dt.tz_localize(None)
)

weeks = pd.date_range(
    START_WEEK,
    END_WEEK,
    freq="7D"
)

predictions = []


# ---------------------------------------------------------
# Generate baseline predictions
# ---------------------------------------------------------
for week in weeks:

    baseline_start = (
        week - pd.Timedelta(days=28)
    )

    recent_start = (
        week - pd.Timedelta(days=7)
    )

    baseline = telemetry[
        (telemetry["date"] >= baseline_start)
        & (telemetry["date"] < week)
    ]

    recent = telemetry[
        (telemetry["date"] >= recent_start)
        & (telemetry["date"] < week)
    ]

    if baseline.empty or recent.empty:
        continue

    # Calculate baseline mean and standard deviation
    stats = (
        baseline
        .groupby("gateway_id")[METRICS]
        .agg(["mean", "std"])
    )

    # Flatten MultiIndex columns
    stats.columns = [
        f"{metric}_{stat}"
        for metric, stat in stats.columns
    ]

    # Merge recent data with baseline statistics
    merged = recent.merge(
        stats,
        left_on="gateway_id",
        right_index=True,
        how="inner"
    )

    scores = []

    # Score every gateway
    for gateway_id, group in merged.groupby(
        "gateway_id"
    ):

        flagged_hours = 0
        first_metric = None

        for metric in METRICS:

            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            threshold = (
                group[mean_col]
                + 3 * group[std_col].fillna(0)
            )

            flags = (
                group[metric] > threshold
            )

            count = int(flags.sum())

            if (
                count > 0
                and first_metric is None
            ):
                first_metric = metric

            flagged_hours += count

        if flagged_hours > 0:

            scores.append({
                "gateway_id": gateway_id,
                "score": flagged_hours,
                "first_metric": first_metric
            })

    if not scores:
        continue

    score_df = pd.DataFrame(scores)

    # Exact baseline ranking logic
    score_df = score_df.sort_values(
        ["score", "gateway_id"],
        ascending=[False, True]
    ).head(15)

    for rank, (_, row) in enumerate(
        score_df.iterrows(),
        start=1
    ):

        reason = (
            f"{int(row['score'])} hour(s) beyond "
            f"3 sigma on baseline telemetry; "
            f"first breach on "
            f"{row['first_metric']}"
        )

        predictions.append({
            "week_start": week.strftime(
                "%Y-%m-%d"
            ),
            "rank": rank,
            "gateway_id": row["gateway_id"],
            "score": float(row["score"]),
            "reason": reason
        })


# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------
predictions = pd.DataFrame(
    predictions
)

OUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

predictions.to_csv(
    OUT_FILE,
    index=False
)

print(
    f"\nSaved: {OUT_FILE}"
)

print(
    f"Prediction rows: "
    f"{len(predictions)}"
)

print(
    f"Weeks: "
    f"{predictions['week_start'].nunique()}"
)


# ---------------------------------------------------------
# Evaluate cost
# ---------------------------------------------------------
predictions["week_start"] = pd.to_datetime(
    predictions["week_start"]
)

evaluation = predictions.merge(
    labels,
    on=["week_start", "gateway_id"],
    how="left"
)

evaluation["needs_visit"] = (
    evaluation["needs_visit"]
    .fillna(0)
    .astype(int)
)


# False visit = selected gateway
# but no confirmed failure
false_visits = int(
    (
        evaluation["needs_visit"] == 0
    ).sum()
)


# Confirmed failures selected by baseline
selected_failures = (
    evaluation[
        evaluation["needs_visit"] == 1
    ]
    .groupby("week_start")
    ["gateway_id"]
    .nunique()
)


# Actual confirmed failures
actual_failures = (
    labels[
        labels["needs_visit"] == 1
    ]
    .groupby("week_start")
    ["gateway_id"]
    .nunique()
)


# Failures that were not selected
missed_failures = int(
    (
        actual_failures
        - selected_failures
    )
    .clip(lower=0)
    .sum()
)


# Cost model from challenge
total_cost = (
    false_visits * 380
    + missed_failures * 600
)

test_weeks = len(weeks)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------
print(
    "\n======================================================="
)

print(
    "HISTORICAL BASELINE"
)

print(
    "======================================================="
)

print(
    f"Test weeks: {test_weeks}"
)

print(
    f"Selected visits: "
    f"{len(predictions)}"
)

print(
    f"False visits: "
    f"{false_visits}"
)

print(
    f"Missed failures: "
    f"{missed_failures}"
)

print(
    f"Total cost: "
    f"EUR {total_cost}"
)

print(
    f"Average weekly cost: "
    f"EUR {total_cost / test_weeks:.2f}"
)