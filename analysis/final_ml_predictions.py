from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "results" / "ml_baseline_features.csv"
OUT_FILE = ROOT / "predictions_ml.csv"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
FEATURES = [
    "offline_flag_hours",
    "disconnect_flag_hours",
    "reboot_flag_hours",
    "total_flagged_hours",
    "recent_offline_mean",
    "recent_disconnect_mean",
    "recent_reboot_mean",
]

TRAIN_END = pd.Timestamp("2026-01-26")

FINAL_START = pd.Timestamp("2026-02-02")
FINAL_END = pd.Timestamp("2026-03-23")


# ---------------------------------------------------------
# Load historical features
# ---------------------------------------------------------
print("Loading historical features...")

df = pd.read_csv(FEATURE_FILE)

df["week_start"] = pd.to_datetime(
    df["week_start"]
)

# Only data available before the first
# challenge prediction week.
train = df[
    df["week_start"] <= TRAIN_END
].copy()

print(
    f"Training rows: {len(train)}"
)

print(
    f"Training weeks: "
    f"{train['week_start'].nunique()}"
)

print(
    f"Training failures: "
    f"{train['needs_visit'].sum()}"
)


# ---------------------------------------------------------
# Normalize features
# ---------------------------------------------------------
for col in FEATURES:

    min_value = train[col].min()
    max_value = train[col].max()

    if max_value > min_value:

        train[col + "_norm"] = (
            (train[col] - min_value)
            / (max_value - min_value)
        )

    else:

        train[col + "_norm"] = 0


NORM_FEATURES = [
    col + "_norm"
    for col in FEATURES
]


# ---------------------------------------------------------
# Train final cost-sensitive model
# ---------------------------------------------------------
print("\nTraining final cost-sensitive ML model...")

model = RandomForestClassifier(
    n_estimators=600,
    max_depth=6,
    min_samples_leaf=3,
    class_weight={
        0: 1.0,
        1: 1.6
    },
    random_state=42,
    n_jobs=-1
)

model.fit(
    train[NORM_FEATURES],
    train["needs_visit"]
)


# ---------------------------------------------------------
# Create features for final challenge weeks
# ---------------------------------------------------------
#
# The existing feature file contains historical
# feature rows only, so we now calculate the same
# baseline-style features for the eight final weeks
# using telemetry data available before each Monday.
#
# ---------------------------------------------------------

print("\nLoading telemetry...")

DATA_DIR = ROOT / "data"

METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
]

telemetry = pd.read_parquet(
    DATA_DIR / "telemetry",
    columns=[
        "gateway_id",
        "ts_utc"
    ] + METRICS
)

telemetry["ts_utc"] = pd.to_datetime(
    telemetry["ts_utc"],
    utc=True,
    errors="coerce"
)

telemetry["gateway_id"] = (
    telemetry["gateway_id"]
    .astype(str)
    .str.replace(
        ":",
        "",
        regex=False
    )
    .str.upper()
)

telemetry = telemetry.dropna(
    subset=[
        "ts_utc",
        "gateway_id"
    ]
)

telemetry["date"] = (
    telemetry["ts_utc"]
    .dt.tz_localize(None)
)


# ---------------------------------------------------------
# Generate final predictions
# ---------------------------------------------------------
weeks = pd.date_range(
    FINAL_START,
    FINAL_END,
    freq="7D"
)

predictions = []


for week in weeks:

    print(
        f"Processing week "
        f"{week.strftime('%Y-%m-%d')}..."
    )

    baseline_start = (
        week - pd.Timedelta(days=28)
    )

    recent_start = (
        week - pd.Timedelta(days=7)
    )

    # IMPORTANT:
    # Only data before the prediction Monday
    # is used.
    baseline = telemetry[
        (telemetry["date"] >= baseline_start)
        & (telemetry["date"] < week)
    ]

    recent = telemetry[
        (telemetry["date"] >= recent_start)
        & (telemetry["date"] < week)
    ]

    if baseline.empty or recent.empty:
        print("No telemetry available.")
        continue

    stats = (
        baseline
        .groupby("gateway_id")[METRICS]
        .agg(["mean", "std"])
    )

    stats.columns = [
        f"{metric}_{stat}"
        for metric, stat in stats.columns
    ]

    merged = recent.merge(
        stats,
        left_on="gateway_id",
        right_index=True,
        how="inner"
    )

    scores = []

    for gateway_id, group in merged.groupby(
        "gateway_id"
    ):

        offline_flags = (
            group["offline_duration_sec"]
            >
            (
                group["offline_duration_sec_mean"]
                + 3 *
                group["offline_duration_sec_std"].fillna(0)
            )
        )

        disconnect_flags = (
            group["disconnection_cnt"]
            >
            (
                group["disconnection_cnt_mean"]
                + 3 *
                group["disconnection_cnt_std"].fillna(0)
            )
        )

        reboot_flags = (
            group["reboot_cnt"]
            >
            (
                group["reboot_cnt_mean"]
                + 3 *
                group["reboot_cnt_std"].fillna(0)
            )
        )

        offline_flag_hours = int(
            offline_flags.sum()
        )

        disconnect_flag_hours = int(
            disconnect_flags.sum()
        )

        reboot_flag_hours = int(
            reboot_flags.sum()
        )

        total_flagged_hours = (
            offline_flag_hours
            + disconnect_flag_hours
            + reboot_flag_hours
        )

        if total_flagged_hours == 0:
            continue

        scores.append({
            "gateway_id": gateway_id,

            "offline_flag_hours":
                offline_flag_hours,

            "disconnect_flag_hours":
                disconnect_flag_hours,

            "reboot_flag_hours":
                reboot_flag_hours,

            "total_flagged_hours":
                total_flagged_hours,

            "recent_offline_mean":
                group[
                    "offline_duration_sec"
                ].mean(),

            "recent_disconnect_mean":
                group[
                    "disconnection_cnt"
                ].mean(),

            "recent_reboot_mean":
                group[
                    "reboot_cnt"
                ].mean(),
        })

    if not scores:
        print("No scored gateways.")
        continue

    week_features = pd.DataFrame(
        scores
    )

    # -----------------------------------------------------
    # Normalize using training statistics
    # -----------------------------------------------------
    for col in FEATURES:

        min_value = train[col].min()
        max_value = train[col].max()

        if max_value > min_value:

            week_features[
                col + "_norm"
            ] = (
                (
                    week_features[col]
                    - min_value
                )
                /
                (
                    max_value
                    - min_value
                )
            )

        else:

            week_features[
                col + "_norm"
            ] = 0

    # -----------------------------------------------------
    # Predict failure probability
    # -----------------------------------------------------
    week_features[
        "failure_probability"
    ] = model.predict_proba(
        week_features[NORM_FEATURES]
    )[:, 1]

    # -----------------------------------------------------
    # Rank top 15
    # -----------------------------------------------------
    selected = week_features.sort_values(
        [
            "failure_probability",
            "total_flagged_hours",
            "gateway_id"
        ],
        ascending=[
            False,
            False,
            True
        ]
    ).head(15)

    for rank, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):

        reason = (
            f"Predicted failure probability "
            f"{row['failure_probability']:.4f}; "
            f"{int(row['total_flagged_hours'])} "
            f"baseline risk hours"
        )

        predictions.append({
            "week_start":
                week.strftime("%Y-%m-%d"),

            "rank":
                rank,

            "gateway_id":
                row["gateway_id"],

            "score":
                float(
                    row["failure_probability"]
                ),

            "reason":
                reason
        })


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
predictions = pd.DataFrame(
    predictions
)

predictions.to_csv(
    OUT_FILE,
    index=False
)

print(
    "\n======================================================="
)

print("FINAL ML PREDICTIONS")

print(
    "======================================================="
)

print(
    f"Rows: {len(predictions)}"
)

print(
    f"Weeks: "
    f"{predictions['week_start'].nunique()}"
)

print(
    f"Output: {OUT_FILE}"
)