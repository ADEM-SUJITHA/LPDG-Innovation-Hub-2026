from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "results" / "ml_baseline_features.csv"
OUT_FILE = ROOT / "results" / "ml_cost_sensitive_test_predictions.csv"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
print("Loading features...")

df = pd.read_csv(FEATURE_FILE)
df["week_start"] = pd.to_datetime(df["week_start"])


FEATURES = [
    "offline_flag_hours",
    "disconnect_flag_hours",
    "reboot_flag_hours",
    "total_flagged_hours",
    "recent_offline_mean",
    "recent_disconnect_mean",
    "recent_reboot_mean",
]


TRAIN_END = pd.Timestamp("2025-11-24")
TEST_START = pd.Timestamp("2025-12-01")


train = df[
    df["week_start"] <= TRAIN_END
].copy()

test = df[
    df["week_start"] >= TEST_START
].copy()


print(f"Training rows: {len(train)}")
print(f"Testing rows: {len(test)}")
print(f"Test weeks: {test['week_start'].nunique()}")
print(f"Training failures: {train['needs_visit'].sum()}")
print(f"Testing failures: {test['needs_visit'].sum()}")


# ---------------------------------------------------------
# Normalize features using training data
# ---------------------------------------------------------
for col in FEATURES:

    min_value = train[col].min()
    max_value = train[col].max()

    if max_value > min_value:

        train[col + "_norm"] = (
            (train[col] - min_value)
            / (max_value - min_value)
        )

        test[col + "_norm"] = (
            (test[col] - min_value)
            / (max_value - min_value)
        )

    else:

        train[col + "_norm"] = 0
        test[col + "_norm"] = 0


NORM_FEATURES = [
    col + "_norm"
    for col in FEATURES
]


# ---------------------------------------------------------
# Cost-sensitive Random Forest
# ---------------------------------------------------------
print("\nTraining cost-sensitive Random Forest...")

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
# Failure probability
# ---------------------------------------------------------
test["failure_probability"] = model.predict_proba(
    test[NORM_FEATURES]
)[:, 1]


# ---------------------------------------------------------
# Select top 15 per week
# ---------------------------------------------------------
predictions = []

for week, group in test.groupby("week_start"):

    selected = group.sort_values(
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
            "week_start": week.strftime("%Y-%m-%d"),
            "rank": rank,
            "gateway_id": row["gateway_id"],
            "score": float(row["failure_probability"]),
            "reason": reason
        })


predictions = pd.DataFrame(predictions)


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------
predictions["week_start"] = pd.to_datetime(
    predictions["week_start"]
)

evaluation = predictions.merge(
    test[
        [
            "week_start",
            "gateway_id",
            "needs_visit"
        ]
    ],
    on=[
        "week_start",
        "gateway_id"
    ],
    how="left"
)

evaluation["needs_visit"] = (
    evaluation["needs_visit"]
    .fillna(0)
    .astype(int)
)


false_visits = int(
    (evaluation["needs_visit"] == 0).sum()
)


selected_failures = (
    evaluation[
        evaluation["needs_visit"] == 1
    ]
    .groupby("week_start")
    ["gateway_id"]
    .nunique()
)


actual_failures = (
    test[
        test["needs_visit"] == 1
    ]
    .groupby("week_start")
    ["gateway_id"]
    .nunique()
)


missed_failures = int(
    (
        actual_failures
        - selected_failures
    )
    .clip(lower=0)
    .sum()
)


# Challenge cost:
# False visit = EUR 380
# Missed failure = EUR 600

total_cost = (
    false_visits * 380
    + missed_failures * 600
)


test_weeks = test["week_start"].nunique()


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
print(
    "\n======================================================="
)

print("COST-SENSITIVE ML MODEL")

print(
    "======================================================="
)

print(f"Test weeks: {test_weeks}")
print(f"Selected visits: {len(predictions)}")
print(f"False visits: {false_visits}")
print(f"Missed failures: {missed_failures}")
print(f"Total cost: EUR {total_cost}")

print(
    f"Average weekly cost: "
    f"EUR {total_cost / test_weeks:.2f}"
)


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------
print("\nFeature importance:")

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(
    ascending=False
)

for feature, value in importance.items():
    print(f"{feature}: {value:.4f}")


# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------
predictions.to_csv(
    OUT_FILE,
    index=False
)

print(
    f"\nSaved: {OUT_FILE}"
)