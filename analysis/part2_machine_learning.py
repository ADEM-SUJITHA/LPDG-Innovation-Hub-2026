import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

INPUT_FILE = RESULTS_DIR / "ml_baseline_features.csv"
OUTPUT_FILE = RESULTS_DIR / "ml_baseline_test_predictions.csv"

FEATURES = [
    "offline_flag_hours",
    "disconnect_flag_hours",
    "reboot_flag_hours",
    "total_flagged_hours",
    "recent_offline_mean",
    "recent_disconnect_mean",
    "recent_reboot_mean",
]

print("Loading baseline-style features...")

df = pd.read_csv(INPUT_FILE)

df["week_start"] = pd.to_datetime(
    df["week_start"]
)

df[FEATURES] = df[FEATURES].fillna(0)

# Train on earlier weeks
train = df[
    df["week_start"] <= "2025-11-24"
].copy()

# Test on later unseen weeks
test = df[
    df["week_start"] >= "2025-12-01"
].copy()

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Test weeks:", test["week_start"].nunique())
print(
    "Training failures:",
    int(train["needs_visit"].sum())
)
print(
    "Testing failures:",
    int(test["needs_visit"].sum())
)

# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=6,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    train[FEATURES],
    train["needs_visit"]
)

test["ml_score"] = model.predict_proba(
    test[FEATURES]
)[:, 1]

# ---------------------------------------------------------
# Rank top 15 per week
# ---------------------------------------------------------

test = test.sort_values(
    ["week_start", "ml_score"],
    ascending=[True, False]
)

test["ml_rank"] = (
    test
    .groupby("week_start")
    .cumcount()
    + 1
)

test["ml_selected"] = (
    test["ml_rank"] <= 15
).astype(int)

# ---------------------------------------------------------
# Calculate operational cost
# ---------------------------------------------------------

selected = test[
    test["ml_selected"] == 1
]

false_visits = int(
    (selected["needs_visit"] == 0).sum()
)

missed_failures = int(
    (
        (test["needs_visit"] == 1)
        & (test["ml_selected"] == 0)
    ).sum()
)

total_cost = (
    false_visits * 380
    + missed_failures * 600
)

weeks = test["week_start"].nunique()

# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    RESULTS_DIR /
    "ml_baseline_feature_importance.csv",
    index=False
)

test.to_csv(
    OUTPUT_FILE,
    index=False
)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("=" * 55)
print("ML + BASELINE FEATURES")
print("=" * 55)

print("Test weeks:", weeks)
print("Selected visits:", len(selected))
print("False visits:", false_visits)
print("Missed failures:", missed_failures)
print("Total cost: EUR", total_cost)
print(
    "Average weekly cost: EUR",
    round(total_cost / weeks, 2)
)

print()
print("Feature importance:")

for _, row in importance.iterrows():
    print(
        f"{row['feature']}: "
        f"{row['importance']:.4f}"
    )

print()
print("Saved:", OUTPUT_FILE)