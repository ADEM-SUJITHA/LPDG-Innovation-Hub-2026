import pandas as pd

# Load data
pred = pd.read_csv("predictions.csv")
meter = pd.read_csv("data/meter_read_success.csv", encoding="cp1252")
visits = pd.read_csv("data/field_visits.csv", encoding="cp1252")

# Normalize gateway IDs
for df in [pred, meter, visits]:
    df["gateway_id"] = (
        df["gateway_id"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
    )

# -----------------------------
# 1. Meter-read success
# -----------------------------
meter["read_success_rate"] = (
    meter["meters_read"] / meter["meters_expected"] * 100
)

meter_summary = (
    meter.groupby("gateway_id")["read_success_rate"]
    .mean()
    .reset_index()
)

# Convert success rate into failure rate
meter_summary["read_failure_rate"] = (
    100 - meter_summary["read_success_rate"]
)

# -----------------------------
# 2. Field visit evidence
# -----------------------------
visits["problem_fixed"] = (
    visits["outcome"] == "Fehler behoben"
).astype(int)

visit_summary = (
    visits.groupby("gateway_id")
    .agg(
        field_visits=("gateway_id", "size"),
        problems_fixed=("problem_fixed", "sum"),
        technician_hours=("technician_hours", "sum")
    )
    .reset_index()
)

# -----------------------------
# 3. Combine with predictions
# -----------------------------
x = pred.merge(
    meter_summary,
    on="gateway_id",
    how="left"
)

x = x.merge(
    visit_summary,
    on="gateway_id",
    how="left"
)

# Fill missing values
x["read_failure_rate"] = x["read_failure_rate"].fillna(0)
x["field_visits"] = x["field_visits"].fillna(0)
x["problems_fixed"] = x["problems_fixed"].fillna(0)
x["technician_hours"] = x["technician_hours"].fillna(0)

# -----------------------------
# 4. Additional evidence score
# -----------------------------
x["evidence_score"] = (
    x["score"]
    + (x["read_failure_rate"] * 0.20)
    + (x["field_visits"] * 0.50)
    + (x["problems_fixed"] * 0.50)
)

# -----------------------------
# 5. Rank within each week
# -----------------------------
x["new_rank"] = (
    x.groupby("week_start")["evidence_score"]
    .rank(method="first", ascending=False)
    .astype(int)
)

# Keep top 15
top15 = x[x["new_rank"] <= 15].copy()

# Sort
top15 = top15.sort_values(
    ["week_start", "new_rank"]
)

# Display first week
print("\n===== IMPROVED TOP 15 =====")
print(
    top15[
        [
            "week_start",
            "new_rank",
            "gateway_id",
            "score",
            "read_failure_rate",
            "field_visits",
            "problems_fixed",
            "evidence_score"
        ]
    ].head(15).to_string(index=False)
)

# Save analysis
top15.to_csv(
    "results/improved_analysis.csv",
    index=False
)

print("\nSaved: results/improved_analysis.csv")