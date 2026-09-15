import pandas as pd

# Load baseline predictions
predictions = pd.read_csv("predictions.csv")

# Load gateway information
gateway = pd.read_csv(
    "data/gateway_master.csv",
    encoding="cp1252"
)

# Load meter read data
meter = pd.read_csv(
    "data/meter_read_success.csv",
    encoding="cp1252"
)

# Calculate meter read success rate
meter["read_success_rate"] = (
    meter["meters_read"] / meter["meters_expected"] * 100
)

# Average weekly success rate for each gateway
meter_summary = (
    meter.groupby("gateway_id")
    .agg(
        avg_read_success=("read_success_rate", "mean"),
        total_meters_expected=("meters_expected", "sum"),
        total_meters_read=("meters_read", "sum")
    )
    .reset_index()
)

# Calculate overall success rate
meter_summary["overall_read_success"] = (
    meter_summary["total_meters_read"]
    / meter_summary["total_meters_expected"]
    * 100
)

# Add gateway information
# Normalize gateway IDs before merging
predictions["gateway_id"] = (
    predictions["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.upper()
)

gateway["gateway_id"] = (
    gateway["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.upper()
)

meter_summary["gateway_id"] = (
    meter_summary["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.upper()
)

# Add gateway information
gateway_info = gateway[
    ["gateway_id", "n_meters_installed", "site_type", "region"]
]
# Merge everything
result = predictions.merge(
    gateway_info,
    on="gateway_id",
    how="left"
)

result = result.merge(
    meter_summary,
    on="gateway_id",
    how="left"
)

# Show first 15 gateways
print("\n===== BASELINE TOP 15 =====")

print(
    result[
        [
            "week_start",
            "rank",
            "gateway_id",
            "score",
            "n_meters_installed",
            "overall_read_success",
            "site_type",
            "region"
        ]
    ].head(15).to_string(index=False)
)

# Save result
result.to_csv(
    "results/baseline_analysis.csv",
    index=False
)

print("\nSaved:")
print("results/baseline_analysis.csv")