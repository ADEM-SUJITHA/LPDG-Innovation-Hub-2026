import pandas as pd

print("Loading gateway master...")
gateway = pd.read_csv("data/gateway_master.csv", encoding="cp1252")

print("\nGateway Master:")
print("Shape:", gateway.shape)
print("Columns:")
print(gateway.columns.tolist())

print("\nMissing values:")
print(gateway.isnull().sum())


print("\nLoading meter read success...")
meter = pd.read_csv("data/meter_read_success.csv", encoding="cp1252")
print("\nMeter Read Success:")
print("Shape:", meter.shape)
print("Columns:")
print(meter.columns.tolist())

print("\nMissing values:")
print(meter.isnull().sum())


print("\nLoading field visits...")
visits = pd.read_csv("data/field_visits.csv", encoding="cp1252")

print("\nField Visits:")
print("Shape:", visits.shape)
print("Columns:")
print(visits.columns.tolist())

print("\nMissing values:")
print(visits.isnull().sum())


print("\nLoading engineer review...")
review = pd.read_excel("data/engineer_review_2026-02.xlsx")

print("\nEngineer Review:")
print("Shape:", review.shape)
print("Columns:")
print(review.columns.tolist())

print("\nMissing values:")
print(review.isnull().sum())


print("\nLoading telemetry sample...")
telemetry = pd.read_csv("data/telemetry_sample_2025-08.csv", encoding="cp1252")

print("\nTelemetry Sample:")
print("Shape:", telemetry.shape)
print("Columns:")
print(telemetry.columns.tolist())

print("\nMissing values:")
print(telemetry.isnull().sum())

print("\n===== QUICK SUMMARY =====")

print("Total gateways:", gateway["gateway_id"].nunique())

print(
    "Total meters installed:",
    gateway["n_meters_installed"].sum()
)

print(
    "Total historical field visits:",
    len(visits)
)

print(
    "Engineer review categories:"
)
print(review["Kategorie"].value_counts())

print(
    "\nAverage meter read success rate:"
)

meter["read_success_rate"] = (
    meter["meters_read"] / meter["meters_expected"] * 100
)

print(
    round(meter["read_success_rate"].mean(), 2),
    "%"
)