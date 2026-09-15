# LPDG Innovation Hub 2026

## Machine Learning Gateway Failure Prediction

This project predicts and ranks the **15 gateways that should be visited each week** based on historical telemetry and operational data.

The project implements a cost-sensitive Machine Learning approach using a **Random Forest classifier** and compares the approach with the provided 3-sigma baseline.

---

## 1. Project Objective

The system identifies the gateways most likely to require a field visit.

For each decision week, the system produces:

- 15 ranked gateways
- A risk score for each gateway
- A reason explaining the ranking
- Predictions for 8 weeks

### Prediction Period

```text
2026-02-02
2026-02-09
2026-02-16
2026-02-23
2026-03-02
2026-03-09
2026-03-16
2026-03-23