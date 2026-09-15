# DECISIONS

## Decision 1 — Use the provided 3-sigma baseline

### Choice
I used the provided `baseline_3sigma.py` as the starting point for the Part 1 ranking.

### Alternative considered
I considered building a machine-learning solution immediately without first reproducing the baseline.

### Why I did not choose it
The baseline provides a clear reference implementation and a fixed operational ranking method. Reproducing it first made it possible to compare later improvements against a known result.

---

## Decision 2 — Use a confirmed failure as the ML target

### Choice
For the machine-learning experiment, I defined a confirmed failure as a field visit whose outcome was `Fehler behoben` (problem fixed).

### Alternative considered
I could have treated every field visit as a failure, including visits recorded as `Kein Fehler gefunden` or `Kein Zugang`.

### Why I did not choose it
A visit does not necessarily mean that the gateway was actually broken. Using `Fehler behoben` gives the model a more specific confirmed-failure target. The other outcomes are therefore not treated as confirmed failures.

---

## Decision 3 — Use only information available before the decision week

### Choice
The ML features use a trailing historical window before each Monday decision week. The baseline-style features use the previous 28 days and recent 7-day telemetry.

### Alternative considered
I could have used telemetry from the same week or information recorded after the decision date.

### Why I did not choose it
That would introduce information leakage because the system would use information that would not have been available when deciding which gateways to visit. Using previous-week information better represents a real operational prediction setting.

---

## Decision 4 — Use a cost-sensitive Random Forest

### Choice
I selected a Random Forest classifier with higher class weight for confirmed failures. The final model uses telemetry anomaly features such as offline duration, disconnections, reboots, and recent averages.

### Alternative considered
I tested a plain Random Forest, a hybrid ML/baseline score, and an ML tie-breaker.

### Why I did not choose them
The cost-sensitive Random Forest produced the strongest result among the tested ML approaches on the historical evaluation. The historical comparison was performed using the challenge costs of EUR 380 for a false visit and EUR 600 for a missed confirmed failure.

The cost-sensitive model produced EUR 57,000 on the historical test period compared with EUR 57,760 for the reproduced baseline.

---

## Decision 5 — Part 2 area selection

### Choice
I selected **Machine Learning (Option E)** for Part 2.

### Alternative considered
I considered Data Science and Software Development.

### Why I did not choose them
Machine Learning is the best fit for the work because I built a supervised Random Forest model to predict confirmed gateway failures from historical telemetry. I also evaluated the model against the provided baseline using the challenge's operational cost structure.

The model uses historical data for training and generates a ranked list of 15 gateways for each decision week.

---

# What It Cannot Do

The solution is a prioritisation system, not a guaranteed failure predictor.

The ML target is based on field visits recorded as `Fehler behoben`. Therefore, the dataset does not provide a perfect definition of every gateway that genuinely "needs a visit".

A high predicted probability does not guarantee that a gateway will fail.

The model also depends on the telemetry patterns represented in the historical data. New gateway behaviour or a major change in the network environment may reduce prediction quality.

The historical evaluation contains only the confirmed failures represented by the available field-visit records. Gateways with no recorded visit cannot automatically be assumed to have been healthy.

## What another two weeks would fix

With another two weeks, I would:

1. Work with the operations team to define and validate the target for "needs a visit".
2. Add more historical operational labels, including the outcome and severity of previous visits.
3. Test the model on additional unseen weeks and gateways.
4. Investigate network changes such as changes in network type or operator.
5. Compare several models using the same time-based evaluation.
6. Tune the model directly against the EUR 380 false-visit and EUR 600 missed-failure costs.
7. Monitor model performance after deployment and retrain when gateway or network behaviour changes.
