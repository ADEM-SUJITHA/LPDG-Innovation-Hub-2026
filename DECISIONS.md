# DECISIONS

## Decision 1 — Use the provided 3-sigma baseline

### Choice
I used the provided `baseline_3sigma.py` as the starting point for the Part 1 ranking.

### Alternative considered
I considered building a new machine-learning model from scratch.

### Why I did not choose it
The challenge explicitly provides a working baseline and states that machine learning is not required. I decided to spend the available time on improving the operational decision using additional evidence rather than adding unnecessary model complexity.

---

## Decision 2 — Combine telemetry with meter-read performance

### Choice
I added meter-read failure rate as supporting evidence for gateway risk.

### Alternative considered
I could have used telemetry anomalies alone.

### Why I did not choose it
Telemetry shows abnormal gateway behaviour, but meter-read performance provides a direct operational signal about whether meters behind a gateway are being successfully read. I therefore used it as additional evidence rather than relying on a single signal.

---

## Decision 3 — Use historical field visits as supporting evidence

### Choice
I included historical field-visit frequency and previously fixed problems in the risk score.

### Alternative considered
I could have ignored historical field visits and used only telemetry and meter-read data.

### Why I did not choose it
Field visits provide operational evidence of real gateway issues. However, I did not treat every visit as proof of a current failure because some visits ended with no error found. Historical visits are therefore supporting evidence rather than the only decision signal.

---

## Decision 4 — Prioritise evidence instead of creating a binary failure rule

### Choice
I created a continuous evidence score and ranked gateways from highest to lowest risk.

### Alternative considered
I could have created a simple rule such as "visit every gateway whose meter-read success is below a fixed threshold."

### Why I did not choose it
The challenge requires exactly 15 ranked visits per week. A continuous score allows different types of evidence to be combined and produces an ordered priority list rather than a simple yes/no decision.

---

## Decision 5 — Part 2 area selection

### Choice
I selected **Data Science** for Part 2.

### Alternative considered
I considered Machine Learning and Software Development.

### Why I did not choose them
I selected Data Science because the challenge specifically asks the candidate to question what "needs a visit" means, test the decision honestly, consider uncertainty, and translate the €380 visit cost and €600 weekly failure cost into an operational decision. This matches the type of analysis I used in Part 1.

---

# What It Cannot Do

The solution is a prioritisation system, not a guaranteed failure predictor.

It cannot guarantee that a gateway ranked highly will fail.

Historical field visits may represent problems that have already been resolved, so they cannot be treated as proof of a current failure.

The scoring weights are manually selected evidence weights rather than weights learned from a validated supervised model.

The available data also does not establish a perfect definition of "needs a visit", so the ranking should be treated as decision support for the operations team.

## What another two weeks would fix

With another two weeks, I would:

1. Define and validate a clear operational target for "needs a visit".
2. Back-test different scoring weights on historical weeks.
3. Estimate the cost of false visits versus missed failures using the €380 and €600 costs.
4. Test the ranking on unseen future weeks.
5. Tune the decision threshold based on operational cost rather than only anomaly scores.

This would provide stronger evidence that the ranking improves field-visit decisions rather than only producing a different ranking from the baseline.