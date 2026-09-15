# AI-USAGE

## How AI tools were used

AI tools were used as a development assistant during this challenge.

They were used for:

- Understanding the challenge requirements.
- Helping structure the analysis workflow.
- Debugging Python and pandas errors.
- Helping identify data-column mismatches.
- Suggesting analysis steps for combining telemetry, meter-read and field-visit evidence.
- Reviewing the final submission structure and validation steps.

All commands were run and checked locally, and the generated results were validated using the provided validation script.

## One mistake caught

During the field-visit analysis, an AI-generated command initially assumed that the dataset contained a column named `visit_date`.

The actual dataset used the column:

`visited_on`

The command failed with:

`KeyError: 'visit_date'`

I checked the actual CSV columns, found the correct column name, and corrected the analysis to use `visited_on`.

This showed why AI-generated code needs to be checked against the actual data rather than being accepted without verification.