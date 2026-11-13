# Data

## Raw dataset

The raw file is not stored in this repository because it is about 102 MB.

1. Go to https://collegescorecard.ed.gov/data/
2. Download "Most Recent Cohorts: Institution" (a zip file).
3. Unzip it and put `Most-Recent-Cohorts-Institution.csv` in this `data/` folder.

Source: U.S. Department of Education, College Scorecard. The version used here was downloaded on 10 November 2026
and has 6,272 colleges and 3,379 columns.

## Processed files

The notebook writes these files to `data/processed/`:

| File | What it is |
|---|---|
| `raw_subset.csv` | The 104 columns of the raw file that the project uses, before any cleaning |
| `model_table.csv` | The cleaned table with one row per institution (3,967 colleges), the engineered features, the target and the train / validation / test label |
| `train_preprocessed_sample.csv` | The training set after imputation, scaling and encoding, as the models see it |
