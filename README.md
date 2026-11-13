# College Earnings Prediction

Predicting how much the former students of a U.S. college earn ten years after they enrolled,
using facts about the college that are known much earlier.

The data comes from the U.S. Department of Education
[College Scorecard](https://collegescorecard.ed.gov/data/) (Most Recent Cohorts, institution level).

## Repository layout

| Path | Contents |
|---|---|
| `notebook.ipynb` | The full pipeline: loading, EDA, cleaning, feature engineering, training and evaluation |
| `src/` | A small custom transformer used by the pipeline |
| `data/` | Processed tables and a note on how to get the raw dataset ([data/README.md](data/README.md)) |
| `models/` | Trained models and `metadata.json` with their settings and scores |
| `reports/` | Result tables and figures produced by the notebook |

## Status

Work in progress. The plan:

- [x] Data pipeline notebook (cleaning, feature engineering, model training, evaluation)
- [ ] Streamlit app with data exploration, preprocessing, model comparison and prediction pages
- [ ] Setup instructions and pinned requirements

## Licence

MIT. See [LICENSE](LICENSE).
