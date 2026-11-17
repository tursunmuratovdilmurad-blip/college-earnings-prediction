# College Earnings Prediction

Predicting how much the former students of a U.S. college earn ten years after they enrolled,
using facts about the college that are known much earlier.

The data comes from the U.S. Department of Education
[College Scorecard](https://collegescorecard.ed.gov/data/) (Most Recent Cohorts, institution level).

## Live app

**https://college-earnings-prediction.streamlit.app**

The app has five pages: an overview of the project, the data with charts, the preprocessing steps,
a comparison of the models, and a page where you can describe a college and get a prediction.

## Repository layout

| Path | Contents |
|---|---|
| `notebook.ipynb` | The full pipeline: loading, EDA, cleaning, feature engineering, training and evaluation |
| `app.py`, `app_pages/` | The Streamlit app (entry file and one file per page) |
| `.streamlit/` | App settings (colour theme) |
| `src/` | A small custom transformer used by the pipeline, and shared helpers for the app |
| `data/` | Processed tables and a note on how to get the raw dataset ([data/README.md](data/README.md)) |
| `models/` | Trained models and `metadata.json` with their settings and scores |
| `reports/` | Result tables and figures produced by the notebook |

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app works with the files already in this repository. To re-run the notebook you also need the raw
dataset, see [data/README.md](data/README.md).

## Status

Work in progress. The plan:

- [x] Data pipeline notebook (cleaning, feature engineering, model training, evaluation)
- [x] Streamlit app with data exploration, preprocessing, model comparison and prediction pages
- [ ] Setup instructions and pinned requirements

## Licence

MIT. See [LICENSE](LICENSE).
