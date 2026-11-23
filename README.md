# College Earnings Prediction

How much do the former students of a U.S. college earn ten years after they started? The government
publishes this number, but it arrives ten years late and is missing for many colleges. This project
predicts it from facts that are known much earlier: the type of college, its prices, its students and
what it teaches.

It is a supervised regression problem. A model learns from 3,967 colleges whose earnings are known and
then predicts earnings for colleges it has not seen.

## Live app

**https://college-earnings-prediction.streamlit.app**

The app has five pages:

| Page | What you can do there |
|---|---|
| Home | See what the project is about and the main numbers |
| Data overview | Filter and browse the data, see its statistics, distributions and correlations |
| Preprocessing | See every cleaning step, the new features, missing values, outliers, scaling and encoding |
| Model comparison | Compare the four models, their scores, their tuning and where they go wrong |
| Predict earnings | Start from a real college, change anything about it and get a prediction with an explanation |

## Results

Four models were trained and compared with a simple baseline. These are their scores on 596 test
colleges that no model saw during training.

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| **SVR (RBF kernel), final model** | **$6,360** | **$3,972** | **8.1%** | **0.888** |
| Histogram Gradient Boosting | $6,516 | $4,098 | 8.2% | 0.882 |
| Random Forest | $6,886 | $4,303 | 8.6% | 0.869 |
| ElasticNet | $7,969 | $5,140 | 10.3% | 0.824 |
| Baseline (predicts the median for every college) | $19,301 | $13,695 | 28.7% | -0.031 |

SVR was chosen because it had the lowest error on the validation set. SVR and Gradient Boosting are
very close, and on this test set they cannot really be told apart. The feature the model relies on most
is the family income of the students a college enrols.

## Data

The data is the [College Scorecard](https://collegescorecard.ed.gov/data/) from the U.S. Department of
Education, file "Most Recent Cohorts: Institution" (6,272 colleges, 3,379 columns). The U.S. Department of
Education publishes it as open data that anyone may use.

The raw file is about 102 MB, so it is not stored here. The app does not need it. To re-run the
notebook you need to download it first: see [data/README.md](data/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `notebook.ipynb` | The full pipeline: loading, EDA, cleaning, feature engineering, training and evaluation |
| `app.py`, `app_pages/` | The Streamlit app: the entry file and one file per page |
| `.streamlit/` | App settings (colour theme) |
| `src/` | A small custom transformer used by the pipeline, and shared helpers for the app |
| `tests/` | Automated tests for the custom transformer, the saved models and the app |
| `data/` | Processed tables and a note on how to get the raw dataset |
| `models/` | Trained models and `metadata.json` with their settings and scores |
| `reports/` | Result tables and figures produced by the notebook |
| `requirements.txt` | Every package the project uses, with its exact version |
| `pyproject.toml`, `uv.lock` | The same packages for the uv tool, with the exact version of every package they depend on |
| `Dockerfile`, `.dockerignore` | A container that runs the app |
| `.github/workflows/` | Checks that run on every push: the tests, and a build of the container that opens every page of the app |

## Setup

You need Python 3.9, 3.10, 3.11 or 3.12. The app online runs on Python 3.12.

1. Download the project and go into its folder:

   ```bash
   git clone https://github.com/tursunmuratovdilmurad-blip/college-earnings-prediction.git
   cd college-earnings-prediction
   ```

2. Make a virtual environment, so the packages do not mix with anything else on your computer.

   On macOS or Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows:

   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

### Exact install with uv

`requirements.txt` fixes the version of every package the project uses directly. `uv.lock` goes one step
further and also fixes the version of every package those depend on, so every computer gets exactly the
same set. The live app installs from `uv.lock`. To use it, install [uv](https://docs.astral.sh/uv/) and run
this in the project folder instead of steps 2 and 3:

```bash
uv sync --locked
```

Then put `uv run` in front of the commands below, for example `uv run streamlit run app.py`.

If you change a version, change it in both `pyproject.toml` and `requirements.txt`, then run `uv lock`.

## Run the app

```bash
streamlit run app.py
```

The app opens in your browser at http://localhost:8501. It uses the files already in this repository,
so it works without the raw dataset.

### Run the app with Docker

If you have [Docker](https://www.docker.com/), you can run the app without installing Python or any
package. The image installs exactly the versions in `uv.lock` on Python 3.12, like the live app.

```bash
docker build -t college-earnings .
docker run -p 8501:8501 college-earnings
```

Then open http://localhost:8501. Every time code is pushed to GitHub, a check builds this image, starts
the app and opens every page (see `.github/workflows/docker.yml`).

## Run the tests

The tests check that the custom outlier capping works as intended, that the saved models load and give
exactly the scores and predictions recorded by the notebook, and that every page of the app opens without
an error. Run them from the project folder:

```bash
pip install pytest==8.4.2
pytest
```

With uv, `uv sync` already installs pytest, so `uv run pytest` is enough. They take about a minute.
GitHub runs them on Python 3.9 and 3.12 every time code is pushed (see `.github/workflows/tests.yml`).

## Run the notebook

1. Download the raw dataset and put `Most-Recent-Cohorts-Institution.csv` in the `data/` folder, as
   described in [data/README.md](data/README.md).
2. Run the notebook from the project folder, because it finds its files from there. Either open
   `notebook.ipynb` in VS Code or Jupyter (for Jupyter, run `pip install notebook` and then
   `jupyter notebook`), or run all of it from the command line:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebook.ipynb --ExecutePreprocessor.timeout=1800
   ```

A full run takes a few minutes (about three on a recent laptop). It writes the cleaned data, the trained
models, the result tables and the figures again. Every random step uses a fixed seed
(`RANDOM_STATE = 42`), so the models, scores and predictions come out exactly the same as the files in
this repository. Only the recorded run times and the date in `models/metadata.json` change.

## Licence

The code is under the MIT licence, see [LICENSE](LICENSE). The data belongs to the U.S. Department of
Education and is free to use.
