# Datanaut

**An AI-Powered Intelligent Platform for Time-Series Data Analysis, Analytics and Insight Generation.**

[![Live app](https://img.shields.io/badge/live%20app-datanaut6.streamlit.app-ff4b4b?logo=streamlit&logoColor=white)](https://datanaut6.streamlit.app)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![CI](https://github.com/xenkrypt/Datanaut/actions/workflows/ci.yml/badge.svg)](https://github.com/xenkrypt/Datanaut/actions/workflows/ci.yml)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff/)

Upload a CSV or Excel file of timestamped readings (energy use, sensors, anything numeric over time) and Datanaut walks you through
profiling, cleaning, analysing and reporting it, with an optional AI analyst you can ask questions in plain language.

**Live demo: <https://datanaut6.streamlit.app>**

---

## Table of contents

- [Features](#features)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Using the app](#using-the-app)
- [Command line](#command-line)
- [AI analyst](#ai-analyst-optional)
- [Limits and privacy](#limits-and-privacy)
- [Testing and CI](#testing-and-ci)
- [Deployment](#deployment)
- [Sample data](#sample-data)

---

## Features

| Step | What it does |
|------|--------------|
| **Upload** | CSV / Excel (`.xlsx`) with encoding detection, or one click to load the bundled sample. |
| **Profile** | Data-quality summary: missing values, duplicates, gaps, sampling interval, column types. |
| **Map columns** | Choose the timestamp (or date + time columns) and the value columns; mark consumption columns. |
| **Clean** | Suggested fixes with previews, applied one by one. Every change creates a new **version** you can undo, with a full pipeline log and rule-based validation. |
| **Analyze** | Daily/weekly **patterns**, **anomalies**, **change points**, **forecast**, **weather** association and **what-if** scenarios for one column. |
| **Insights** | Plain-language findings and a guided investigation, with the evidence behind each statement. |
| **Ask** | Optional AI analyst: ask questions and get answers backed by evidence cards and a number check. |
| **Report** | Download a PDF report of the analysis. |

Other highlights:

- **Reproducible cleaning**: a recipe file records every step with data fingerprints, so a cleaned dataset can be rebuilt and verified.
- **CSV-safe export** that neutralises formula-injection in downloaded files.
- **Dark, animated dashboard** with live KPI cards and respect for reduced-motion settings.
- **Guarded by design**: size limits enforced in code, no tracebacks shown to users, nothing stored server-side.

## Tech stack

| Area | Tools |
|------|-------|
| Language | Python 3.11+ |
| UI | [Streamlit](https://streamlit.io), [Plotly](https://plotly.com/python/), custom CSS |
| Data | pandas, NumPy, SciPy, openpyxl, charset-normalizer |
| Analytics / ML | statsmodels (seasonality, forecasting), scikit-learn, ruptures (change points) |
| Validation | pydantic v2 |
| Reporting | fpdf2, matplotlib |
| AI analyst | Google Gemini or Groq (Llama 3.3) over plain HTTPS, allow-listed queries only |
| Quality | pytest, Hypothesis, ruff, pypdf |
| CI / hosting | GitHub Actions (tests on 3.11 and 3.12, `pip-audit`), Streamlit Community Cloud |

## Project structure

```text
Datanaut/
├── app.py                     # Streamlit entry point and page router
├── pyproject.toml             # Package metadata, dependencies, ruff and pytest config
├── requirements.txt           # Pinned runtime dependencies for hosting
├── sample_data/               # Household energy sample + 12-month demo dataset (CSV / XLSX)
├── scripts/                   # Sample/demo data generators, AI eval questions, stress test
├── .github/workflows/ci.yml   # Lint, format check, tests (3.11 / 3.12), dependency audit
├── src/datanaut/
│   ├── cli.py, __main__.py    # Command line (profile, clean, replay, analyze, eval-ai)
│   ├── errors.py, limits.py   # Friendly error types and every size limit in one place
│   ├── startup.py             # Start-up environment checks
│   ├── core/                  # The analysis engine (no UI code)
│   │   ├── loader.py, timeindex.py, profiler.py, characterize.py
│   │   ├── cleaner.py, validation.py, versioning.py, fingerprint.py, log.py, export.py
│   │   ├── patterns.py, anomalies.py, changepoints.py, forecast.py
│   │   └── weather.py, whatif.py, findings.py, investigate.py, evidence.py
│   ├── ai/                    # Opt-in AI analyst
│   │   ├── llm.py, config.py  # Provider calls, models and limits
│   │   ├── analyst.py, executor.py, schemas.py   # Allow-listed query plans and execution
│   │   └── context.py, suggest.py, evaluate.py, text.py
│   ├── report/                # PDF report: model, charts, text, pdf
│   └── ui/                    # Streamlit layer
│       ├── theme.py           # Dark theme CSS and animated dashboard cards
│       ├── state.py, actions.py, overview.py, charts.py, ...
│       └── pages/             # upload, profile, map_columns, clean, analyze, insights, ask, report, sidebar
└── tests/                     # core, ai, ui, report and audit tests
```

## Getting started

**Requirements:** Python 3.11 or newer.

```powershell
git clone https://github.com/xenkrypt/Datanaut.git
cd Datanaut
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

Then open <http://localhost:8501>. For development tools (pytest, ruff, Hypothesis), use `pip install -e ".[dev]"`.

## Using the app

1. **Upload** a file, or click **Use sample data**.
2. Check the **Profile**, then **Map columns** (time and value columns) and build the time index to create version V1.
3. **Clean**: preview and apply the suggested fixes; review the validation and the versions.
4. **Analyze** a column, read the **Insights**, optionally **Ask** questions, and download the **Report**.

## Command line

The same engine is available without the UI:

```powershell
python -m datanaut profile data.csv
python -m datanaut clean data.csv --recipe-out recipe.json --cleaned-out cleaned.csv
python -m datanaut replay data.csv recipe.json --cleaned-out rebuilt.csv
python -m datanaut analyze data.csv --column global_active_power --consumption
```

Run `python -m datanaut <command> --help` for all options (for example `--rules` and `--strict` for validation,
`--time-col` for date + time columns, `--sheet` for Excel).

## AI analyst (optional)

The **Ask** page is off until you provide a key. Supply either:

- `GEMINI_API_KEY` (Google Gemini), or
- `GROQ_API_KEY` (Groq, Llama 3.3 70B)

as an environment variable, or in `.streamlit/secrets.toml` (locally) or the app's **Secrets** settings (on Streamlit Cloud):

```toml
GEMINI_API_KEY = "your-key"
```

The model never receives your full dataset and never runs free-form code. It can only request queries from a fixed allow-list,
answers are shown with evidence cards, and the numbers in each answer are checked against the computed results. Usage is capped
per session. Evaluate the analyst with `python -m datanaut eval-ai`.

## Limits and privacy

- Files are processed in the browser session only; nothing is stored or shared between sessions.
- Limits (in [`limits.py`](src/datanaut/limits.py)): 35 MB per upload, 750,000 rows, 200 columns, 20 AI questions per session.
  Charts plot at most 5,000 points; the full data is still used for every calculation.
- Results describe **association, not cause**.

## Testing and CI

```powershell
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

GitHub Actions runs lint, formatting and the full test suite on Python 3.11 and 3.12, plus a separate
`pip-audit` of the pinned dependencies. No test uses the network. `scripts/stress_test.py` measures what a given
file size costs in time and memory.

## Deployment

The app is deployed on **Streamlit Community Cloud** at <https://datanaut6.streamlit.app>.
To deploy your own copy: push the repo to GitHub, create a new app on [share.streamlit.io](https://share.streamlit.io)
pointing at `app.py`, and add any AI keys under **Secrets**. `requirements.txt` pins the runtime dependencies.

## Sample data

- `sample_data/household_energy_sample.csv`: hourly household energy readings with outdoor temperature (used by **Use sample data**).
- `sample_data/demo_energy_12_months.csv` / `.xlsx`: a 12-month demo dataset.
- Regenerate with `scripts/make_sample.py` and `scripts/make_demo_dataset.py`.
