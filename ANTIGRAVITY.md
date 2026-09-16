# Working on WeatherMind in Google Antigravity

This project is already scaffolded and passing its tests — you are not starting from a blank
repo. Paste the relevant section below as your task description.

## First run inside the IDE

```bash
pip install -r requirements.txt
python scripts/generate_training_data.py
python tests/test_pipeline.py
streamlit run app.py
```

Let the agent use its browser-testing capability against `http://localhost:8501` and walk the
seven tabs. Everything should render with no API key configured.

## Ground rules to give the agent

Paste this verbatim — these are the invariants that make the project's contribution claim true,
and an agent optimising for "nicer output" will break them if not told:

> - Never let the counterfactual engine modify any feature listed in
>   `config.IMMUTABLE_FEATURES`. `tests/test_pipeline.py` enforces this; keep it passing.
> - Never replace the threshold band with a single number.
> - Never let the LLM layer produce a risk class, a rainfall figure or an incident that is not
>   already in the payload built by `llm_brief.build_payload`.
> - Keep the app working with no API key: the deterministic template must remain a first-class
>   path, not a degraded error state.
> - Keep `data/historical_incidents.csv` rows marked `illustrative-verify` unless a real source
>   has actually been checked.
> - Run `python tests/test_pipeline.py` after every change to `src/`.

## Good next tasks, in order of demo value

1. **Verify the historical incidents.** Check each row in `data/historical_incidents.csv`
   against a real source, correct the figures, and set `source_status` to the source name.
   Highest credibility-per-hour of anything on this list.
2. **Real rainfall percentiles.** Compute `daily_p90_mm`, `daily_p99_mm` and `daily_record_mm`
   per city from a real IMD or Kaggle rainfall series and overwrite
   `data/rainfall_baselines.csv`. This makes the plausibility filter genuinely empirical.
3. **A second threshold variable in the UI.** `threshold_band` already accepts any actionable
   feature; expose a selector so an officer can ask "how short would this window have to be?"
   as well as "how much rain?".
4. **Per-area calibration display.** Plot predicted probability against the generated labels
   per area to show where the model is over- or under-confident. Feeds the model card.
5. **Feedback-weighted retraining.** Read `logs/officer_feedback.csv`, upweight confirmed rows,
   retrain, and show the before/after on the model card. This closes the loop that is currently
   only described.
6. **PDF export of the decision brief.** Officers want something they can forward.

## Things to deliberately not build

Stated in the plan and worth holding to — a lean stack is part of the design, and each of these
costs demo time while adding nothing a judge will score:

* A FastAPI/React split
* A database
* A full GIS pipeline
* Live weather API integration (the what-if simulator is more demonstrable than a live feed,
  because you control what happens on stage)
