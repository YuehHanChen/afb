# afb — AlignmentForecastBench browser

Static site generated from `app.py`. Live at:
**https://YuehHanChen.github.io/afb/**

Click any heatmap cell to see the actual MCQs, the misaligned answer, and how
the model voted across 20 samples.

## Run locally

```bash
pip install -r requirements.txt
python -m flask --app app run --port 5070 --debug
# open http://localhost:5070
```

## Rebuild + deploy

The GitHub Action in `.github/workflows/deploy.yml` runs on every push to
`main` — it executes `build_static.py` and publishes `dist/` to GitHub Pages.

Manual rebuild:
```bash
URL_PREFIX=/afb STATIC_MODE=1 python build_static.py
# output: ./dist/
```

## Layout

- `app.py`, `templates/`, `static/` — Flask source.
- `build_static.py` — pre-renders all 3000+ pages into `dist/`.
- `data/` — bundled data the app reads at runtime:
  - `AFB.csv` — per-(target, ft_dataset) aggregated stats.
  - `hp_configs.json` — base-model alias map.
  - `passed/*_passed.jsonl` — MCQ source (16 failure modes, 200 questions each).
  - `eval_results/<dir>/<fm>_{eval.jsonl,summary.json}` — model answers per cell.
