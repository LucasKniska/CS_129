## Front-end Model Export

The React UI needs a browser-consumable form of `models/xgb_model_old_era.pkl`. Browsers can’t unpickle Python objects, so we:

1) Convert the trained XGBoost regressor to its native JSON format (`Booster.save_model(...)`) so the trees, thresholds, and leaves are in a language-neutral structure.  
2) Duplicate the ordered feature list (`models/feature_cols_old_era.json`) alongside the JSON model so the UI can assemble inputs in the correct order.

Run `web/convert_model.py` (from the repo root) to regenerate `web/data/xgb_model_old_era.json` and `web/data/feature_cols_old_era.json`.

```bash
python web/convert_model.py
```

### Why JSON?

- JSON is static and can be fetched directly from GitHub Pages.
- The client-side JavaScript can walk the saved tree structure instead of running Python.
- You can either write an interpreter or use a JS/WASM runtime that supports XGBoost JSON.

### Future training runs

If you retrain or adjust `XG_Boost_Exp1.py`, export JSON alongside the pickle:

```python
model.get_booster().save_model("models/xgb_model_old_era.json")
```

Then rerun `web/convert_model.py` (or just copy the feature file) before deploying the UI.

Run `web/extract_roster.py` to flatten `nba_data/final/nba_ml_dataset.csv` into `web/data/rosters.json`, which the React UI loads for teams, years, players, and the “previous season” stat logic.

### Front-end structure

- `index.html`: single-page shell; pulls React/ReactDOM UMD bundles from Unpkg.  
- `app.css`: styles the header, draggable player tiles, and roster footer.  
- `app.js`: uses the raw React API (`React.createElement`) so no JSX/bundler is required. It fetches `data/rosters.json`, lets you pick a team + year, reorders player tiles via drag/drop, shows historical stats, and the “Run model” button currently just opens the site in a new tab—replace the handler once the interpreter is ready.

To refresh roster/model assets after retraining, rerun both scripts so the UI stays in sync.

### Serving the UI

Because the page fetches JSON assets (`rosters.json`, `xgb_model_old_era.json`, `feature_cols_old_era.json`), run it over HTTP:

```bash
cd web
python -m http.server 4173
```

Visit `http://localhost:4173`. The same `web/` directory (including `index.html`, `app.js`, `app.css`, and `data/*`) can be deployed directly to GitHub Pages.
