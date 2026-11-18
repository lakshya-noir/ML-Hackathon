# Reproducing the experiments and report

Use a separate checkout and organiser development data. Do not overwrite the frozen release or retained result files merely to reproduce a run. Python dependencies are pinned; the baseline comparison used seed 42 and the saved trip-grouped folds.

## Initial five models and search

```bash
python scripts/import_dataset.py '/path/to/Student folder.zip'
python scripts/train_models.py --max-observations 200 --trials 12
```

This generates the ignored `outputs/features-200.joblib` cache, all baseline models, comparison JSON, tuning JSON and SQLite study. If a compatible study already exists, `--trials 12` appends twelve trials. Start in a fresh checkout/results directory for an exact twelve-trial recreation. Rebuild feature caches after changing labels, features, code or raw data; the original cache filename alone does not validate its contents.

```bash
python scripts/inspect_maps.py
python scripts/prepare_evaluation.py
python scripts/model_diagnostics.py
python scripts/report_measurements.py
```

Diagnostics use the same cached candidate rows and a probability threshold >=0.5. Five-model continuity uses the original sampled sequences and the graph route constructor. These are separate from final full-trip scores. No fitting occurs in the two diagnostics scripts.

## Controlled final-model experiment

```bash
python scripts/improve_model.py --points 1000 --output outputs/reproduction-new
```

This compares nearest/200, monotone/200 and monotone/up-to-1000 training. Selection uses grouped training Top-1 under common monotone validation labels. Settings/features/decoder remain fixed. It creates a new experiment folder and does not replace the frozen inference model. Retained original study: `outputs/model-improvement/20261118-120758/`.

## Frozen evaluation and maps

```bash
python scripts/evaluate_decoder.py
python scripts/make_folium_map.py --all
python scripts/final_results_report.py
```

These defaults use `artifacts/final_release.json` and the saved decoder JSON. Final development evaluation covers every observation in ten student-test trips. Keep equal-trip means distinct from pooled point agreement. No reserved evaluator result may select a model, threshold or parameter.

## HTTP checks

Start the API on the student-test partition and UI on port 5500 in separate terminals:

```bash
python -m pathsnap.api.app --host 127.0.0.1 --port 8003 --trips-dir data/student_test
python -m http.server 5500 --bind 127.0.0.1 --directory ui
python scripts/verify_demo.py --api http://127.0.0.1:8003 --expected-trips-dir student_test --data-dir data/student_test --release-manifest artifacts/final_release.json --output output/local-checks/api.json
```

For a remote demonstration bind the API to `0.0.0.0` and test from the other laptop. Local HTTP success is not remote evidence.

## Plots and packaging checks

```bash
python scripts/report_figures.py
python scripts/check_submission.py
git diff --check
```

Quantitative plots are written to `results/figures/`. The technical report is submitted separately. Its Word/PDF exports, authoring scripts and internal coordination notes are intentionally excluded from the source repository.
