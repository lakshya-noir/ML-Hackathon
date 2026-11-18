# Final submission coverage

Submit the public repository link and technical report separately. This repository contains the complete implementation and reproducible experiment evidence; report exports and internal coordination files stay local.

| Required component | Source and retained evidence |
|---|---|
| Data processing | `pathsnap/data.py`; `scripts/import_dataset.py`, `inventory.py`, `prepare_evaluation.py`; `results/data_inventory.csv`; `outputs/evaluation_1130/trip_audit.csv` |
| Feature engineering | `pathsnap/features/candidates.py`, `reference.py`; nine-feature schema saved with each model |
| Every model experiment | `scripts/train_models.py`, `improve_model.py`; five baseline weights and tuned/final weights in `artifacts/models/`; controlled variants under `outputs/model-improvement/20261118-120758/` |
| Evaluation and confusion matrices | `scripts/model_diagnostics.py`, `evaluate_decoder.py`, `prepare_evaluation.py`, `report_measurements.py`; `results/model_comparison.json`, diagnostics, confusion outputs and full-trip decoder CSV/JSON |
| Bayesian optimisation | `scripts/train_models.py`; `results/tuning_trials.json`, `tuning.sqlite3`; ranges, objective and outcomes in the report |
| HMM and Viterbi | `pathsnap/decoder/graph.py`, `viterbi.py`, `evaluation.py`; `scripts/test_decoder.py` |
| API and model loading | `pathsnap/api/app.py`, `pathsnap/models/predict.py`, `pathsnap/release.py`; `scripts/verify_demo.py`, `smoke_test_api.py` |
| Full evaluator UI | `ui/index.html`; bundled Leaflet JS/CSS/images and license |
| Deployment | README; pinned requirements; `.gitattributes`; Windows launchers; `docs/TWO_LAPTOP_DEMO.md`; final manifest/config |
| Visualisations | `scripts/make_folium_map.py`, `inspect_maps.py`, `report_figures.py`, `final_results_report.py`; interactive HTML in `analysis/maps/`; quantitative figures in `results/figures/` |
| Supporting results | `results/`, the three retained evidence folders under `outputs/`, and quantitative plot outputs |

## Public versus local files

Publish source, all seven model artifacts, experiment configurations/results, trial history, development maps and quantitative plots. Keep raw organiser/evaluator data, feature caches, `.venv`, runtime logs, credentials, render QA output and duplicate ZIP bundles local. `.gitignore` implements these exclusions. Historical files already in Git remain in history; ignoring a file does not remove it from earlier commits.

No current file exceeds GitHub's ordinary 100 MiB file limit. The largest model is the original Random Forest. Preserve the saved model/config bytes: release SHA-256 values must match the manifest.

## Verification

Run `python scripts/check_submission.py`, `python scripts/test_decoder.py` and the HTTP checks in [REPRODUCING.md](REPRODUCING.md). The frozen manifest must match model/config bytes. Before submission, verify public evaluator access and the actual two-laptop demo separately.
