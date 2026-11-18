# PathSnap

GPS trajectory map matching using a trained candidate classifier and HMM/Viterbi route reconstruction. FastAPI performs inference and server-side evaluation; a separate HTML/CSS/JavaScript UI displays results using Leaflet.

**Frozen release:** `pathsnap-monotone-v1` — [manifest](artifacts/final_release.json), [decoder settings](artifacts/decoder_config.json), [final results](docs/report/FINAL_RESULTS.md).

## Install

Run from the repository root. Use the recorded Python 3.14 environment where available; a Windows demo also loaded the release under Python 3.13. Keep pinned package versions when loading joblib models.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` includes the pinned demo/ML dependencies. Laptop 2 only needs Python's standard library and `ui/`.

## Data stays local

Raw data, evaluator data, virtual environments and feature caches are excluded from Git. They are not required on Laptop 2.

```bash
python scripts/import_dataset.py '/path/to/Student folder.zip'
python scripts/inventory.py
```

For authorised final evaluation, extract the supplied evaluator ZIP locally so each trip's files are directly under:

```text
data/evaluator_test/TRIP_ID/TRIP_ID.track
data/evaluator_test/TRIP_ID/TRIP_ID.nodes
data/evaluator_test/TRIP_ID/TRIP_ID.arcs
data/evaluator_test/TRIP_ID/TRIP_ID.route
```

Avoid an extra nested `evaluator_test` directory. Track coordinates are longitude, latitude, timestamp. Road IDs are local to each trip; reference routes have no per-observation timestamps.

## Two-laptop deployment

Laptop 1 hosts the model, evaluator data and API:

```bash
python -m pathsnap.api.app --host 0.0.0.0 --port 8003 --trips-dir data/evaluator_test
```

For development select `--trips-dir data/student_test`. No retraining is required. Health must report `model_loaded: true`, release `pathsnap-monotone-v1`, the intended partition and ten trips.

Laptop 2 serves the UI:

```bash
python -m http.server 5500 --bind 127.0.0.1 --directory ui
```

Open `http://localhost:5500`. Set the API field to `http://LAPTOP_1_LAN_IP:8003`, click **Connect**, select a trip and run **fresh inference**. Both laptops need a reachable network. Open `http://LAPTOP_1_LAN_IP:8003/health` from Laptop 2 first. Show receiving API logs during the demo. Basemap tiles need internet; bundled Leaflet and returned geometry do not.

See [deployment and troubleshooting](docs/TWO_LAPTOP_DEMO.md). Port 8000 remains the application default; 8003 is the documented demo example. Match the port in the server command, firewall rule and UI.

## Measured development results

| Final metric | Classifier only | HMM/Viterbi |
|---|---:|---:|
| Point agreement | 87.48% | 91.95% |
| Route F1 | 84.22% | 96.80% |
| Continuity | 97.97% | 99.87% |
| Mean route pieces | 17.7 | 1.6 |

Equal-weight means across ten complete student-test trips (19,160 observations), using monotone geometric surrogate labels. Route F1 measures length-weighted arc-set overlap, not order. HMM point agreement improves on nine trips and declines on trip 7. Earlier five-model results used 2,000 sampled observations and nearest-reference labels; they are a separate experiment.

Five families were compared: Logistic Regression, Gaussian NB, Decision Tree, Random Forest and Histogram Gradient Boosting. Twelve Optuna TPE trials tuned the initial CV winner. A later controlled label/coverage study selected the final model. Decoder settings were fixed, not separately tuned. Saved model variants and their training source are included.

## Source and evidence

| Location | Contents |
|---|---|
| `pathsnap/data.py`, `pathsnap/features/` | Raw loading, retrieval, nine features and original labels |
| `pathsnap/models/` | Artifact loading and normalised candidate scoring |
| `pathsnap/decoder/` | Directed graph, Dijkstra, HMM/Viterbi, route pieces and metrics |
| `pathsnap/api/`, `ui/` | Complete backend and browser interface |
| `scripts/` | Training, tuning, improvement, evaluation, checks and map/report generation |
| `artifacts/` | Five baseline models, tuned/final weights, config and manifest |
| `results/` | CV/tuning records, diagnostics, per-trip metrics and verification |
| `analysis/maps/` | Interactive development Folium maps |
| `outputs/evaluation_1130/`, `outputs/inspection/` | Historical data/feature/model evidence |
| `outputs/model-improvement/20261118-120758/` | Controlled experiment and its model variants |
| `docs/report/FINAL_RESULTS.md` | Frozen development results and metric definitions |

[Submission coverage](docs/SUBMISSION.md) maps every required component. The technical report is submitted separately; internal checklists, viva notes and report authoring files stay local.

## Verify and reproduce

```bash
python scripts/test_decoder.py
python scripts/check_submission.py
# With API/UI running and student-test selected on Laptop 1:
python scripts/verify_demo.py --api http://127.0.0.1:8003 --expected-trips-dir student_test --data-dir data/student_test --release-manifest artifacts/final_release.json --output output/local-checks/api.json
```

Training/report commands and cache/study precautions are in [REPRODUCING.md](docs/REPRODUCING.md). Reproduce in a separate checkout to preserve retained records. Never overwrite the frozen release just to make a test pass.

## Team and assistance

Team: Lakshya, Nakul and Nikhil. [CONTRIBUTING.md](CONTRIBUTING.md) explains responsibilities and genuine contribution recording. Codex and Claude Code assisted implementation, debugging and documentation; actual experiments ran locally. Members remain responsible for reviewing and understanding the submission. Preserve authentic authorship; do not invent commits or results.

Leaflet 1.9.4 is bundled under its [BSD 2-Clause license](ui/vendor/leaflet/LICENSE). Maps attribute OpenStreetMap. Reserved references never enter prediction or the evaluator UI.
