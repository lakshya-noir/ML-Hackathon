# Decoder, API and backend — Nakul

## Files

| Path | Purpose |
|---|---|
| `pathsnap/decoder/graph.py` | Directed trip-local road graph; bounded, cached Dijkstra; shortest-path arcs |
| `pathsnap/decoder/viterbi.py` | `decode(trip, candidates, scores, config)` (contract), `classifier_only`, HMM lattice, Viterbi, route building |
| `pathsnap/decoder/evaluation.py` | Server-side metrics; monotone observation↔route alignment (reads `.route`, never used by prediction) |
| `pathsnap/api/app.py` | FastAPI: `/health`, `/trips`, `/predict`, `/predict/{id}`, `/logs/recent`; request log in `logs/api_requests.log` |
| `ui/index.html` | Evaluator UI (Leaflet bundled in `ui/vendor`, works offline except background tiles) |
| `scripts/evaluate_decoder.py` | Classifier-only vs HMM on full trips → `results/decoder_comparison_*.csv`, `results/decoder_summary_*.json` |
| `scripts/test_decoder.py` | Synthetic edge-case checks (H4) |
| `scripts/make_folium_map.py` | Folium maps → `analysis/maps/*.html` |
| `scripts/smoke_test_api.py` | Calls a running API for every trip (run from Laptop 2) |
| `run_api.bat`, `run_ui.bat` | Windows launchers |

## Run

```bash
python -m pip install -r requirements.txt -r requirements-ml.txt
python scripts/test_decoder.py                      # 6/6 PASS expected
python scripts/evaluate_decoder.py                  # all 10 student_test trips
python -m pathsnap.api.app --host 0.0.0.0 --port 8000           # Laptop 1
python -m pathsnap.api.app --host 0.0.0.0 --port 8000 --trips-dir data/evaluator_test   # final demo
cd ui && python -m http.server 5500                 # Laptop 2, open http://localhost:5500
python scripts/smoke_test_api.py --api http://<laptop-1-ip>:8000
```

Windows firewall: allow Python on private networks when prompted, or the UI cannot reach port 8000.
Environment overrides: `PATHSNAP_TRIPS_DIR`, `PATHSNAP_MODEL_DIR` (final default `artifacts/models/final_monotone_v1`), `PATHSNAP_DECODER_CONFIG` (default `artifacts/decoder_config.json`). Shared defaults are in `pathsnap/release.py`. The default pair must match `artifacts/final_release.json`; missing config or checksum mismatch fails model readiness. Custom overrides are marked custom releases.

## HMM design (report section 5)

- **Graph interpretation.** Arcs are directed (`from_node → to_node`). Most streets appear as two opposite arcs, and consecutive `.route` arcs chain head-to-tail (to-node = next from-node in ~99.9% of steps). Distances are metres in a local equirectangular projection centred on the first GPS fix (same as `features/candidates.py`).
- **Hidden states.** For observation *t*, one state per candidate arc from `generate_candidates` (with its projection point). Candidate ↔ arc correspondence is preserved by index; output is one arc (or `None`) per original `.track` row.
- **Emission.** `log e = w·log(score) + (1−w)·(−½(d/σ)²)`, where `score` is the normalised classifier score from `score_candidates` and *d* the GPS-to-arc distance. Defaults: w = 0.5, σ = 10 m. If the nearest candidate is beyond 120 m the observation is treated as an outlier (uniform emission).
- **Transition.** `log a = −|route_dist − gps_dist| / β_t` (Newson & Krumm, 2009). `route_dist` = directed driving distance from the previous projection point to the next (same arc forward → along-arc distance; otherwise remaining arc + Dijkstra between nodes + offset on the new arc). `β_t = β·√max(1, Δt)`, doubled when Δt > 10 s. β = 5 m.
- **Viterbi.** Log-space recurrence `δ_t(j) = max_i [δ_{t−1}(i) + log a_ij] + log e_t(j)`, renormalised each step; back-pointers recover the path.
- **Missing candidates.** Observation keeps its slot with `None`; HMM links the neighbouring observations directly; a warning is returned.
- **Large GPS gaps.** Search limit = 400 m + 2 × straight-line jump (cap 15 km), and β widens with Δt.
- **Disconnected transitions.** Unreachable pairs get a finite penalty (−40), so decoding never fails. If no path exists between consecutive matched arcs, the route is split into separate pieces. No straight shortcut is drawn; the API returns the pieces as separate GeoJSON lines.
- **GPS jitter.** Moving backwards ≤ 15 m on the same arc counts as 0 m.

## Evaluation definitions

- **Observation label.** Monotone dynamic-programming alignment of time-ordered GPS points to the ordered `.route` arcs (non-decreasing route position, minimum total distance). This respects loops and repeated streets, unlike nearest-arc labelling. It remains a geometric surrogate because `.route` has no timestamps.
- **point_accuracy.** Observation Top-1 against that label. **point_macro_f1** is macro F1 over the reference arc IDs.
- **route_precision / recall / F1.** Length-weighted overlap of predicted route arcs with `.route` arcs.
- **route_continuity.** `1 − (pieces − 1) / (consecutive distinct matched arc pairs)`. A value of 1 means one drivable route.
- **candidate_miss_rate.** Share of observations whose label arc is not among the candidates.

## Measured results (student_test, 10 full trips, tuned HGB model, default decoder config)

Source: `results/decoder_summary_student_test.json`, `results/decoder_comparison_student_test.csv`.

| Mode | Point acc | Point macro F1 | Route precision | Route recall | Route F1 | Continuity | Mean pieces |
|---|---|---|---|---|---|---|---|
| Classifier only | 0.8748 | 0.9207 | 0.7414 | 0.9973 | 0.8422 | 0.9797 | 17.7 |
| Classifier + HMM/Viterbi | 0.9195 | 0.9619 | 0.9421 | 0.9971 | 0.9680 | 0.9987 | 1.6 |

Final release `pathsnap-monotone-v1`; equal-weight trip means over all 19,160 student-test observations. HMM improves point agreement on nine trips but trip 00000007 declines. Route F1 improves on nine trips and decreases slightly on trip 7 (0.977250 to 0.976512). Trip 77 is weakest (HMM ~0.763, classifier ~0.680). Candidate miss rate averages ~0.38%. Labels are geometric surrogates; route overlap ignores order and point macro F1 uses reference arc classes per trip.

Refreshed offline joint candidate/scoring/lattice/decoder pipeline: about 0.51s per trip on this machine, excluding loading, evaluation, network and UI. Both mode rows include this joint timing, not standalone classifier latency. API timings/hashes are in `results/api_verification_final.json`; authoritative report references are in `docs/report/FINAL_RESULTS.md`.
