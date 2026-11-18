# Frozen final results — use these for report and demo

Release: **pathsnap-monotone-v1**. Model: `artifacts/models/final_monotone_v1/model.joblib`. Decoder: `artifacts/decoder_config.json`.

Model SHA-256: `457a1520c1d9e9da2297a946d1d588f8a564dc7ef6c384240bff8bba83e1b3dd`.
Decoder SHA-256: `fba4744deb30176013f6b226c61c96bf6892d7c3599252a5791293968e94f4fc`.

## Evaluation population and definitions

80 training trips; selected model trained on up to 1,000 observations per trip using monotone labels. Final development evaluation covers all 19,160 observations in ten student-test trips. No evaluator data used. Both modes use the same candidates, classifier and surrogate labels. Summary values are equal-weight means across trips, not pooled observation metrics.

Point labels come from monotone reference-route alignment, still a geometric approximation because the reference route has no timestamps. Point macro F1 uses reference arc classes per trip. Route precision/recall/F1 measure length-weighted sets of arcs, not traversal order. Continuity is the documented break/pair measure, not percent of observations matched correctly.

## Actual metrics

- point_accuracy: classifier **0.8748**; HMM **0.9195**.
- point_macro_f1: classifier **0.9207**; HMM **0.9619**.
- route_precision: classifier **0.7414**; HMM **0.9421**.
- route_recall: classifier **0.9973**; HMM **0.9971**.
- route_f1: classifier **0.8422**; HMM **0.9680**.
- route_continuity: classifier **0.9797**; HMM **0.9987**.
- route_pieces: classifier **17.7000**; HMM **1.6000**.
- candidate_miss_rate: classifier **0.0038**; HMM **0.0038**.

Observation-weighted Top-1: classifier 85.5324%; HMM 90.4123%. Do not interchange these with the equal-trip means above.

HMM improves point agreement on nine trips; trip 00000007 declines. Missing candidates and disconnected route pieces remain. The larger model was chosen by grouped training CV Top-1; it improved held-out point agreement relative to the old model, while mean route F1 stayed essentially unchanged.

## Frozen settings and verification

Decoder settings were retained, not separately tuned: sigma=10m, beta=5m, classifier weight=0.5, gap threshold=10s, outlier threshold=120m, backward tolerance=15m, unreachable log penalty=-40, search base=400m and cap=15000m.

The restarted default API passed ten fresh predictions; every recorded classifier/HMM metric agrees with the offline final evaluation within API rounding. Release hashes agree. CORS/assets/error/cache checks passed. Physical two-laptop reachability remains a separate check.

Mean joint offline candidate/scoring/lattice/decoder pipeline time on this run: 0.507s per trip. This is shared by both mode rows, includes HMM computation, and is not standalone classifier latency. Actual API component times are in `results/api_verification_final.json`; they exclude network/UI time.

## Evidence and reproduction

- `artifacts/final_release.json`: exact release identity, provenance and packages.
- `results/decoder_comparison_student_test.csv`: per-trip final results.
- `results/decoder_summary_student_test.json`: aggregate final results.
- `results/api_verification_final.json`: restarted main API verification.
- `results/final_results.png`: report-ready plot.
- `analysis/maps/student_test_*.html`: regenerated final Folium development maps.

Run `python scripts/evaluate_decoder.py`, then `python scripts/verify_demo.py --expected-model-dir artifacts/models/final_monotone_v1 --release-manifest artifacts/final_release.json --output results/api_verification_final.json`, then `python scripts/final_results_report.py`. Serve API and UI before the HTTP test. Default serving: `python -m pathsnap.api.app --host 0.0.0.0 --port 8000`. Do not rerun `freeze_release.py` over an existing release.

## Historical experiments

Keep the five-model sampled nearest-label comparison and the 12-trial tuning comparison as earlier experiments with their own methods. Keep the controlled improvement report as model-selection evidence. Earlier screenshots/results from the old model are historical and must not be described as final-release screenshots. This document is a report source, not the required consolidated report PDF.
