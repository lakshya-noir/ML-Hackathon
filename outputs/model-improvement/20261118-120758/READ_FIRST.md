> Historical controlled experiment and candidate verification. The candidate was subsequently frozen as `pathsnap-monotone-v1`; use `artifacts/final_release.json` and `docs/report/FINAL_RESULTS.md` for current defaults. Statements below about the earlier default describe the experiment at that time.

# Controlled model improvement — 12:07 run

**Later update:** this candidate has now been promoted to release `pathsnap-monotone-v1`, default model `artifacts/models/final_monotone_v1`. Historical statements below describe the experiment and temporary-server stage. Use `docs/report/FINAL_RESULTS.md` and `results/api_verification_final.json` for current default results. The original model remains preserved, but is no longer the default.

Completed in about 71 seconds; current live model was not replaced. No evaluator data used.

We tested three versions with identical feature/candidate definitions, tuned classifier hyperparameters and default HMM configuration:

1. Nearest-reference labels; 200 training observations per trip (reproduces the live baseline).
2. Monotone reference labels; 200 per trip (isolates label consistency).
3. Monotone reference labels; up to 1,000 per trip (adds training coverage).

All validation and student-test scores use the same monotone surrogate labels. They still do not establish independently verified physical truth. Initial five-model results elsewhere used a different label definition and sample; do not combine them as a single experiment.

## Measured results

- Grouped-CV Top-1: **83.753% → 84.165% → 84.476%**. Mean over trip scores in each of three held-out folds, then over folds. Version 3 selected using training CV only.
- Candidate training rows: **115,090 → 115,090 → 518,231**.
- Full student-test classifier Top-1, equal-weight mean over ten trips: **86.327% → 87.086% → 87.476%**.
- Full student-test HMM Top-1, equal-weight mean over ten trips: **91.217% → 91.821% → 91.949%**.
- HMM route F1: **96.801% → 96.906% → 96.798%**. Overall road overlap is essentially unchanged, slightly lower for version 3.
- Observation-weighted HMM Top-1, baseline → version 3: **89.671% → 90.412%**. This differs from equal-weight trip averaging because longer trips contribute more observations.
- Trip 77 HMM agreement improves **73.306% → 76.298%**. Trip 0 declines **97.603% → 96.524%**. Improvement is not universal.

The recreated nearest/200 model exactly matched the live model's per-trip outputs in this run. Baseline hash, fold trip IDs, exact scores and per-trip outputs are in `report.json`. New models live in the three named subdirectories. `comparison.png` visualises the tradeoff.

## Recommendation

Version 3 is a reasonable candidate because it wins the declared grouped-CV Top-1 objective and improves held-out pointwise agreement with route F1 practically unchanged. The API still uses the original artifact. Before adoption: test the candidate through the actual API, record its version/path and restart the API to clear artifact/response caches. Then refresh report/demo metrics from that exact artifact. Do not select on a single favourable trip.

Next useful experiment is train-only validation of decoder parameters for the new classifier. More classifier tuning is optional; two-laptop evidence and the final report remain deadline-critical.

## Candidate API verification — passed at 12:14

A separate API instance loaded `monotone_more/model.joblib` on localhost port 8001. `scripts/verify_demo.py` confirmed the intended model directory, served UI assets, browser CORS preflight, fresh inference for all ten trips, response geometry/counts, invalid-trip 404, cached-versus-fresh deterministic output and every classifier/HMM metric against `report.json` (API rounding tolerance 0.000051).

Evidence: `api_verification.json`, including candidate artifact SHA-256. The baseline artifact checksum remained unchanged. These are HTTP/interface checks, not a rendered-browser or physical two-laptop test. The temporary candidate server was stopped after verification; the original demo model remains the default.

To reproduce candidate serving on macOS/Linux from the repository root:

```bash
PATHSNAP_MODEL_DIR=outputs/model-improvement/20261118-120758/monotone_more \
  .venv/bin/python -m pathsnap.api.app --host 127.0.0.1 --port 8001
```

In a second terminal, with the static UI server running on port 5500:

```bash
python3 scripts/verify_demo.py --api http://127.0.0.1:8001 \
  --expected-model-dir outputs/model-improvement/20261118-120758/monotone_more \
  --experiment-report outputs/model-improvement/20261118-120758/report.json \
  --output outputs/model-improvement/20261118-120758/api_verification.json
```

For PowerShell set `$env:PATHSNAP_MODEL_DIR` to that same relative directory, then run `python -m pathsnap.api.app --host 127.0.0.1 --port 8001`. Use `--host 0.0.0.0` only when intending another laptop to access this candidate server; the verifier's loopback test alone does not establish remote reachability.
