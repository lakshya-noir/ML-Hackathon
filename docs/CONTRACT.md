# Shared integration contract — version 1

Agree on changes together before changing function signatures or API fields.

## Data and units

- Coordinates everywhere: `(longitude, latitude)` in WGS84.
- Physical distances: metres. Timestamps and durations: seconds.
- `.nodes`: two columns, longitude and latitude; 0-based row index is node ID.
- `.arcs`: from-node ID and to-node ID; 0-based row index is local arc ID.
- `.track`: longitude, latitude, timestamp; observation index is 0-based row index.
- `.route`: chronological sequence of reference arc IDs; NOT one label per GPS point.
- IDs are local to each trip. Do not train a global classifier over raw arc IDs.
- Preserve original observation indices through preprocessing.
- Fit learned transformations only on permitted training folds. Cross-validation groups are trip IDs.
- Runtime prediction must load only track, nodes and arcs. Evaluation reads `.route` separately.

## Model ↔ decoder

Implemented interfaces:

```python
# pathsnap/features/candidates.py
generate_candidates(trip) -> list[list[dict]]
# Outer list follows track order. Candidate:
# {"arc_id": int, "distance_m": float, "projected_lon": float,
#  "projected_lat": float, "features": list[float]}
# Feature names/order must be saved with the model.

# pathsnap/models/predict.py
score_candidates(trip, candidates, artifact_dir) -> list[list[float]]
# Same nested shape/order as candidates. Nonnegative normalized scores per
# nonempty observation candidate set. Document conversion/calibration used.

# pathsnap/decoder/viterbi.py
decode(trip, candidates, scores, config) -> dict
# {"observation_arc_ids": list[int | None],
#  "route_arc_ids": list[int], "warnings": list[str]}
```

An empty candidate list must retain its observation position. Missing predictions use `None`, not an invented arc. Decoder owns transition feasibility, route completion and explicit gap policy. Backend builds geometry from decoded graph arcs; do not draw straight shortcuts across disconnected sections.

Initial experiments use nearest-reference labels; final training/evaluation use monotone route alignment. Both remain geometric surrogate labels. Never silently zip `.track` with `.route`.

## API ↔ UI

- `GET /health` → `{"status": "ok"}` for server reachability. This alone does not establish model readiness.
- `GET /trips` → `{"trips": [{"trip_id": "00000000"}]}` for the configured allowed partition only.
- `POST /predict` with `{"trip_id": "00000000"}`.
- Successful response fields:
  - `trip_id`: string
  - `observation_arc_ids`: integer or null per original observation
  - `route_arc_ids`: ordered predicted local arc IDs
  - `route_geojson`: GeoJSON geometry in longitude/latitude order; preserve gaps as separate lines
  - `metrics`: named measured values; unavailable metrics are null
  - `timing_ms`: processing timings
  - `warnings`: list of strings
- Unknown trip: 404. Unavailable model/assets: 503. Do not substitute fake predictions.
- API resolves trip IDs against its allowed local registry, never arbitrary client paths.
- Ground truth stays server-side and outside the prediction pipeline. Do not return reference routes or reference labels to the evaluator UI.
- Browser UI requires appropriate CORS; non-browser server-side requests do not.

## Demo network

Run the backend on Laptop 1 with `--host 0.0.0.0 --port 8000` . Laptop 2 must use `http://<laptop-1-LAN-IP>:8000`, not localhost. Test `/health` immediately on the actual network. Wi-Fi client isolation may require a shared hotspot.

## Evidence

Record model configuration, seed, fold trip IDs, feature schema, label-alignment method, metrics and timing. Keep five-model comparison and tuning trial records in `results/`. Ground-truth-based metrics must state their alignment assumptions.
