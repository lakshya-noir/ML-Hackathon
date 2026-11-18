"""PathSnap FastAPI backend -- Laptop 1 (owner: Nakul).

Run from the repository root:
    python -m pathsnap.api.app --host 0.0.0.0 --port 8000
    python -m pathsnap.api.app --trips-dir data/evaluator_test      # final demo

Endpoints (docs/CONTRACT.md):
    GET  /health            reachability + model readiness
    GET  /trips             trip ids in the configured folder (no labels)
    POST /predict           {"trip_id": "00000000"}
    GET  /predict/{trip_id} same, for browser/curl checks
    GET  /logs/recent       recent request log lines (demo screen)

Prediction path: load_trip -> generate_candidates -> score_candidates ->
decode (HMM/Viterbi). The .route file is opened only afterwards, in
evaluation, and only metrics are returned -- never reference arcs/geometry.
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from pathsnap.data import load_reference, load_trip  # noqa: E402
from pathsnap.decoder import DecoderConfig, Lattice, classifier_only, decode  # noqa: E402
from pathsnap.decoder.evaluation import evaluate, monotone_alignment  # noqa: E402
from pathsnap.features.candidates import generate_candidates  # noqa: E402
from pathsnap.models.predict import load_artifact, score_candidates  # noqa: E402
from pathsnap.release import DEFAULT_MODEL_DIR, DEFAULT_DECODER_CONFIG, release_identity

TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR", ROOT / "data" / "student_test"))
MODEL_DIR = Path(os.environ.get("PATHSNAP_MODEL_DIR", DEFAULT_MODEL_DIR))
DECODER_CONFIG = Path(os.environ.get("PATHSNAP_DECODER_CONFIG", DEFAULT_DECODER_CONFIG))
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

log = logging.getLogger("pathsnap")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
for _h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / "api_requests.log", encoding="utf-8")):
    _h.setFormatter(_fmt)
    log.addHandler(_h)
RECENT = deque(maxlen=200)


def note(msg):
    log.info(msg)
    RECENT.append(time.strftime("%H:%M:%S ") + msg)


class State:
    trips_dir = TRIPS_DIR
    model_ok = False
    model_error = None
    release = {}
    cfg = DecoderConfig()
    cache = OrderedDict()
    lock = threading.Lock()


S = State()


def list_trip_ids():
    if not S.trips_dir.exists():
        return []
    return sorted(d.name for d in S.trips_dir.iterdir() if d.is_dir() and (d / f"{d.name}.track").exists())


def load_assets():
    try:
        load_artifact(str(MODEL_DIR))
        S.cfg = DecoderConfig.from_dict(json.loads(DECODER_CONFIG.read_text()))
        S.release = release_identity(MODEL_DIR, DECODER_CONFIG)
        S.model_ok, S.model_error = True, None
    except Exception as e:  # noqa: BLE001
        S.model_ok, S.model_error = False, f"{type(e).__name__}: {e}"
    S.cache.clear()
    note(f"MODEL  {MODEL_DIR} -> {'loaded' if S.model_ok else 'NOT LOADED: ' + str(S.model_error)}")
    note(f"HMM    {S.cfg.to_dict()}")
    note(f"TRIPS  {S.trips_dir} ({len(list_trip_ids())} trips)")


@asynccontextmanager
async def lifespan(_app):
    load_assets()
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    if request.url.path != "/favicon.ico":
        client = request.client.host if request.client else "?"
        note(f"HTTP   {client:<15} {request.method:<4} {request.url.path:<22} -> {resp.status_code} "
             f"in {1000 * (time.perf_counter() - t0):.0f} ms")
    return resp


def _lonlat(trip, node_ids):
    return [[round(float(trip.nodes[n, 0]), 6), round(float(trip.nodes[n, 1]), 6)] for n in node_ids]


def pieces_geojson(trip, pieces):
    """Route pieces -> GeoJSON MultiLineString in (lon, lat); gaps stay gaps."""
    lines = []
    for piece in pieces:
        cur = []
        for a in piece:
            u, v = map(int, trip.arcs[a])
            if cur and cur[-1] != u:  # defensive: never draw across a disconnect
                lines.append(cur)
                cur = []
            if not cur:
                cur = [u]
            cur.append(v)
        if len(cur) > 1:
            lines.append(cur)
    return {"type": "MultiLineString", "coordinates": [_lonlat(trip, l) for l in lines]}


def _r(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def run_prediction(trip_id):
    if trip_id not in list_trip_ids():  # resolve only against the allowed registry
        raise HTTPException(404, f"unknown trip_id {trip_id!r}")
    if not S.model_ok:
        raise HTTPException(503, f"model not available: {S.model_error}")
    with S.lock:
        if trip_id in S.cache:
            S.cache.move_to_end(trip_id)
            return {**S.cache[trip_id], "cached": True}

    tdir = S.trips_dir / trip_id
    tm = {}
    t0 = time.perf_counter(); trip = load_trip(tdir); tm["load"] = time.perf_counter() - t0
    t0 = time.perf_counter(); cands = generate_candidates(trip); tm["candidates"] = time.perf_counter() - t0
    t0 = time.perf_counter(); scores = score_candidates(trip, cands, MODEL_DIR); tm["scoring"] = time.perf_counter() - t0
    t0 = time.perf_counter(); lat = Lattice(trip, cands, scores, cfg=S.cfg); tm["lattice"] = time.perf_counter() - t0
    t0 = time.perf_counter(); hmm = decode(trip, cands, scores, S.cfg, lattice=lat); tm["viterbi_and_route"] = time.perf_counter() - t0
    clf = classifier_only(trip, cands, scores, lattice=lat)
    tm["prediction_total"] = sum(tm.values())

    # ---- server-side evaluation (reference never leaves this function) ----
    metrics = {"hmm": None, "classifier_only": None, "label_method": "monotone route alignment"}
    ref_path = tdir / f"{trip_id}.route"
    if ref_path.exists():
        t0 = time.perf_counter()
        route = load_reference(tdir)
        labels = monotone_alignment(trip, route)
        metrics["hmm"] = _r(evaluate(trip, hmm, labels, route, lat.graph, cands))
        metrics["classifier_only"] = _r(evaluate(trip, clf, labels, route, lat.graph, cands))
        tm["evaluation"] = time.perf_counter() - t0

    out = {
        "trip_id": trip_id,
        "release": S.release,
        "model": f"{MODEL_DIR.name} (HistGradientBoosting) + HMM/Viterbi",
        "n_observations": int(len(trip.track)),
        "observation_arc_ids": hmm["observation_arc_ids"],
        "route_arc_ids": hmm["route_arc_ids"],
        "route_geojson": pieces_geojson(trip, hmm["route_pieces"]),
        "classifier_route_geojson": pieces_geojson(trip, clf["route_pieces"]),
        "gps_lonlat": [[round(float(x), 6), round(float(y), 6)] for x, y, _ in trip.track],
        "candidate_roads_geojson": {"type": "MultiLineString", "coordinates": [
            _lonlat(trip, trip.arcs[a]) for a in np.unique(lat.arc)]},
        "metrics": metrics,
        "decoder": {**hmm["stats"], "config": S.cfg.to_dict()},
        "timing_ms": {k: round(1000 * v, 1) for k, v in tm.items()},
        "warnings": hmm["warnings"],
        "cached": False,
    }
    h, c = metrics["hmm"] or {}, metrics["classifier_only"] or {}
    nan = float("nan")
    note(f"PRED   trip={trip_id} obs={len(trip.track)} pred={tm['prediction_total']:.2f}s "
         f"pieces={hmm['stats'].get('route_pieces')} | HMM acc={h.get('point_accuracy', nan):.3f} "
         f"routeF1={h.get('route_f1', nan):.3f} | CLF acc={c.get('point_accuracy', nan):.3f} "
         f"routeF1={c.get('route_f1', nan):.3f}" + (f" | warn: {'; '.join(hmm['warnings'])}" if hmm["warnings"] else ""))
    with S.lock:
        S.cache[trip_id] = out
        while len(S.cache) > 12:
            S.cache.popitem(last=False)
    return out


class PredictRequest(BaseModel):
    trip_id: str
    fresh: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": S.model_ok, "model_dir": str(MODEL_DIR),
            "release": S.release,
            "model_error": S.model_error, "decoder_config": S.cfg.to_dict(),
            "trips_dir": S.trips_dir.name, "n_trips": len(list_trip_ids()),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


@app.get("/trips")
def trips():
    out = []
    for tid in list_trip_ids():
        t = np.loadtxt(S.trips_dir / tid / f"{tid}.track", ndmin=2)
        out.append({"trip_id": tid, "n_observations": int(len(t)), "duration_s": float(t[-1, 2] - t[0, 2])})
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    note(f"REQ    predict trip={req.trip_id}")
    if req.fresh:
        with S.lock:
            S.cache.pop(req.trip_id.strip(), None)
    return run_prediction(req.trip_id.strip())


@app.get("/predict/{trip_id}")
def predict_get(trip_id: str):
    note(f"REQ    predict trip={trip_id}")
    return run_prediction(trip_id.strip())


@app.get("/logs/recent")
def recent_logs(n: int = 50):
    return {"lines": list(RECENT)[-n:]}


def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--trips-dir", default=None)
    a = ap.parse_args()
    if a.trips_dir:
        S.trips_dir = Path(a.trips_dir).resolve()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
