"""V1/V2: classifier-only vs classifier + HMM/Viterbi on full trips (owner: Nakul).

    python scripts/evaluate_decoder.py                         # all 10 student_test trips
    python scripts/evaluate_decoder.py --partition train --limit 20

Same candidates, scores and labels for both modes. Writes
results/decoder_comparison_<partition>.csv (per trip x mode) and
results/decoder_summary_<partition>.json (means + config + metric definitions).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import csv
import numpy as np

from pathsnap.data import load_reference, load_trip
from pathsnap.decoder import DecoderConfig, Lattice, classifier_only, decode
from pathsnap.decoder.evaluation import evaluate, monotone_alignment
from pathsnap.features.candidates import generate_candidates
from pathsnap.models.predict import score_candidates
from pathsnap.release import DEFAULT_MODEL_DIR, DEFAULT_DECODER_CONFIG, release_identity

DEFS = {
    "point_accuracy": "share of observations whose predicted arc equals the monotone-aligned reference arc (observation Top-1)",
    "point_macro_f1": "macro F1 over reference arc IDs at observation level",
    "route_precision/recall/f1": "length-weighted overlap between predicted route arcs and .route arcs",
    "route_continuity": "1 - (route pieces - 1) / (consecutive distinct matched arc pairs); 1.0 = one drivable route",
    "adjacency_rate": "share of consecutive distinct matched arcs that share a node directly",
    "candidate_miss_rate": "share of observations whose reference arc is not among candidates",
}

ap = argparse.ArgumentParser()
ap.add_argument("--partition", default="student_test", choices=["student_test", "train"])
ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
ap.add_argument("--config", default=str(DEFAULT_DECODER_CONFIG))
ap.add_argument("--output-dir", default=str(ROOT / 'results'))
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
cfg = DecoderConfig.from_dict(json.loads(Path(a.config).read_text()))
identity = release_identity(a.model_dir, a.config)
pdir = ROOT / "data" / a.partition
trips = sorted(p.name for p in pdir.iterdir() if p.is_dir())[: a.limit or None]
rows = []
for tid in trips:
    t = load_trip(pdir / tid)
    t0 = time.perf_counter()
    c = generate_candidates(t)
    s = score_candidates(t, c, a.model_dir)
    lat = Lattice(t, c, s, cfg=cfg)
    h = decode(t, c, s, cfg, lattice=lat)
    secs = time.perf_counter() - t0
    k = classifier_only(t, c, s, lattice=lat)
    route = load_reference(pdir / tid)
    lab = monotone_alignment(t, route)
    for mode, res in (("classifier_only", k), ("hmm_viterbi", h)):
        rows.append({"trip_id": tid, "mode": mode, "n_obs": len(t.track), "pipeline_s": round(secs, 2),
                     "hmm_break_steps": h["stats"]["hmm_break_steps"], **evaluate(t, res, lab, route, lat.graph, c)})
    r0, r1 = rows[-2], rows[-1]
    print(f"{tid}: acc {r0['point_accuracy']:.3f} -> {r1['point_accuracy']:.3f} | route F1 {r0['route_f1']:.3f} -> "
          f"{r1['route_f1']:.3f} | pieces {r0['route_pieces']} -> {r1['route_pieces']} | {secs:.1f}s", flush=True)
out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
with (out / f"decoder_comparison_{a.partition}.csv").open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
keys = [k for k in rows[0] if k not in ('trip_id','mode')]
summ = {mode:{key:round(float(np.mean([r[key] for r in rows if r['mode']==mode])),4)
              for key in keys} for mode in ('classifier_only','hmm_viterbi')}
json.dump({"partition": a.partition, "trips": trips, "model_dir": Path(a.model_dir).name,
           "decoder_config": cfg.to_dict(), "label_method": "monotone route alignment (pathsnap/decoder/evaluation.py)",
           "release": identity, "aggregation": "unweighted mean over trip-level metrics",
           "observations": sum(r['n_obs'] for r in rows if r['mode']=='hmm_viterbi'),
           "metric_definitions": DEFS, "means": summ},
          open(out / f"decoder_summary_{a.partition}.json", "w"), indent=2)
print(json.dumps(summ,indent=2))
