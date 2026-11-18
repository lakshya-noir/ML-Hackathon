"""Supplement the technical report without refitting or changing the release.

Reconstruct classifier-only routes for the original sampled five-model
experiment. Check original Top-1 values before recording continuity. This
uses only the permitted student-test partition and the original cached rows.
"""
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from pathsnap.data import load_trip
from pathsnap.decoder import Lattice, DecoderConfig, classifier_only
from pathsnap.decoder.graph import RoadGraph
from pathsnap.decoder.evaluation import evaluate
from pathsnap.models.predict import score_candidates

original = json.loads((ROOT/'results/model_comparison.json').read_text())
cache = joblib.load(ROOT/'outputs/features-200.joblib')
cfg = DecoderConfig.from_dict(json.loads((ROOT/'artifacts/decoder_config.json').read_text()))
names = list(original['models'])
rows = []
correct = {name: 0 for name in names}
total = 0
for record in cache['student_test']:
    trip = load_trip(ROOT/'data/student_test'/record['trip_id'])
    indices = np.unique(np.linspace(0, len(trip.track)-1, min(200,len(trip.track))).astype(int))
    sampled = replace(trip, track=trip.track[indices])
    candidates = [row for row,label in record['observations']]
    labels = np.asarray([label for row,label in record['observations']])
    assert len(indices) == len(candidates)
    graph = RoadGraph(sampled)
    total += len(labels)
    for name in names:
        scores = score_candidates(sampled, candidates, ROOT/'artifacts/models'/name)
        lat = Lattice(sampled,candidates,scores,graph=graph,cfg=cfg)
        start = time.perf_counter()
        decoded = classifier_only(sampled,candidates,scores,lattice=lat)
        route_build_s = time.perf_counter()-start
        pred = np.array(decoded['observation_arc_ids'])
        correct[name] += int(np.sum(pred == labels))
        # Continuity and pieces depend on the predicted arc sequence, not labels.
        seq = [int(a) for i,a in enumerate(pred) if i==0 or a != pred[i-1]]
        continuity = 1-(len(decoded['route_pieces'])-1)/max(1,len(seq)-1)
        rows.append(dict(model=name,trip_id=trip.trip_id,observations=len(labels),
                         continuity=float(continuity),pieces=len(decoded['route_pieces']),
                         route_build_seconds=route_build_s))
    print(record['trip_id'],'complete',flush=True)
for name in names:
    measured=correct[name]/total
    assert abs(measured-original['models'][name]['top1_accuracy']) < 1e-12, (name,measured)
summary={name:dict(sampled_top1=correct[name]/total,
    mean_trip_route_continuity=float(np.mean([r['continuity'] for r in rows if r['model']==name])),
    mean_trip_route_pieces=float(np.mean([r['pieces'] for r in rows if r['model']==name]))) for name in names}
out=ROOT/'results';out.mkdir(exist_ok=True)
(out/'report_supplement.json').write_text(json.dumps(dict(
    scope='post-selection reporting measurement; no refitting or setting changes',
    partition='student_test',observations=total,trip_count=10,
    sample='same original 200 evenly spaced observations per trip, cached original candidate features',
    route_policy='classifier argmax plus directed shortest-path connectors; build_route default 1500m limit',
    aggregation='continuity is an equal-weight mean over the ten sampled trip sequences',
    definition='1 - (pieces - 1) / max(1, consecutive distinct predicted arc pairs)',
    top1_reproduces_original=True,models=summary),indent=2))
with (out/'report_supplement_by_trip.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
