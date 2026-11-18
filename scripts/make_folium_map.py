"""Folium maps of decoded routes (owner: Nakul; development/report use only).

    python scripts/make_folium_map.py --trip 00000056
    python scripts/make_folium_map.py --all                 # all student_test trips
    python scripts/make_folium_map.py --trip 00000011 --partition train

Layers: road network near the trajectory, reference route (.route),
classifier-only route, HMM/Viterbi route, GPS observations; legend + layer
control. Saves analysis/maps/<partition>_<trip>.html. The reference layer is
for development analysis only -- the evaluator UI never receives it.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import folium
import numpy as np

from pathsnap.data import load_reference, load_trip
from pathsnap.decoder import DecoderConfig, Lattice, classifier_only, decode
from pathsnap.decoder.evaluation import evaluate, monotone_alignment
from pathsnap.features.candidates import generate_candidates
from pathsnap.models.predict import score_candidates
from pathsnap.release import DEFAULT_MODEL_DIR, DEFAULT_DECODER_CONFIG, release_identity

OUT = ROOT / "analysis" / "maps"


def latlon_lines(trip, pieces):
    lines = []
    for piece in pieces:
        cur = []
        for a in piece:
            u, v = map(int, trip.arcs[a])
            if cur and cur[-1] != u:
                lines.append(cur); cur = []
            if not cur:
                cur = [u]
            cur.append(v)
        if len(cur) > 1:
            lines.append(cur)
    return [[[float(trip.nodes[n, 1]), float(trip.nodes[n, 0])] for n in l] for l in lines]


def make(partition, tid, model_dir, cfg):
    d = ROOT / "data" / partition / tid
    t = load_trip(d)
    c = generate_candidates(t)
    s = score_candidates(t, c, model_dir)
    lat = Lattice(t, c, s, cfg=cfg)
    h, k = decode(t, c, s, cfg, lattice=lat), classifier_only(t, c, s, lattice=lat)
    route = load_reference(d)
    lab = monotone_alignment(t, route)
    eh, ek = evaluate(t, h, lab, route, lat.graph, c), evaluate(t, k, lab, route, lat.graph, c)
    gps = t.track[:, [1, 0]]
    m = folium.Map(location=gps.mean(0).tolist(), zoom_start=14, control_scale=True)
    m.fit_bounds([gps.min(0).tolist(), gps.max(0).tolist()])
    layers = [
        ("Road network (candidates)", [[int(a)] for a in np.unique(lat.arc)], dict(color="#9ca3af", weight=2, opacity=.6), True),
        ("Reference route (.route)", [list(map(int, route))], dict(color="#111827", weight=8, opacity=.35), True),
        ("Classifier-only route", k["route_pieces"], dict(color="#f97316", weight=3, dash_array="6 6"), False),
        ("HMM/Viterbi route", h["route_pieces"], dict(color="#16a34a", weight=4, opacity=.9), True),
    ]
    for name, pieces, style, show in layers:
        fg = folium.FeatureGroup(name, show=show)
        for line in latlon_lines(t, pieces):
            folium.PolyLine(line, **style).add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup("GPS observations", show=True)
    step = max(1, len(gps) // 2000)
    for i in range(0, len(gps), step):
        folium.CircleMarker(gps[i].tolist(), radius=2, color="#2563eb", fill=True, fill_opacity=.8, weight=0,
                            tooltip=f"obs {i} t={t.track[i, 2]:.0f}s HMM arc={h['observation_arc_ids'][i]}").add_to(fg)
    folium.Marker(gps[0].tolist(), tooltip="start", icon=folium.Icon(color="green")).add_to(fg)
    folium.Marker(gps[-1].tolist(), tooltip="end", icon=folium.Icon(color="red")).add_to(fg)
    fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    sw = '<span style="display:inline-block;width:18px;height:{h}px;background:{c};opacity:{o}"></span>'
    legend = (f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:#fff;padding:10px 12px;'
              f'border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);font:12px/1.7 sans-serif"><b>Trip {tid} ({partition})</b><br>'
              f'<span style="color:#2563eb">&#9679;</span> GPS observation<br>{sw.format(h=6, c="#111827", o=.35)} Reference route<br>'
              f'{sw.format(h=4, c="#16a34a", o=1)} HMM route: acc {eh["point_accuracy"]:.3f}, route F1 {eh["route_f1"]:.3f}<br>'
              f'{sw.format(h=3, c="#f97316", o=1)} Classifier only: acc {ek["point_accuracy"]:.3f}, route F1 {ek["route_f1"]:.3f}<br>'
              f'{sw.format(h=2, c="#9ca3af", o=1)} Road network</div>')
    m.get_root().html.add_child(folium.Element(legend))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{partition}_{tid}.html"
    m.save(str(path))
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    print(f"saved {path.relative_to(ROOT)} | HMM acc {eh['point_accuracy']:.3f} vs classifier {ek['point_accuracy']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip")
    ap.add_argument("--partition", default="student_test", choices=["student_test", "train"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    a = ap.parse_args()
    cfgp = DEFAULT_DECODER_CONFIG
    release_identity(a.model_dir, cfgp)
    cfg = DecoderConfig.from_dict(json.loads(cfgp.read_text()))
    trips = sorted(p.name for p in (ROOT / "data" / a.partition).iterdir()) if a.all else [a.trip]
    for tid in trips:
        make(a.partition, tid, a.model_dir, cfg)
