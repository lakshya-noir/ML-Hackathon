"""Server-side evaluation against the .route reference (owner: Nakul).

Never imported by the prediction path. Observation labels come from a
MONOTONE alignment of GPS points to the ordered .route arcs (dynamic
programming: point i -> route position j_i, j non-decreasing, minimising total
point-to-arc distance). Unlike nearest-arc labelling this respects route order,
so loops / repeated streets get the visit that matches the time order.
It is still a geometric surrogate: the .route file has no timestamps.
"""
import numpy as np
from sklearn.metrics import f1_score

from pathsnap.features.candidates import project, segment_projection


def monotone_alignment(trip, route):
    origin = trip.track[0, :2]
    pts = project(trip.track[:, :2], origin)
    nodes = project(trip.nodes, origin)
    a, b = nodes[trip.arcs[route, 0]], nodes[trip.arcs[route, 1]]
    n, m = len(pts), len(route)
    back = np.empty((n, m), np.int16 if m < 32767 else np.int32)
    idx = np.arange(m)
    prev = None
    for i in range(n):
        d, _ = segment_projection(pts[i], a, b)
        if prev is None:
            cur = d
            back[i] = idx
        else:
            pm = np.minimum.accumulate(prev)
            back[i] = np.maximum.accumulate(np.where(prev <= pm, idx, 0))
            cur = d + pm
        prev = cur
    pos = np.empty(n, int)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return route[pos]


def _route_overlap(pred_arcs, ref_arcs, length):
    p, r = set(map(int, pred_arcs)), set(map(int, ref_arcs))
    inter = sum(length[x] for x in p & r)
    lp, lr = sum(length[x] for x in p), sum(length[x] for x in r)
    prec = inter / lp if lp else 0.0
    rec = inter / lr if lr else 0.0
    return float(prec), float(rec), float(2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def evaluate(trip, decoded, labels, route, graph, candidates=None):
    """Metrics for one decoded result. labels = per-observation reference arcs."""
    pred = np.array([-1 if a is None else a for a in decoded["observation_arc_ids"]])
    lab = np.asarray(labels)
    prec, rec, f1 = _route_overlap(decoded["route_arc_ids"], route, graph.length)
    seq = [a for i, a in enumerate(pred) if a >= 0 and (i == 0 or a != pred[i - 1])]
    adj = np.mean([graph.arcs[x, 1] == graph.arcs[y, 0] for x, y in zip(seq[:-1], seq[1:])]) if len(seq) > 1 else 1.0
    n_pairs = max(1, len(seq) - 1)
    out = {
        "point_accuracy": float(np.mean(pred == lab)),
        "point_macro_f1": float(f1_score(lab, pred, labels=np.unique(lab), average="macro", zero_division=0)),
        "route_precision": prec, "route_recall": rec, "route_f1": f1,
        "route_continuity": 1.0 - (len(decoded["route_pieces"]) - 1) / n_pairs,
        "route_pieces": len(decoded["route_pieces"]),
        "adjacency_rate": float(adj),
        "unmatched_observations": int(np.sum(pred < 0)),
    }
    if candidates is not None:
        out["candidate_miss_rate"] = float(np.mean([l not in {c["arc_id"] for c in row}
                                                    for l, row in zip(lab, candidates)]))
    return out
