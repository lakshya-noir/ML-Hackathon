"""H4: synthetic decoder checks (owner: Nakul).   python scripts/test_decoder.py

Builds tiny hand-made road networks and checks: connected route, impossible
(reverse-direction) transition, missing candidate set, repeated point,
large GPS gap and a disconnected graph. Prints PASS/FAIL per case.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from pathsnap.data import Trip
from pathsnap.decoder import decode

DEG = 1 / 111320  # ~1 m in latitude; tests sit on the equator-ish scale


def make_trip(nodes_m, arcs, pts_m, times):
    nodes = np.array(nodes_m, float) * DEG
    track = np.c_[np.array(pts_m, float) * DEG, times]
    return Trip("synthetic", track, nodes, np.array(arcs, int))


def cand(arc, dist, frac):
    return {"arc_id": arc, "distance_m": dist, "projected_lon": 0, "projected_lat": 0,
            "features": [dist, 1, 1, 100, frac, 10, 0, 0, 0]}


results = []


def check(name, cond, detail):
    results.append(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}")


# straight road 0->1->2->3 (100 m arcs) plus reverse arcs 3..5, as two-way street
nodes = [(0, 0), (100, 0), (200, 0), (300, 0)]
arcs = [(0, 1), (1, 2), (2, 3), (1, 0), (2, 1), (3, 2)]
pts = [(10, 3), (60, 3), (110, 3), (160, 3), (210, 3), (260, 3)]
t = make_trip(nodes, arcs, pts, np.arange(6))

# 1) connected route; candidates include the wrong-direction twin with a HIGHER score
C = [[cand(f, 3, (x % 100) / 100), cand(r, 3, 1 - (x % 100) / 100)] for (x, _), f, r in
     zip(pts, [0, 0, 1, 1, 2, 2], [3, 3, 4, 4, 5, 5])]
S = [[0.45, 0.55]] * 6
out = decode(t, C, S)
check("connected route / direction", out["observation_arc_ids"] == [0, 0, 1, 1, 2, 2],
      f"obs arcs {out['observation_arc_ids']}, route {out['route_arc_ids']}")

# 2) impossible transition: a single noisy fix whose top-scored candidate goes backwards
C2 = [list(r) for r in C]
S2 = [[0.9, 0.1]] * 6
S2[3] = [0.2, 0.8]
out = decode(t, C2, S2)
check("impossible transition suppressed", out["observation_arc_ids"][3] == 1, f"obs3 -> {out['observation_arc_ids'][3]}")

# 3) missing candidate set keeps its slot as None
C3 = [list(r) for r in C2]; S3 = [list(r) for r in S2]
C3[2], S3[2] = [], []
out = decode(t, C3, S3)
check("missing candidates -> None", out["observation_arc_ids"][2] is None and len(out["observation_arc_ids"]) == 6,
      f"{out['observation_arc_ids']} warnings={out['warnings']}")

# 4) repeated point (same position and timestamp)
t4 = make_trip(nodes, arcs, pts[:3] + [pts[2]] + pts[3:5], [0, 1, 2, 2, 3, 4])
C4 = C2[:3] + [C2[2]] + C2[3:5]; S4 = [[0.9, 0.1]] * 6
out = decode(t4, C4, S4)
check("repeated point", out["observation_arc_ids"] == [0, 0, 1, 1, 1, 2], f"{out['observation_arc_ids']}")

# 5) large GPS gap (120 s between fixes 2 and 3)
t5 = make_trip(nodes, arcs, pts, [0, 1, 2, 122, 123, 124])
out = decode(t5, C2, [[0.9, 0.1]] * 6)
check("GPS gap", out["observation_arc_ids"] == [0, 0, 1, 1, 2, 2] and any("gap" in w for w in out["warnings"]),
      f"{out['observation_arc_ids']} warnings={out['warnings']}")

# 6) disconnected graph: two separate roads, trip jumps from one to the other
nodes6 = [(0, 0), (100, 0), (0, 500), (100, 500)]
arcs6 = [(0, 1), (2, 3)]
pts6 = [(20, 2), (60, 2), (40, 502), (80, 502)]
t6 = make_trip(nodes6, arcs6, pts6, np.arange(4))
C6 = [[cand(0, 2, .2)], [cand(0, 2, .6)], [cand(1, 2, .4)], [cand(1, 2, .8)]]
out = decode(t6, C6, [[1.0]] * 4)
ok = out["observation_arc_ids"] == [0, 0, 1, 1] and len(out["route_pieces"]) == 2
check("disconnected graph -> split route, no shortcut", ok, f"pieces={out['route_pieces']} warnings={out['warnings']}")

# arc ids valid
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
