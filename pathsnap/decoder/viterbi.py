"""HMM map matching + Viterbi decoding (owner: Nakul).

Contract: decode(trip, candidates, scores, config) -> dict
    {"observation_arc_ids": [int | None per .track row],
     "route_arc_ids": [int, ...],        # continuous where the network allows
     "route_pieces": [[int, ...], ...],  # route split where no path exists
     "warnings": [str, ...], "stats": {...}}

Hidden state  : candidate arc (with its projection point) of an observation.
Emission      : log P(z_t | arc) = w * log(score) + (1-w) * (-0.5 (d / sigma)^2)
                score = Lakshya's normalised classifier score, d = GPS-to-arc metres.
Transition    : log P = -| route_dist - gps_dist | / beta_t   (Newson & Krumm 2009)
                route_dist = directed driving distance between the two projection
                points; gps_dist = straight line between the GPS fixes;
                beta_t = beta * sqrt(max(1, dt)), doubled when dt > gap_s.
Gap/robustness policy
  * Empty candidate set   -> observation skipped by the HMM, output None, warning.
  * Unreachable pair      -> log P = unreachable_logp (finite), so the chain never
                             dies; a step where EVERY pair is unreachable is a
                             recorded route break and the route is split there
                             (no straight shortcut is invented).
  * GPS gap (dt > gap_s)  -> wider transition tolerance and a search limit that
                             grows with the straight-line jump (capped).
  * Outlier fix (nearest candidate > outlier_m) -> uniform emission.
  * Backward jitter on the same arc up to backtrack_tol_m counts as 0 m.
  * Repeated point (dt = 0, same position) -> same-arc transition costs 0.
"""
from dataclasses import asdict, dataclass

import numpy as np

from pathsnap.features.candidates import project
from .graph import RoadGraph

FRACTION_INDEX = 4  # 'projection_fraction' in FEATURE_NAMES


@dataclass
class DecoderConfig:
    sigma_m: float = 10.0
    beta_m: float = 5.0
    clf_weight: float = 0.5
    unreachable_logp: float = -40.0
    gap_s: float = 10.0
    outlier_m: float = 120.0
    backtrack_tol_m: float = 15.0
    search_base_m: float = 400.0
    search_cap_m: float = 15000.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def _flatten(candidates, scores):
    obs, arc, frac, dist, sc = [], [], [], [], []
    for i, (row, srow) in enumerate(zip(candidates, scores)):
        for c, s in zip(row, srow):
            obs.append(i)
            arc.append(c["arc_id"])
            frac.append(c["features"][FRACTION_INDEX])
            dist.append(c["distance_m"])
            sc.append(s)
    return (np.array(obs, int), np.array(arc, int), np.array(frac, float),
            np.array(dist, float), np.array(sc, float))


class Lattice:
    """Candidate layout + driving distances between consecutive steps.
    Built once per trip; decode parameters can then be varied cheaply."""

    def __init__(self, trip, candidates, scores, graph=None, cfg=None):
        cfg = cfg or DecoderConfig()
        self.graph = graph or RoadGraph(trip)
        g = self.graph
        self.n_obs = len(candidates)
        self.obs, self.arc, self.frac, self.dist, self.score = _flatten(candidates, scores)
        self.steps = np.unique(self.obs)  # observations that have candidates
        self.starts = np.searchsorted(self.obs, self.steps, "left")
        self.ends = np.searchsorted(self.obs, self.steps, "right")
        xy = project(trip.track[:, :2], g.origin)[self.steps]
        t = trip.track[self.steps, 2]
        self.gps_d = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)]
        self.dt = np.r_[0.0, np.diff(t)]
        self.min_dist = np.array([self.dist[a:b].min() for a, b in zip(self.starts, self.ends)])
        L = g.length[self.arc]
        pos, rem = self.frac * L, (1 - self.frac) * L
        frm, to = g.arcs[self.arc, 0], g.arcs[self.arc, 1]
        self.route_d = [None]
        for s in range(1, len(self.steps)):
            p0, p1, c0, c1 = self.starts[s - 1], self.ends[s - 1], self.starts[s], self.ends[s]
            limit = min(cfg.search_cap_m, cfg.search_base_m + 2.0 * self.gps_d[s])
            M = np.full((p1 - p0, c1 - c0), np.inf)
            cur_arc, cur_pos, cur_from = self.arc[c0:c1], pos[c0:c1], frm[c0:c1]
            for i, p in enumerate(range(p0, p1)):
                fwd = (cur_arc == self.arc[p]) & (cur_pos >= pos[p] - cfg.backtrack_tol_m)
                M[i, fwd] = np.maximum(0.0, cur_pos[fwd] - pos[p])
                dmap, _ = g.search(int(to[p]), limit)
                for j in np.flatnonzero(~fwd):
                    dn = dmap.get(int(cur_from[j]))
                    if dn is not None:
                        M[i, j] = rem[p] + dn + cur_pos[j]
            self.route_d.append(M)


def viterbi_rows(lat: Lattice, cfg: DecoderConfig):
    """Best flat candidate row per step + list of break steps."""
    gauss = -0.5 * (lat.dist / cfg.sigma_m) ** 2
    le = cfg.clf_weight * np.log(np.clip(lat.score, 1e-9, 1)) + (1 - cfg.clf_weight) * gauss
    outlier_step = lat.min_dist > cfg.outlier_m
    le = np.where(np.repeat(outlier_step, lat.ends - lat.starts), 0.0, le)
    n = len(lat.steps)
    score = le[lat.starts[0]:lat.ends[0]].copy()
    back, breaks = [None], []
    for s in range(1, n):
        M = lat.route_d[s]
        beta_t = cfg.beta_m * np.sqrt(max(1.0, lat.dt[s])) * (2.0 if lat.dt[s] > cfg.gap_s else 1.0)
        ok = np.isfinite(M)
        lt = np.where(ok, -np.abs(np.where(ok, M, 0.0) - lat.gps_d[s]) / beta_t, cfg.unreachable_logp)
        if not ok.any():
            breaks.append(s)
        tot = score[:, None] + lt
        bp = np.argmax(tot, axis=0)
        score = tot[bp, np.arange(tot.shape[1])] + le[lat.starts[s]:lat.ends[s]]
        score -= score.max()
        back.append(bp)
    best = np.empty(n, int)
    k = int(np.argmax(score))
    for s in range(n - 1, -1, -1):
        best[s] = lat.starts[s] + k
        if s > 0:
            k = int(back[s][k])
    return best, breaks, int(outlier_step.sum())


def build_route(arc_seq, graph: RoadGraph, limit=1500.0):
    """Join consecutive distinct arcs by shortest directed paths.
    Returns pieces: list of continuous arc lists (split where no path exists)."""
    seq = [int(a) for i, a in enumerate(arc_seq) if i == 0 or a != arc_seq[i - 1]]
    if not seq:
        return []
    pieces, cur = [], [seq[0]]
    for a, b in zip(seq[:-1], seq[1:]):
        mid = graph.path_arcs(int(graph.arcs[a, 1]), int(graph.arcs[b, 0]), limit)
        if mid is None:
            pieces.append(cur)
            cur = [b]
        else:
            cur.extend(mid)
            cur.append(b)
    pieces.append(cur)
    return pieces


def _package(lat, rows, breaks, n_outlier, warnings, mode):
    obs_arcs = [None] * lat.n_obs
    for s, r in zip(lat.steps, rows):
        obs_arcs[int(s)] = int(lat.arc[r])
    pieces = build_route(lat.arc[rows].tolist(), lat.graph)
    if len(pieces) > 1:
        warnings.append(f"{mode}: route split into {len(pieces)} pieces (no drivable path between "
                        f"{len(pieces) - 1} consecutive matched arcs)")
    return {"observation_arc_ids": obs_arcs,
            "route_arc_ids": [a for p in pieces for a in p],
            "route_pieces": pieces,
            "warnings": warnings,
            "stats": {"hmm_break_steps": len(breaks), "outlier_observations": n_outlier,
                      "route_pieces": len(pieces),
                      "mean_candidates": float(len(lat.arc) / max(1, len(lat.steps)))}}


def _base_warnings(trip, lat):
    w = []
    empty = lat.n_obs - len(lat.steps)
    if empty:
        w.append(f"{empty} observations had no candidates; their arc is None")
    if np.any(np.diff(trip.track[:, 2]) < 0):
        w.append("timestamps are not monotonic; decoded in file order")
    big = int(np.sum(lat.dt > 30))
    if big:
        w.append(f"{big} GPS gaps longer than 30 s (max {lat.dt.max():.0f} s); gap policy applied")
    return w


def decode(trip, candidates, scores, config=None, lattice=None):
    """Classifier scores + HMM/Viterbi -> one arc per observation + route."""
    cfg = config if isinstance(config, DecoderConfig) else DecoderConfig.from_dict(config)
    lat = lattice or Lattice(trip, candidates, scores, cfg=cfg)
    if len(lat.steps) == 0:
        return {"observation_arc_ids": [None] * lat.n_obs, "route_arc_ids": [], "route_pieces": [],
                "warnings": ["no candidates for any observation"], "stats": {}}
    rows, breaks, n_out = viterbi_rows(lat, cfg)
    return _package(lat, rows, breaks, n_out, _base_warnings(trip, lat), "hmm")


def classifier_only(trip, candidates, scores, lattice=None):
    """Baseline: independent argmax of the classifier score per observation."""
    lat = lattice or Lattice(trip, candidates, scores)
    rows = np.array([a + int(np.argmax(lat.score[a:b])) for a, b in zip(lat.starts, lat.ends)], int)
    return _package(lat, rows, [], 0, [], "classifier-only")
