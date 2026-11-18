"""Metric geometry and candidate features; never reads reference routes."""
import numpy as np
from scipy.spatial import cKDTree

FEATURE_NAMES = ['distance_m', 'heading_cos', 'heading_abs_cos', 'segment_length_m',
                 'projection_fraction', 'speed_mps', 'previous_distance_m',
                 'next_distance_m', 'distance_rank']


def project(points, origin):
    return (np.asarray(points) - origin) * np.array([
        111320 * np.cos(np.radians(origin[1])), 111320])


def segment_projection(points, starts, ends):
    delta = ends - starts
    fraction = np.clip(np.sum((points-starts)*delta, axis=-1) /
                       np.maximum(np.sum(delta*delta, axis=-1), 1e-12), 0, 1)
    nearest = starts + fraction[..., None]*delta
    return np.linalg.norm(points-nearest, axis=-1), fraction


def generate_candidates(trip, k=8, radius_m=100):
    origin = trip.track[0, :2]
    xy = project(trip.track[:, :2], origin)
    nodes = project(trip.nodes, origin)
    starts, ends = nodes[trip.arcs[:, 0]], nodes[trip.arcs[:, 1]]
    lengths = np.linalg.norm(ends-starts, axis=1)
    # Sample long segments too: midpoint-only search can overlook their ends.
    counts = np.maximum(1, np.ceil(lengths/30).astype(int))
    arc_ids = np.repeat(np.arange(len(starts)), counts+1)
    fractions = np.concatenate([np.linspace(0, 1, c+1) for c in counts])
    samples = starts[arc_ids] + fractions[:, None]*(ends-starts)[arc_ids]
    tree = cKDTree(samples)
    _, hits = tree.query(xy, k=min(128, len(samples)))
    if hits.ndim == 1:
        hits = hits[:, None]
    previous = xy[np.maximum(np.arange(len(xy))-1, 0)]
    following = xy[np.minimum(np.arange(len(xy))+1, len(xy)-1)]
    heading = following-previous
    elapsed = trip.track[np.minimum(np.arange(len(xy))+1, len(xy)-1), 2] - trip.track[
        np.maximum(np.arange(len(xy))-1, 0), 2]
    speed = np.linalg.norm(heading, axis=1)/np.maximum(elapsed, 1)
    result = []
    scale = np.array([111320*np.cos(np.radians(origin[1])), 111320])
    for i, point in enumerate(xy):
        ids = np.unique(arc_ids[hits[i]])
        distances, fraction = segment_projection(point, starts[ids], ends[ids])
        order = np.argsort(distances)[:k]
        # Keep nearest fallback when GPS is outside the search radius.
        order = order[distances[order] <= radius_m] if distances[order[0]] <= radius_m else order[:1]
        row = []
        for rank, j in enumerate(order):
            arc = int(ids[j])
            delta = ends[arc]-starts[arc]
            cosine = float(np.dot(heading[i], delta)/max(np.linalg.norm(heading[i])*lengths[arc], 1e-9))
            pd, _ = segment_projection(previous[i], starts[arc], ends[arc])
            nd, _ = segment_projection(following[i], starts[arc], ends[arc])
            nearest = (starts[arc]+fraction[j]*delta)/scale+origin
            features = [float(distances[j]), cosine, abs(cosine), float(lengths[arc]),
                        float(fraction[j]), float(speed[i]), float(pd), float(nd), float(rank)]
            row.append(dict(arc_id=arc, distance_m=float(distances[j]),
                            projected_lon=float(nearest[0]), projected_lat=float(nearest[1]),
                            features=features))
        result.append(row)
    return result
