"""Geometric surrogate observation labels, separate from inference.

The supplied route is not timestamped. Nearest reference geometry is an
approximation, especially at loops/intersections; metrics must say so.
"""
import numpy as np
from .candidates import project, segment_projection


def align_reference(trip, route):
    if len(route) == 0 or route.min() < 0 or route.max() >= len(trip.arcs):
        raise ValueError('Invalid reference arc IDs')
    origin = trip.track[0, :2]
    nodes = project(trip.nodes, origin)
    arcs = trip.arcs[route]
    starts, ends = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    points = project(trip.track[:, :2], origin)
    labels, offsets, ambiguous = [], [], []
    for point in points:
        distance, _ = segment_projection(point, starts, ends)
        j = int(np.argmin(distance))
        labels.append(int(route[j]))
        offsets.append(float(distance[j]))
        close = np.unique(route[distance <= distance[j]+2])
        ambiguous.append(len(close)>1)
    return np.array(labels), np.array(offsets), np.array(ambiguous)
