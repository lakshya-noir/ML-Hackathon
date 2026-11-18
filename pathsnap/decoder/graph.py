"""Directed trip-local road graph with bounded, cached Dijkstra (owner: Nakul).

Arcs are treated as DIRECTED (from_node -> to_node): in the data most streets
appear as two opposite arcs, consecutive reference arcs chain head-to-tail
(to_node of one == from_node of the next), so direction is meaningful.
Distances are metres in a local equirectangular projection (same projection as
pathsnap.features.candidates.project, origin = first GPS fix).
"""
import heapq

import numpy as np

from pathsnap.features.candidates import project


class RoadGraph:
    def __init__(self, trip):
        self.origin = trip.track[0, :2]
        self.arcs = trip.arcs
        xy = project(trip.nodes, self.origin)
        self.nodes_xy = xy
        d = xy[self.arcs[:, 1]] - xy[self.arcs[:, 0]]
        self.length = np.hypot(d[:, 0], d[:, 1])
        n = len(xy)
        order = np.argsort(self.arcs[:, 0], kind="stable")
        ptr = np.r_[0, np.cumsum(np.bincount(self.arcs[:, 0], minlength=n))]
        self._to = self.arcs[:, 1].tolist()
        self._len = self.length.tolist()
        self._out = order.tolist()
        self._ptr = ptr.tolist()
        self._cache = {}

    def search(self, src, limit):
        """Shortest distances from node src, up to `limit` metres -> (dist, pred_arc)."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1], hit[2]
        dist, pred, heap = {src: 0.0}, {}, [(0.0, src)]
        to, ln, out, ptr = self._to, self._len, self._out, self._ptr
        while heap:
            du, u = heapq.heappop(heap)
            if du > dist.get(u, np.inf):
                continue
            for k in range(ptr[u], ptr[u + 1]):
                e = out[k]
                v, nd = to[e], du + ln[e]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = e
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = (limit, dist, pred)
        return dist, pred

    def path_arcs(self, src_node, dst_node, limit=1500.0):
        """Arc ids of the shortest directed path, [] if same node, None if unreachable."""
        if src_node == dst_node:
            return []
        dist, pred = self.search(src_node, limit)
        if dst_node not in dist:
            return None
        out, v = [], dst_node
        while v != src_node:
            e = pred[v]
            out.append(e)
            v = int(self.arcs[e, 0])
        return out[::-1]
