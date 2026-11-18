"""Shared raw trip loader. References are deliberately loaded separately."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray


def load_trip(directory: str | Path) -> Trip:
    directory = Path(directory)
    trip_id = directory.name
    track = np.loadtxt(directory / f"{trip_id}.track", ndmin=2)
    nodes = np.loadtxt(directory / f"{trip_id}.nodes", ndmin=2)
    arcs = np.loadtxt(directory / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError(f"Unexpected column counts in trip {trip_id}")
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError(f"Nonfinite coordinates/timestamps in trip {trip_id}")
    if arcs.size and (arcs.min() < 0 or arcs.max() >= len(nodes)):
        raise ValueError(f"Invalid node reference in trip {trip_id}")
    return Trip(trip_id, track, nodes, arcs)


def load_reference(directory: str | Path) -> np.ndarray:
    """For training-label construction/evaluation only, never inference."""
    directory = Path(directory)
    return np.loadtxt(directory / f"{directory.name}.route", dtype=np.int64, ndmin=1)
