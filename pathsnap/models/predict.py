"""Model scoring contract used by the backend and decoder."""
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
from pathsnap.features.candidates import FEATURE_NAMES


@lru_cache(maxsize=4)
def load_artifact(artifact_dir):
    artifact = joblib.load(Path(artifact_dir)/'model.joblib')
    if artifact['feature_names'] != FEATURE_NAMES:
        raise ValueError('Model feature schema mismatch')
    return artifact


def score_candidates(trip, candidates, artifact_dir):
    model = load_artifact(str(artifact_dir))['model']
    nonempty = [c for row in candidates for c in row]
    if not nonempty:
        return [[] for row in candidates]
    probabilities = model.predict_proba(np.array([c['features'] for c in nonempty]))[:, 1]
    result, offset = [], 0
    for row in candidates:
        p = np.clip(probabilities[offset:offset+len(row)], 1e-9, 1)
        result.append((p/p.sum()).tolist() if len(p) else [])
        offset += len(row)
    return result
