"""Shared frozen defaults for demo, evaluation and development maps."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / 'artifacts/models/final_monotone_v1'
DEFAULT_DECODER_CONFIG = ROOT / 'artifacts/decoder_config.json'


def release_identity(model_dir, config_path):
    model_dir, config_path = Path(model_dir).resolve(), Path(config_path).resolve()
    identity = dict(model_sha256=hashlib.sha256((model_dir/'model.joblib').read_bytes()).hexdigest(),
                    decoder_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    if model_dir == DEFAULT_MODEL_DIR.resolve() and config_path == DEFAULT_DECODER_CONFIG.resolve():
        manifest = json.loads((ROOT/'artifacts/final_release.json').read_text())
        for key,value in identity.items():
            if manifest[key] != value:
                raise ValueError(f'Frozen release integrity mismatch: {key}')
        identity['release_id'] = manifest['release_id']
    else:
        identity['release_id'] = 'custom-override'
    return identity
