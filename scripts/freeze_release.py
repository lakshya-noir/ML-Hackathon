"""Create first immutable release manifest and copy verified model; refuses overwrite."""
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pathsnap.release import ROOT,DEFAULT_MODEL_DIR,DEFAULT_DECODER_CONFIG

source=ROOT/'outputs/model-improvement/20261118-120758/monotone_more/model.joblib'
manifest=ROOT/'artifacts/final_release.json'
if manifest.exists() or DEFAULT_MODEL_DIR.exists():
    raise SystemExit('Release already exists; refusing overwrite.')
DEFAULT_MODEL_DIR.mkdir(parents=True)
shutil.copy2(source,DEFAULT_MODEL_DIR/'model.joblib')
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
release=dict(release_id='pathsnap-monotone-v1',model_dir=str(DEFAULT_MODEL_DIR.relative_to(ROOT)),
    decoder_config=str(DEFAULT_DECODER_CONFIG.relative_to(ROOT)),
    model_sha256=hashfile(source),decoder_config_sha256=hashfile(DEFAULT_DECODER_CONFIG),
    source_experiment='outputs/model-improvement/20261118-120758',
    label_method='monotone route alignment; geometric surrogate',
    training='80 training trips; up to 1000 observations per trip; grouped CV selected model',
    decoder_policy='fixed documented defaults; no separate decoder tuning claimed',
    packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','joblib','fastapi','uvicorn']})
manifest.write_text(json.dumps(release,indent=2)+'\n')
print(json.dumps(release,indent=2))
