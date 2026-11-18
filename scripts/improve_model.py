"""Versioned, controlled model experiments. Never changes the live model.

Compare nearest-label/200-point baseline, monotone-label/200-point training,
and monotone-label/1000-point training. Candidate features/decoder stay fixed.
Select by trip-grouped validation Top-1 against one common monotone label
definition, then report development full-trip metrics for each arm.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from pathsnap.data import load_trip,load_reference
from pathsnap.features.candidates import generate_candidates,FEATURE_NAMES
from pathsnap.features.reference import align_reference
from pathsnap.decoder import Lattice,DecoderConfig,decode,classifier_only
from pathsnap.decoder.evaluation import monotone_alignment,evaluate

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--points',type=int,default=1000)
ap.add_argument('--output',default=None)
args=ap.parse_args()
out=Path(args.output or 'outputs/model-improvement/'+datetime.now().strftime('%Y%m%d-%H%M%S'))
out.mkdir(parents=True,exist_ok=False)
started=time.perf_counter()
baseline_path=Path('artifacts/models/tuned/model.joblib')
original=joblib.load(baseline_path)
cfg=DecoderConfig()
report={'label_caveat':'Both methods derive surrogate labels from reference geometry, not independently verified pointwise ground truth.',
    'baseline_sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    'selection':'mean held-out-trip observation Top-1 agreement with monotone surrogate labels in 3 grouped folds',
    'feature_names':FEATURE_NAMES,'decoder_config':cfg.to_dict(),'experiments':{},'status':'preparing data'}
def save():
    (out/'report.json').write_text(json.dumps(report,indent=2))
save()

def prepare(directory):
    trip=load_trip(directory);route=load_reference(directory)
    candidates=generate_candidates(trip)
    nearest,_,_=align_reference(trip,route)
    monotone=monotone_alignment(trip,route)
    lengths=np.array([len(row) for row in candidates])
    offsets=np.r_[0,np.cumsum(lengths)]
    features=np.array([c['features'] for row in candidates for c in row])
    arc_ids=np.array([c['arc_id'] for row in candidates for c in row])
    return dict(trip=trip,route=route,candidates=candidates,x=features,ids=arc_ids,
        offsets=offsets,nearest=nearest,monotone=monotone)

train=[]
for directory in sorted(Path('data/train').iterdir()):
    record=prepare(directory);train.append(record)
    print('Prepared train',directory.name,len(record['monotone']),flush=True)
# References are used here to construct training/validation labels only.
def matrix(records,label_method,cap):
    xs=[];ys=[]
    for r in records:
        indices=np.unique(np.linspace(0,len(r['monotone'])-1,min(cap,len(r['monotone']))).astype(int))
        for i in indices:
            a,b=r['offsets'][i:i+2]
            xs.append(r['x'][a:b]);ys.append((r['ids'][a:b]==r[label_method][i]).astype(int))
    return np.concatenate(xs),np.concatenate(ys)

def scores(model,r):
    p=np.clip(model.predict_proba(r['x'])[:,1],1e-9,1)
    result=[]
    for a,b in zip(r['offsets'][:-1],r['offsets'][1:]):
        result.append((p[a:b]/p[a:b].sum()).tolist() if b>a else [])
    return result

def top1(model,r):
    p=model.predict_proba(r['x'])[:,1]
    pred=np.array([r['ids'][a+int(np.argmax(p[a:b]))] if b>a else -1
                   for a,b in zip(r['offsets'][:-1],r['offsets'][1:])])
    return float(np.mean(pred==r['monotone']))

splits=list(GroupKFold(n_splits=3).split(np.arange(len(train)),groups=[r['trip'].trip_id for r in train]))
report['folds']=[dict(train_trips=[train[i]['trip'].trip_id for i in a],
                       validation_trips=[train[i]['trip'].trip_id for i in b]) for a,b in splits]
arms=[('nearest_200','nearest',200),('monotone_200','monotone',200),
      ('monotone_more','monotone',args.points)]
for name,method,cap in arms:
    fold_scores=[];t0=time.perf_counter()
    for a,b in splits:
        x,y=matrix([train[i] for i in a],method,cap)
        model=clone(original['model']).fit(x,y)
        values=[top1(model,train[i]) for i in b]
        fold_scores.append(float(np.mean(values)))
        print(name,'CV fold',len(fold_scores),'mean trip Top-1',fold_scores[-1],flush=True)
    x,y=matrix(train,method,cap)
    model=clone(original['model']).fit(x,y)
    target=out/name;target.mkdir()
    joblib.dump(dict(model=model,feature_names=FEATURE_NAMES,label_method=method,
                    max_observations_per_trip=cap),target/'model.joblib')
    report['experiments'][name]=dict(label_method=method,cap=cap,candidate_training_rows=len(y),
        folds=fold_scores,cv_top1=float(np.mean(fold_scores)),training_and_cv_seconds=time.perf_counter()-t0)
    save()
winner=max(report['experiments'],key=lambda k:report['experiments'][k]['cv_top1'])
report['selected_by_training_cv']=winner
report['status']='evaluating student-test full trips';save()
# Every variant is evaluated with the SAME full-trip monotone labels and decoder.
models={'live_tuned':original['model']}
models.update({name:joblib.load(out/name/'model.joblib')['model'] for name,_,_ in arms})
report['student_test']={name:[] for name in models}
for directory in sorted(Path('data/student_test').iterdir()):
    r=prepare(directory)
    for name,model in models.items():
        t0=time.perf_counter();p=scores(model,r)
        lat=Lattice(r['trip'],r['candidates'],p,cfg=cfg)
        clf=classifier_only(r['trip'],r['candidates'],p,lattice=lat)
        hmm=decode(r['trip'],r['candidates'],p,cfg,lattice=lat)
        row=dict(trip_id=directory.name,observations=len(r['monotone']),
            classifier=evaluate(r['trip'],clf,r['monotone'],r['route'],lat.graph,r['candidates']),
            hmm=evaluate(r['trip'],hmm,r['monotone'],r['route'],lat.graph,r['candidates']),
            scoring_lattice_decoder_seconds=time.perf_counter()-t0)
        report['student_test'][name].append(row)
        print('Student test',directory.name,name,'HMM Top-1',row['hmm']['point_accuracy'],flush=True)
    save()
report['student_test_means']={}
for name,rows in report['student_test'].items():
    report['student_test_means'][name]={mode:{metric:float(np.mean([r[mode][metric] for r in rows]))
        for metric in rows[0][mode]} for mode in ('classifier','hmm')}
report['total_seconds']=time.perf_counter()-started
report['status']='complete; no live artifact changed'
assert hashlib.sha256(baseline_path.read_bytes()).hexdigest()==report['baseline_sha256']
save()
print(json.dumps({'output':str(out),'selected_by_training_cv':winner,
                  'student_test_means':report['student_test_means']},indent=2),flush=True)
