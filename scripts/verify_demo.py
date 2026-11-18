"""Verify UI assets, CORS, fresh inference and response shapes on a running demo."""
import argparse
import json
from pathlib import Path
import urllib.request
import urllib.error
import hashlib
import math

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--api',default='http://127.0.0.1:8000')
ap.add_argument('--ui',default='http://127.0.0.1:5500')
ap.add_argument('--output',default='results/api_verification_local.json')
ap.add_argument('--expected-model-dir')
ap.add_argument('--experiment-report')
ap.add_argument('--variant',default='monotone_more')
ap.add_argument('--release-manifest')
ap.add_argument('--expected-trips-dir',help='Expected partition directory name from API health')
ap.add_argument('--data-dir',help='Optional local nodes/arcs for structural output checks; never reads route labels')
a=ap.parse_args()
def request(path,body=None):
    req=urllib.request.Request(a.api+path,data=json.dumps(body).encode() if body else None,
        headers={'Content-Type':'application/json','Origin':a.ui})
    with urllib.request.urlopen(req,timeout=300) as response:
        assert response.headers.get('Access-Control-Allow-Origin')=='*'
        return json.load(response)

for asset in ['/','/vendor/leaflet/leaflet.js','/vendor/leaflet/leaflet.css']:
    with urllib.request.urlopen(a.ui+asset,timeout=10) as response:
        assert response.status==200 and len(response.read())>100
req=urllib.request.Request(a.api+'/predict',method='OPTIONS',headers={
    'Origin':a.ui,'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
with urllib.request.urlopen(req) as response:
    assert response.status==200 and response.headers.get('Access-Control-Allow-Origin')=='*'
health=request('/health'); assert health['model_loaded']
if a.expected_trips_dir:
    assert health['trips_dir']==a.expected_trips_dir,'Wrong trip partition selected'
manifest=json.loads(Path(a.release_manifest).read_text()) if a.release_manifest else None
if manifest:
    for key in ['release_id','model_sha256','decoder_config_sha256']:
        assert health['release'][key]==manifest[key],key
if a.expected_model_dir:
    assert Path(health['model_dir']).resolve()==Path(a.expected_model_dir).resolve(), 'Wrong model loaded'
expected=json.loads(Path(a.experiment_report).read_text()) if a.experiment_report else None
expected_trips={r['trip_id']:r for r in expected['student_test'][a.variant]} if expected else {}
trips=request('/trips')['trips']; assert len(trips)==10
out={'scope':'local HTTP checks; not a browser render or physical two-laptop test',
     'api':a.api,'ui':a.ui,'model_loaded':True,'model_dir':health['model_dir'],
     'trips_dir':health['trips_dir'],
     'ui_assets_and_cors':'passed','trips':[]}
if manifest:
    out['release']=health['release']
if a.expected_model_dir:
    out['model_sha256']=hashlib.sha256((Path(a.expected_model_dir)/'model.joblib').read_bytes()).hexdigest()
for trip in trips:
    result=request('/predict',{'trip_id':trip['trip_id'],'fresh':True})
    assert not result['cached']
    if manifest:
        assert result['release']==health['release']
    assert len(result['observation_arc_ids'])==len(result['gps_lonlat'])==result['n_observations']
    assert result['route_geojson']['type']=='MultiLineString'
    assert result['n_observations']==trip['n_observations']
    for line in result['route_geojson']['coordinates']:
        assert len(line)>=2
        for point in line:
            assert len(point)==2 and all(math.isfinite(v) for v in point)
            assert -180<=point[0]<=180 and -90<=point[1]<=90
    if a.data_dir:
        directory=Path(a.data_dir)/trip['trip_id'];tid=trip['trip_id']
        with (directory/f'{tid}.nodes').open() as stream:
            nodes=[tuple(round(float(v),6) for v in line.split()) for line in stream if line.strip()]
        with (directory/f'{tid}.arcs').open() as stream:
            arcs=[tuple(map(int,line.split())) for line in stream if line.strip()]
        assert all(v is None or isinstance(v,int) and 0<=v<len(arcs) for v in result['observation_arc_ids'])
        assert all(isinstance(v,int) and 0<=v<len(arcs) for v in result['route_arc_ids'])
        predicted_edges={(nodes[arcs[v][0]],nodes[arcs[v][1]]) for v in result['route_arc_ids']}
        for line in result['route_geojson']['coordinates']:
            assert all((tuple(p),tuple(q)) in predicted_edges for p,q in zip(line[:-1],line[1:])), 'Geometry shortcut outside predicted arcs'
    assert 'reference_route' not in result and 'reference_arc_ids' not in result
    if expected:
        record=expected_trips[trip['trip_id']]
        for mode,api_mode in [('classifier','classifier_only'),('hmm','hmm')]:
            for key,value in record[mode].items():
                assert abs(result['metrics'][api_mode][key]-value)<=0.000051, (trip['trip_id'],mode,key)
    out['trips'].append({k:result[k] for k in ['trip_id','n_observations','cached','timing_ms','metrics']})
    print(result['trip_id'],'fresh prediction and geometry/observation counts passed',flush=True)
try:
    request('/predict',{'trip_id':'not-a-trip','fresh':True})
    raise AssertionError('Expected 404')
except urllib.error.HTTPError as e:
    assert e.code==404
out['unknown_trip_404']='passed'
if expected:
    baseline=Path('artifacts/models/tuned/model.joblib')
    assert hashlib.sha256(baseline.read_bytes()).hexdigest()==expected['baseline_sha256']
    out['experiment_metric_parity']='passed for both modes, every recorded metric, all ten trips (API rounding tolerance)'
    out['live_baseline_artifact_unchanged']=True
cached=request('/predict',{'trip_id':trips[-1]['trip_id']})
assert cached['cached']
fresh=request('/predict',{'trip_id':trips[-1]['trip_id'],'fresh':True})
assert not fresh['cached'] and fresh['observation_arc_ids']==cached['observation_arc_ids']
out['cache_and_fresh_consistency']='passed'
if a.data_dir:
    out['arc_ids_and_road_geometry']='passed for all trips; no reference labels read by client checker'
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
Path(a.output).write_text(json.dumps(out,indent=2))
print('10/10 fresh predictions, static assets, CORS and 404 check passed; evidence saved.')
