"""Development-only Folium inspection maps and point-level audit evidence."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import folium
import numpy as np
from pathsnap.data import load_trip, load_reference
from pathsnap.features.candidates import generate_candidates
from pathsnap.features.reference import align_reference
from pathsnap.models.predict import score_candidates


def road_layer(map_, trip, arc_ids, name, color, weight, show=True):
    layer = folium.FeatureGroup(name=name, show=show)
    coordinates = [trip.nodes[trip.arcs[int(arc)]].tolist() for arc in np.unique(arc_ids)]
    folium.GeoJson({'type': 'Feature', 'properties': {}, 'geometry': {
        'type': 'MultiLineString', 'coordinates': coordinates}},
        style_function=lambda feature: dict(color=color, weight=weight, opacity=0.8)).add_to(layer)
    layer.add_to(map_)


def inspect(directory, output, model, reason):
    trip = load_trip(directory)
    candidates = generate_candidates(trip)
    scores = score_candidates(trip, candidates, model)
    chosen = [row[int(np.argmax(p))] for row, p in zip(candidates, scores)]
    predicted = np.array([c['arc_id'] for c in chosen])
    route = load_reference(directory)
    labels, offsets, ambiguous = align_reference(trip, route)
    misses = np.array([label not in [c['arc_id'] for c in row] for label,row in zip(labels,candidates)])
    wrong = predicted != labels
    dt = np.diff(trip.track[:,2])
    summary = dict(trip_id=trip.trip_id, partition=directory.parent.name, selection_reason=reason,
        observations=len(labels), surrogate_top1_agreement=float(np.mean(~wrong)),
        candidate_miss_rate=float(misses.mean()), ambiguous_fraction=float(ambiguous.mean()),
        reference_offset_median_m=float(np.median(offsets)), reference_offset_p95_m=float(np.percentile(offsets,95)),
        nonpositive_time_steps=int(np.sum(dt<=0)), gaps_over_1_second=int(np.sum(dt>1)),
        max_time_gap_seconds=float(dt.max()) if len(dt) else None,
        model=model, label_method='nearest reference geometry surrogate, not verified pointwise ground truth')
    title = f'Trip {trip.trip_id} — {reason}; full trip, {len(labels)} points'
    map_ = folium.Map(location=trip.track[0,:2][::-1].tolist(), zoom_start=14, tiles=None)
    folium.TileLayer('OpenStreetMap', name='Street basemap (requires internet)').add_to(map_)
    road_layer(map_,trip,np.arange(len(trip.arcs)),'Road network', '#999999',1,False)
    road_layer(map_,trip,route,'Supplied reference route segments','#159447',6)
    road_layer(map_,trip,predicted,'Classifier-selected road segments (not HMM route)','#ed8b23',3)
    gps = folium.FeatureGroup(name='Raw GPS points')
    disagreements = folium.FeatureGroup(name='Disagrees with surrogate label',show=False)
    ambiguity = folium.FeatureGroup(name='Ambiguous surrogate label',show=False)
    missing = folium.FeatureGroup(name='Surrogate reference missing from candidates',show=False)
    for i, (lon,lat,timestamp) in enumerate(trip.track):
        detail = (f'Observation {i}; time {timestamp:g}s<br>Predicted arc {predicted[i]}; '
            f'surrogate arc {labels[i]}<br>Reference offset {offsets[i]:.1f}m; '
            f'ambiguous={bool(ambiguous[i])}; candidate miss={bool(misses[i])}')
        folium.CircleMarker([lat,lon],radius=2,color='#2878bd',weight=1,
            fill=True,fill_opacity=.7,popup=detail).add_to(gps)
        for condition,layer,color in [(wrong[i],disagreements,'#c92b36'),
                                       (ambiguous[i],ambiguity,'#7847a3'), (misses[i],missing,'#111111')]:
            if condition:
                folium.CircleMarker([lat,lon],radius=5,color=color,weight=2,popup=detail).add_to(layer)
    for layer in (gps,disagreements,ambiguity,missing):
        layer.add_to(map_)
    folium.LayerControl(collapsed=False).add_to(map_)
    map_.fit_bounds([trip.track[:,:2].min(axis=0)[::-1].tolist(), trip.track[:,:2].max(axis=0)[::-1].tolist()])
    legend = (f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;'
        f'padding:12px;max-width:390px;font:13px sans-serif;border:1px solid #999">'
        f'<b>{html.escape(title)}</b><br>Blue: GPS · Green: reference segments · Orange: classifier segments'
        f'<br>Optional layers: red disagreements; purple ambiguous labels; black candidate misses.'
        f'<br>Surrogate agreement: {summary["surrogate_top1_agreement"]:.2%}.'
        '<br>Development inspection only. Labels are approximate. Orange segments are independent '
        'classifier choices; no connecting route is invented. Overlapping directions can look identical.</div>')
    map_.get_root().html.add_child(folium.Element(legend))
    map_.save(str(output/f'{trip.trip_id}.html'))
    with (output/f'{trip.trip_id}_points.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['observation_index','longitude','latitude','timestamp_s','predicted_arc',
            'surrogate_arc','disagreement','ambiguous','candidate_miss','reference_offset_m'])
        for i,point in enumerate(trip.track):
            writer.writerow([i,*point,int(predicted[i]),int(labels[i]),bool(wrong[i]),
                             bool(ambiguous[i]),bool(misses[i]),float(offsets[i])])
    print(json.dumps(summary),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='artifacts/models/tuned')
    parser.add_argument('--output',default='outputs/inspection')
    args=parser.parse_args()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    records=json.loads(Path('results/model_comparison.json').read_text())['data_quality']['student_test']
    with Path('results/data_inventory.csv').open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['partition']=='student_test' and r['file_type']=='track']
    shortest=min(rows,key=lambda r:int(r['rows']))['trip_id']
    longest=max(rows,key=lambda r:int(r['rows']))['trip_id']
    difficult=max((r for r in records if r['trip_id'] not in {shortest,longest}),
                  key=lambda r:r['ambiguous_fraction'])['trip_id']
    selected=[(shortest,'shortest student-test trip'),(longest,'longest student-test trip'),
              (difficult,'highest recorded reference ambiguity among remaining student-test trips')]
    summaries=[inspect(Path('data/student_test')/trip_id,output,args.model,reason) for trip_id,reason in selected]
    (output/'summary.json').write_text(json.dumps(summaries,indent=2))
    links=''.join(f'<li><a href="{s["trip_id"]}.html">{s["trip_id"]}: {html.escape(s["selection_reason"])}</a>'
        f' — {s["observations"]} GPS points; <a href="{s["trip_id"]}_points.csv">point audit CSV</a></li>' for s in summaries)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>PathSnap inspection</title>'
        '<h1>Development map inspection</h1><p>Full-trip classifier predictions compared with approximate labels. '
        'These are separate from the earlier 2,000-point experiment. Open layers and click points for observation IDs.</p><ul>'
        +links+'</ul><p>Street tiles and map JavaScript require internet. Record human findings separately; '
        'generating these maps does not verify label correctness.</p>')


if __name__=='__main__':
    main()
