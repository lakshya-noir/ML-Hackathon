"""Audit all development trips and generate reproducible first-evaluation evidence."""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from pathsnap.data import load_trip, load_reference
from pathsnap.features.candidates import project

OUT=Path('outputs/evaluation_1130')
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def save(fig,name):
    fig.tight_layout(rect=(0,.04,1,.96))
    fig.savefig(OUT/name,dpi=160)
    plt.close(fig)


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


inventory=[]
for partition in ('train','student_test'):
    for directory in sorted((Path('data')/partition).iterdir()):
        trip=load_trip(directory);route=load_reference(directory)
        dt=np.diff(trip.track[:,2]);nodes=trip.nodes;arcs=trip.arcs
        if len(route)==0 or route.min()<0 or route.max()>=len(arcs):
            raise ValueError(f'Invalid route in {directory}')
        graph=coo_matrix((np.ones(len(arcs)),(arcs[:,0],arcs[:,1])),shape=(len(nodes),len(nodes))).tocsr()
        components,labels=connected_components(graph,directed=True,connection='weak')
        degrees=np.bincount(arcs.ravel(),minlength=len(nodes))
        xy=project(nodes,trip.track[0,:2]);length=np.linalg.norm(xy[arcs[:,1]]-xy[arcs[:,0]],axis=1)
        coord=trip.track[:,:2]
        ref=arcs[route]
        inventory.append(dict(partition=partition,trip_id=trip.trip_id,observations=len(trip.track),nodes=len(nodes),arcs=len(arcs),
            missing_nonfinite_values=int((~np.isfinite(trip.track)).sum()+(~np.isfinite(nodes)).sum()),
            invalid_lonlat_rows=int(np.sum((abs(coord[:,0])>180)|(abs(coord[:,1])>90))),
            nonpositive_time_steps=int(np.sum(dt<=0)),duplicate_timestamps=int(len(trip.track)-len(np.unique(trip.track[:,2]))),
            exact_1s_steps=int(np.sum(dt==1)),time_steps=len(dt),gaps_over_1s=int(np.sum(dt>1)),
            max_gap_s=float(dt.max()) if len(dt) else 0,median_interval_s=float(np.median(dt)) if len(dt) else 0,
            longitude_min=float(coord[:,0].min()),longitude_max=float(coord[:,0].max()),
            latitude_min=float(coord[:,1].min()),latitude_max=float(coord[:,1].max()),
            weak_components=int(components),isolated_nodes=int(np.sum(degrees==0)),
            zero_length_arcs=int(np.sum(length<1e-6)),reference_arc_count=len(route),
            directed_reference_breaks=int(np.sum(ref[:-1,1]!=ref[1:,0])),
            largest_weak_component_fraction=float(np.bincount(labels).max()/len(nodes))))
write_csv('trip_audit.csv',inventory)
summary={}
for partition in ('train','student_test'):
    rows=[r for r in inventory if r['partition']==partition]
    totals={k:sum(r[k] for r in rows) for k in ['observations','nodes','arcs','missing_nonfinite_values',
        'invalid_lonlat_rows','nonpositive_time_steps','duplicate_timestamps','exact_1s_steps','time_steps',
        'gaps_over_1s','isolated_nodes','zero_length_arcs','directed_reference_breaks']}
    totals.update(trips=len(rows),shortest_trip=min(rows,key=lambda r:r['observations'])['trip_id'],
        longest_trip=max(rows,key=lambda r:r['observations'])['trip_id'],
        max_gap_trip=max(rows,key=lambda r:r['max_gap_s'])['trip_id'],max_gap_s=max(r['max_gap_s'] for r in rows),
        trips_with_multiple_weak_components=sum(r['weak_components']>1 for r in rows))
    summary[partition]=totals
(OUT/'data_summary.json').write_text(json.dumps(summary,indent=2))
print('DATA SUMMARY',json.dumps(summary),flush=True)

fig,axes=plt.subplots(1,3,figsize=(13,4))
for partition,color in [('train','#226b9b'),('student_test','#bb6622')]:
    rows=[r for r in inventory if r['partition']==partition]
    axes[0].hist([r['observations'] for r in rows],bins=15,alpha=.6,label=partition,color=color)
    axes[1].scatter([r['nodes'] for r in rows],[r['arcs'] for r in rows],s=20,label=partition,color=color)
    axes[2].scatter([r['observations'] for r in rows],[r['max_gap_s'] for r in rows],s=20,label=partition,color=color)
for ax,title,x,y in zip(axes,['GPS observations per trip','Road graph sizes','Largest timestamp gap per trip'],
                       ['GPS observations (count)','Nodes (count)','GPS observations (count)'],
                       ['Trips (count)','Arcs (count)','Largest gap (seconds)']):
    ax.set(title=title,xlabel=x,ylabel=y);ax.legend(fontsize=8)
fig.text(.03,.01,'Source: all 80 train and 10 student-test trips; raw development dataset, 18 Nov 2026.',fontsize=9)
save(fig,'01_data_overview.png')

results=json.loads(Path('results/model_comparison.json').read_text())
names=list(results['models']);short=['Logistic','Gaussian NB','Decision tree','Random forest','Hist. boosting']
fig,axes=plt.subplots(1,3,figsize=(14,4.5))
cv=[results['models'][n]['mean_cv_candidate_binary_macro_f1'] for n in names]
sd=[np.std(results['models'][n]['cv_candidate_binary_macro_f1']) for n in names]
axes[0].bar(short,cv,yerr=sd,color='#226b9b',capsize=3)
axes[0].set(title='Grouped-CV candidate binary macro F1',ylabel='Macro F1 (0–1); bars ± fold SD')
x=np.arange(len(names));w=.38
axes[1].bar(x-w/2,[results['models'][n]['top1_accuracy'] for n in names],w,label='Top-1 agreement',color='#226b9b')
axes[1].bar(x+w/2,[results['models'][n]['observation_arc_macro_f1'] for n in names],w,label='Observation arc macro F1',color='#bb6622')
axes[1].set(xticks=x,xticklabels=short,title='Student-test observation metrics',ylabel='Score (0–1)',ylim=(0,1.15));axes[1].legend(fontsize=8)
axes[2].bar(short,[results['models'][n]['conditional_candidate_log_loss'] for n in names],color='#226b9b')
axes[2].set(title='Student-test conditional candidate log loss',ylabel='Mean negative log probability (nats)')
for ax in axes:
    ax.set_xlabel('Model');ax.tick_params(axis='x',labelrotation=35)
fig.text(.02,.005,'Source: recorded 200-point/trip experiment; 3 trip-grouped CV folds; 2,000 test observations. Approximate labels. Log loss excludes candidate misses.',fontsize=8)
save(fig,'02_model_comparison.png')

# Independent diagnostics from existing saved models, without retraining or changing old results.
cache=joblib.load('outputs/features-200.joblib')
def flatten(records):
    x=[];y=[]
    for r in records:
        for row,label in r['observations']:
            for c in row:
                x.append(c['features']);y.append(c['arc_id']==label)
    return np.array(x),np.array(y)
xtrain,ytrain=flatten(cache['train']);xtest,ytest=flatten(cache['student_test'])
diagnostics=[]
for name in names+['tuned']:
    model=joblib.load(Path('artifacts/models')/name/'model.joblib')['model']
    # Match the original recorded binary diagnostics' >= 0.5 threshold,
    # including exact ties; estimator.predict() can resolve ties differently.
    pred_train=model.predict_proba(xtrain)[:,1]>=.5;pred_test=model.predict_proba(xtest)[:,1]>=.5
    diagnostics.append(dict(model=name,training_candidate_binary_macro_f1=float(f1_score(ytrain,pred_train,average='macro')),
        student_test_candidate_binary_macro_f1=float(f1_score(ytest,pred_test,average='macro')),
        student_test_candidate_binary_macro_precision=float(precision_score(ytest,pred_test,average='macro',zero_division=0)),
        student_test_candidate_binary_macro_recall=float(recall_score(ytest,pred_test,average='macro',zero_division=0))))
write_csv('model_diagnostics.csv',diagnostics)

for trip_id in ['00000052','00000056','00000065']:
    trip=load_trip(Path('data/student_test')/trip_id);origin=trip.track[0,:2]
    nodes=project(trip.nodes,origin);points=project(trip.track[:,:2],origin)
    with (Path('outputs/inspection')/f'{trip_id}_points.csv').open() as stream: audit=list(csv.DictReader(stream))
    ref=load_reference(Path('data/student_test')/trip_id)
    predicted=np.unique([int(r['predicted_arc']) for r in audit])
    wrong=np.array([r['disagreement']=='True' for r in audit]);miss=np.array([r['candidate_miss']=='True' for r in audit])
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax in axes:
        ax.add_collection(LineCollection(nodes[trip.arcs],colors='#dddddd',linewidths=.5))
        ax.scatter(points[:,0],points[:,1],s=3,color='#2878bd',label='Raw GPS',zorder=3)
        ax.set(xlabel='East of trip origin (metres)',ylabel='North of trip origin (metres)',aspect='equal')
        ax.set_xlim(points[:,0].min()-100,points[:,0].max()+100);ax.set_ylim(points[:,1].min()-100,points[:,1].max()+100)
    axes[0].add_collection(LineCollection(nodes[trip.arcs[ref]],colors='#159447',linewidths=2,label='Reference segments'))
    axes[0].set_title('Supplied reference road segments')
    axes[1].add_collection(LineCollection(nodes[trip.arcs[predicted]],colors='#ed8b23',linewidths=2,label='Classifier segments'))
    axes[1].scatter(points[wrong,0],points[wrong,1],s=10,color='#c92b36',label='Surrogate disagreement',zorder=4)
    axes[1].scatter(points[miss,0],points[miss,1],s=35,facecolors='none',edgecolors='black',label='Candidate miss',zorder=5)
    axes[1].set_title('Independent classifier selections, before HMM')
    for ax in axes: ax.legend(fontsize=8)
    fig.suptitle(f'Trip {trip_id}: {len(points)} GPS observations, development inspection')
    fig.text(.02,.01,'Source: full student-test trip and saved tuned classifier. Local metric projection. Labels approximate; disconnected segments are not joined.',fontsize=8)
    save(fig,f'map_{trip_id}.png')

# Close-ups make individual disagreement examples readable without HTML.
details=[]
fig,axes=plt.subplots(1,3,figsize=(15,5.8))
for ax,(trip_id,index) in zip(axes,[('00000052',None),('00000056',214),('00000065',267)]):
    trip=load_trip(Path('data/student_test')/trip_id)
    with (Path('outputs/inspection')/f'{trip_id}_points.csv').open() as stream: rows=list(csv.DictReader(stream))
    if index is None: index=next(int(r['observation_index']) for r in rows if r['disagreement']=='True')
    row=rows[index];origin=trip.track[index,:2];nodes=project(trip.nodes,origin);points=project(trip.track[:,:2],origin)
    predicted=int(row['predicted_arc']);reference=int(row['surrogate_arc'])
    ax.add_collection(LineCollection(nodes[trip.arcs],colors='#dddddd',linewidths=1))
    for arc,color,label in [(reference,'#159447','Surrogate reference'),(predicted,'#ed8b23','Classifier choice')]:
        segment=nodes[trip.arcs[arc]]
        ax.plot(segment[:,0],segment[:,1],color=color,lw=4,label=f'{label}: {arc}')
        a,b=segment
        ax.annotate('',xy=a+.65*(b-a),xytext=a+.35*(b-a),arrowprops=dict(arrowstyle='->',color=color,lw=2))
    lo=max(0,index-3);hi=min(len(points),index+4)
    ax.scatter(points[lo:hi,0],points[lo:hi,1],color='#2878bd',s=22,label='Nearby GPS points',zorder=5)
    for i in range(lo,hi): ax.annotate(str(i),points[i],xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.scatter([0],[0],s=65,facecolors='none',edgecolors='#c92b36',zorder=6,label='Inspected GPS point')
    ax.set(xlim=(-120,120),ylim=(-120,120),aspect='equal',title=f'Trip {trip_id}, observation {index}',
        xlabel='East of inspected point (metres)',ylabel='North of inspected point (metres)')
    ax.legend(fontsize=7,loc='upper left')
    reverse=bool(np.array_equal(trip.arcs[predicted],trip.arcs[reference][::-1]))
    details.append(dict(trip_id=trip_id,observation_index=index,predicted_arc=predicted,surrogate_arc=reference,
        reversed_endpoints=reverse,reference_offset_m=float(row['reference_offset_m']),
        candidate_miss=row['candidate_miss']=='True',ambiguous=row['ambiguous']=='True'))
fig.suptitle('Three measured disagreements: directed road arrows and GPS context')
fig.text(.02,.01,'Source: full-trip audit CSVs and raw graphs; ±120m close-ups, local projection. Approximate reference labels can be ambiguous.',fontsize=9)
save(fig,'04_disagreement_closeups.png')
(OUT/'disagreement_details.json').write_text(json.dumps(details,indent=2))
print('DISAGREEMENT DETAILS',json.dumps(details),flush=True)

baseline=results['models'][results['selected_by_group_cv']];tuned=results['tuning']
fig,axes=plt.subplots(1,3,figsize=(11,4))
for ax,key,title in zip(axes,['mean_cv_candidate_binary_macro_f1','top1_accuracy','conditional_candidate_log_loss'],
        ['Grouped-CV binary macro F1','Student-test Top-1 agreement','Conditional candidate log loss']):
    ax.bar(['Baseline','Tuned'],[baseline[key],tuned[key]],color=['#226b9b','#bb6622'])
    ax.set(title=title,xlabel='Model configuration',ylabel='Nats' if 'loss' in key else 'Score (0–1)')
fig.text(.02,.01,'Source: recorded 12-trial Bayesian search; same sampled data and surrogate labels. CV objective improves, held-out Top-1 slightly declines.',fontsize=8)
save(fig,'03_tuning_comparison.png')

images=['01_data_overview.png','02_model_comparison.png','03_tuning_comparison.png','04_disagreement_closeups.png',
        'map_00000052.png','map_00000056.png','map_00000065.png']
(OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>11:30 evaluation evidence</title>'
    '<h1>PathSnap development evidence</h1><p>All numbers measured; surrogate labels; classifier only, before HMM.</p>'
    +''.join(f'<p><a href="{n}">{n}</a></p><img style="max-width:100%" src="{n}">' for n in images))
print('Evidence saved to',OUT,flush=True)
