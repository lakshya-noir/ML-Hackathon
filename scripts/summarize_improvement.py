"""Plot a completed improvement experiment without rerunning training."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('run_dir')
a=ap.parse_args();out=Path(a.run_dir)
r=json.loads((out/'report.json').read_text())
assert r['status'].startswith('complete')
fig,axes=plt.subplots(1,3,figsize=(13,4.5))
names=['nearest_200','monotone_200','monotone_more']
labels=['Nearest\n200/trip','Monotone\n200/trip','Monotone\n1000/trip']
for ax,values,title,ylabel in [
    (axes[0],[r['experiments'][n]['cv_top1'] for n in names],
     'Grouped training validation','Mean fold/trip Top-1 agreement (%)'),
    (axes[1],[r['student_test_means'][n]['hmm']['point_accuracy'] for n in names],
     'Full student-test trips + HMM','Mean trip Top-1 agreement (%)'),
    (axes[2],[r['student_test_means'][n]['hmm']['route_f1'] for n in names],
     'Full student-test trips + HMM','Mean trip road-overlap F1 (%)')]:
    ax.bar(labels,[100*v for v in values],color=['#999999','#5586ad','#226b9b'])
    ax.set(title=title,ylabel=ylabel,xlabel='Training label method and sampling cap',ylim=(0,108))
    for i,v in enumerate(values): ax.text(i,100*v+1,f'{100*v:.2f}',ha='center',fontsize=9)
fig.text(.02,.015,'Source: this run/report.json; 80 train trips, 3 grouped folds, 10 full student-test trips. All evaluation uses monotone geometric surrogate labels. Same features, classifier settings and HMM.',fontsize=8)
fig.tight_layout(rect=(0,.07,1,1));fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
print(out/'comparison.png')
