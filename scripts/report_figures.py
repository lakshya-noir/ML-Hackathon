"""Draw report figures only from recorded development experiments."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/figures');OUT.mkdir(parents=True,exist_ok=True)
d=json.loads(Path('results/model_comparison.json').read_text())
names=list(d['models']);short=['Logistic','Gaussian NB','Decision tree','Random forest','Hist boosting']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
def finish(fig,name):
    fig.tight_layout();fig.savefig(OUT/name,dpi=170,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,2.8))
vals=[d['models'][n]['mean_cv_candidate_binary_macro_f1'] for n in names]
sd=[np.std(d['models'][n]['cv_candidate_binary_macro_f1']) for n in names]
ax.bar(short,vals,yerr=sd,capsize=4,color='#355c7d');ax.set(ylabel='Candidate binary macro F1',ylim=(0,1))
for i,v in enumerate(vals):ax.text(i,v+.035,f'{v:.4f}',ha='center',fontsize=10)
finish(fig,'cv_comparison.png')
fig,ax=plt.subplots(figsize=(8,2.8));x=np.arange(5);w=.36
ax.bar(x-w/2,[d['models'][n]['top1_accuracy'] for n in names],w,label='Observation Top-1',color='#355c7d')
ax.bar(x+w/2,[d['models'][n]['observation_arc_macro_f1'] for n in names],w,label='Observation arc macro F1',color='#b97535')
ax.set(xticks=x,xticklabels=short,ylabel='Score',ylim=(0,1.12));ax.legend(loc='upper left',ncol=2,frameon=False)
finish(fig,'observation_comparison.png')
fig,ax=plt.subplots(figsize=(8,2.8));vals=[d['models'][n]['conditional_candidate_log_loss'] for n in names]
ax.bar(short,vals,color='#355c7d');ax.set(ylabel='Conditional log loss in nats',ylim=(0,1.02))
for i,v in enumerate(vals):ax.text(i,v+.025,f'{v:.4f}',ha='center',fontsize=10)
finish(fig,'loss_comparison.png')
trials=json.loads(Path('results/tuning_trials.json').read_text());scores=[r['value'] for r in trials]
fig,ax=plt.subplots(figsize=(8,2.6));ax.plot(np.arange(1,len(scores)+1),scores,'o-',label='Trial CV objective',color='#355c7d')
ax.plot(np.arange(1,len(scores)+1),np.maximum.accumulate(scores),'--',label='Best so far',color='#b97535')
ax.axvline(10.5,color='#999999',linestyle=':',label='End of startup exploration');ax.set(xlabel='Trial number',ylabel='CV binary macro F1',xticks=range(1,13));ax.legend(fontsize=9,frameon=False)
finish(fig,'tuning_trials.png')
r=json.loads(Path('outputs/model-improvement/20261118-120758/report.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(8,2.8));keys=['nearest_200','monotone_200','monotone_more'];labs=['Nearest 200','Monotone 200','Monotone 1000']
axes[0].bar(labs,[r['experiments'][k]['cv_top1'] for k in keys],color='#355c7d');axes[0].set(ylabel='Grouped validation Top-1',ylim=(0,1))
axes[1].bar(labs,[r['student_test_means'][k]['hmm']['point_accuracy'] for k in keys],color='#b97535');axes[1].set(ylabel='Student-test mean HMM agreement',ylim=(0,1))
for ax in axes:ax.tick_params(axis='x',labelrotation=15)
finish(fig,'controlled_improvement.png')
print('Generated five report comparison figures.')
