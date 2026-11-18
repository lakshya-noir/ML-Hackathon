"""Build final report references only after API/offline/release consistency checks."""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathsnap.release import DEFAULT_MODEL_DIR,DEFAULT_DECODER_CONFIG,release_identity

identity=release_identity(DEFAULT_MODEL_DIR,DEFAULT_DECODER_CONFIG)
summary=json.loads(Path('results/decoder_summary_student_test.json').read_text())
api=json.loads(Path('results/api_verification_final.json').read_text())
assert summary['release']==api['release']==identity
with Path('results/decoder_comparison_student_test.csv').open() as stream:
    rows=list(csv.DictReader(stream))
by_key={(r['trip_id'],r['mode']):r for r in rows}
assert len(by_key)==20 and len(api['trips'])==10
for trip in api['trips']:
    for mode,api_mode in [('classifier_only','classifier_only'),('hmm_viterbi','hmm')]:
        r=by_key[trip['trip_id'],mode]
        for metric,value in trip['metrics'][api_mode].items():
            assert abs(float(r[metric])-value)<=.000051,(trip['trip_id'],metric)
means=summary['means'];c=means['classifier_only'];h=means['hmm_viterbi']
fig,axes=plt.subplots(1,3,figsize=(12,4.5))
for ax,key,title in zip(axes,['point_accuracy','route_f1','route_continuity'],
                        ['Point agreement with surrogate labels','Length-weighted road overlap F1','Route continuity']):
    values=[c[key]*100,h[key]*100]
    ax.bar(['Classifier','HMM/Viterbi'],values,color=['#777777','#226b9b'])
    ax.set(title=title,ylabel='Mean over trips (%)',ylim=(0,108))
    for i,v in enumerate(values): ax.text(i,v+1,f'{v:.2f}%',ha='center')
fig.suptitle(identity['release_id']+' · final development results')
fig.text(.02,.015,'Source: results/decoder_summary_student_test.json; 10 complete student-test trips / 19,160 observations. Monotone geometric surrogate labels; not independent physical truth.',fontsize=8)
fig.tight_layout(rect=(0,.06,1,.94));fig.savefig('results/final_results.png',dpi=160);plt.close(fig)
weighted={mode:sum(float(r['point_accuracy'])*int(r['n_obs']) for r in rows if r['mode']==mode)/19160
          for mode in ('classifier_only','hmm_viterbi')}
lines=['# Frozen final results — use these for report and demo','',
    f'Release: **{identity["release_id"]}**. Model: `artifacts/models/final_monotone_v1/model.joblib`. Decoder: `artifacts/decoder_config.json`.',
    '',f'Model SHA-256: `{identity["model_sha256"]}`.',f'Decoder SHA-256: `{identity["decoder_config_sha256"]}`.','',
    '## Evaluation population and definitions','',
    '80 training trips; selected model trained on up to 1,000 observations per trip using monotone labels. Final development evaluation covers all 19,160 observations in ten student-test trips. No evaluator data used. Both modes use the same candidates, classifier and surrogate labels. Summary values are equal-weight means across trips, not pooled observation metrics.',
    '', 'Point labels come from monotone reference-route alignment, still a geometric approximation because the reference route has no timestamps. Point macro F1 uses reference arc classes per trip. Route precision/recall/F1 measure length-weighted sets of arcs, not traversal order. Continuity is the documented break/pair measure, not percent of observations matched correctly.','',
    '## Actual metrics','']
for key in ['point_accuracy','point_macro_f1','route_precision','route_recall','route_f1','route_continuity','route_pieces','candidate_miss_rate']:
    lines.append(f'- {key}: classifier **{c[key]:.4f}**; HMM **{h[key]:.4f}**.')
lines += ['',f'Observation-weighted Top-1: classifier {weighted["classifier_only"]:.4%}; HMM {weighted["hmm_viterbi"]:.4%}. Do not interchange these with the equal-trip means above.',
    '', 'HMM improves point agreement on nine trips; trip 00000007 declines. Missing candidates and disconnected route pieces remain. The larger model was chosen by grouped training CV Top-1; it improved held-out point agreement relative to the old model, while mean route F1 stayed essentially unchanged.',
    '', '## Frozen settings and verification','',
    'Decoder settings were retained, not separately tuned: sigma=10m, beta=5m, classifier weight=0.5, gap threshold=10s, outlier threshold=120m, backward tolerance=15m, unreachable log penalty=-40, search base=400m and cap=15000m.',
    '', 'The restarted default API passed ten fresh predictions; every recorded classifier/HMM metric agrees with the offline final evaluation within API rounding. Release hashes agree. CORS/assets/error/cache checks passed. Physical two-laptop reachability remains a separate check.',
    '',f'Mean joint offline candidate/scoring/lattice/decoder pipeline time on this run: {h["pipeline_s"]:.3f}s per trip. This is shared by both mode rows, includes HMM computation, and is not standalone classifier latency. Actual API component times are in `results/api_verification_final.json`; they exclude network/UI time.',
    '', '## Evidence and reproduction','',
    '- `artifacts/final_release.json`: exact release identity, provenance and packages.',
    '- `results/decoder_comparison_student_test.csv`: per-trip final results.',
    '- `results/decoder_summary_student_test.json`: aggregate final results.',
    '- `results/api_verification_final.json`: restarted main API verification.',
    '- `results/final_results.png`: report-ready plot.',
    '- `analysis/maps/student_test_*.html`: regenerated final Folium development maps.',
    '', 'Run `python scripts/evaluate_decoder.py`, then `python scripts/verify_demo.py --expected-model-dir artifacts/models/final_monotone_v1 --release-manifest artifacts/final_release.json --output results/api_verification_final.json`, then `python scripts/final_results_report.py`. Serve API and UI before the HTTP test. Default serving: `python -m pathsnap.api.app --host 0.0.0.0 --port 8000`. Do not rerun `freeze_release.py` over an existing release.',
    '', '## Historical experiments','',
    'Keep the five-model sampled nearest-label comparison and the 12-trial tuning comparison as earlier experiments with their own methods. Keep the controlled improvement report as model-selection evidence. Earlier screenshots/results from the old model are historical and must not be described as final-release screenshots. This document is a report source, not the required consolidated report PDF.']
Path('docs/report/FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('Final API/offline/release parity passed; report references and chart generated.')
