"""Check the public working-tree payload without staging, committing or pushing.

Only tracked and nonignored files are checked. Reports go to ignored output/.
This cannot prove originality, GitHub identity association or remote deployment.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    required = [
        'README.md', 'CONTRIBUTING.md', '.gitattributes', 'requirements.txt',
        'requirements-demo.txt', 'requirements-ml.txt',
        'pathsnap/data.py', 'pathsnap/features/candidates.py', 'pathsnap/features/reference.py',
        'pathsnap/models/predict.py', 'pathsnap/decoder/viterbi.py',
        'pathsnap/decoder/graph.py', 'pathsnap/decoder/evaluation.py',
        'pathsnap/api/app.py', 'pathsnap/release.py', 'ui/index.html',
        'ui/vendor/leaflet/LICENSE', 'artifacts/final_release.json',
        'artifacts/decoder_config.json', 'results/model_comparison.json',
        'results/tuning_trials.json', 'results/tuning.sqlite3',
        'results/confusion_matrices.csv', 'results/confusion_matrices.json',
        'results/confusion_matrices.png', 'results/decoder_summary_student_test.json',
        'results/report_supplement.json', 'results/data_inventory.csv',
        'docs/TWO_LAPTOP_DEMO.md', 'docs/SUBMISSION.md', 'docs/REPRODUCING.md',
        'docs/report/FINAL_RESULTS.md',
    ]
    required += ['scripts/'+name+'.py' for name in (
        'import_dataset', 'inventory', 'train_models', 'improve_model',
        'prepare_evaluation', 'evaluate_decoder', 'model_diagnostics',
        'make_folium_map', 'inspect_maps', 'report_figures', 'verify_demo',
        'report_measurements', 'test_decoder')]
    required += ['artifacts/models/'+name+'/model.joblib' for name in (
        'logistic', 'gaussian_nb', 'decision_tree', 'random_forest',
        'hist_gradient_boosting', 'tuned', 'final_monotone_v1')]
    files = sorted(set(subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        cwd=ROOT).decode().split('\0')) - {''})
    files = [name for name in files if (ROOT/name).is_file()]
    for name in required:
        if name not in files:
            errors.append('Missing or ignored required file: '+name)
    for name in files:
        path = ROOT/name
        if path.stat().st_size > 100*1024*1024:
            errors.append('Exceeds 100 MiB: '+name)
        if name.startswith(('data/', '.venv/', 'output/', 'node_modules/', 'logs/')):
            errors.append('Local-only path in public payload: '+name)
        if '__pycache__/' in name or path.name in ('.env', '.DS_Store'):
            errors.append('Cache or environment file in public payload: '+name)
    manifest = json.loads((ROOT/'artifacts/final_release.json').read_text())
    for name, key in ((manifest['model_dir']+'/model.joblib', 'model_sha256'),
                      (manifest['decoder_config'], 'decoder_config_sha256')):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != manifest[key]:
            errors.append('Frozen release hash mismatch: '+name)
    comparison = json.loads((ROOT/'results/model_comparison.json').read_text())
    if len(comparison['models']) != 5:
        errors.append('Expected five baseline models')
    for fold in comparison['folds']:
        if set(fold['train_trips']) & set(fold['validation_trips']):
            errors.append('Trip overlap in grouped validation')
    trials = json.loads((ROOT/'results/tuning_trials.json').read_text())
    if len(trials) != 12 or any(t['state'] != 'COMPLETE' for t in trials):
        errors.append('Expected twelve completed retained Bayesian trials')
    final = json.loads((ROOT/'results/decoder_summary_student_test.json').read_text())
    if final['release'] != {key: manifest[key] for key in final['release']}:
        errors.append('Final result release metadata differs from manifest')
    if len(final['trips']) != 10 or final['observations'] != 19160:
        errors.append('Unexpected final development-evaluation coverage')
    patterns = {
        'private key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'GitHub token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
        'AWS access key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
        'OpenAI-style secret': r'\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b',
    }
    text_ext = {'.py', '.js', '.cjs', '.json', '.md', '.txt', '.yaml', '.yml', '.toml', '.bat', '.html', '.css'}
    for name in files:
        if (ROOT/name).suffix.lower() not in text_ext:
            continue
        content = (ROOT/name).read_text(errors='replace')
        for kind, pattern in patterns.items():
            if re.search(pattern, content):
                errors.append('Potential '+kind+' in '+name)
        if name.endswith('.md') and (name == 'README.md' or name.startswith('docs/')):
            for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', content):
                if '://' in link or link.startswith('#'):
                    continue
                target = link.split('#')[0].strip('<>')
                resolved = ((ROOT/name).parent/target).resolve()
                try:
                    relative = resolved.relative_to(ROOT).as_posix()
                except ValueError:
                    errors.append('Nonportable documentation link in '+name)
                    continue
                if relative not in files:
                    errors.append('Missing or unpublished documentation target in '+name+': '+target)
    pending = [
        'Verify public evaluator access and submit the separate report.',
        'Verify the physical two-laptop demonstration independently of local HTTP checks.',
    ]
    report = dict(packaging_passed=not errors, files=len(files),
                  bytes=sum((ROOT/name).stat().st_size for name in files),
                  secret_scan='limited token-pattern scan; manual review still required',
                  frozen_release=manifest['release_id'], errors=errors, external_handoff=pending)
    out = ROOT/'output'
    out.mkdir(exist_ok=True)
    (out/'submission-readiness.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'push-files.txt').write_text('\n'.join(files)+'\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
