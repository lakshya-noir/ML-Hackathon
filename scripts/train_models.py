"""Five-model experiment with trip-grouped CV and explicit surrogate labels."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss
from pathsnap.data import load_trip, load_reference
from pathsnap.features.candidates import generate_candidates, FEATURE_NAMES
from pathsnap.features.reference import align_reference

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--max-observations', type=int, default=200)
parser.add_argument('--folds', type=int, default=3)
parser.add_argument('--trials', type=int, default=0, help='Bayesian trials for the CV-selected model')
args = parser.parse_args()
Path('results').mkdir(exist_ok=True)
Path('outputs').mkdir(exist_ok=True)
cache = Path(f'outputs/features-{args.max_observations}.joblib')
if cache.exists():
    partitions = joblib.load(cache)
else:
    partitions = {}
    for partition in ['train', 'student_test']:
        records = []
        for directory in sorted((Path('data')/partition).iterdir()):
            trip = load_trip(directory)
            labels, offsets, ambiguous = align_reference(trip, load_reference(directory))
            # Subsample for initial experiments while retaining original temporal neighbours.
            candidates = generate_candidates(trip)
            indices = np.unique(np.linspace(0, len(candidates)-1,
                min(args.max_observations, len(candidates))).astype(int))
            observations = [(candidates[i], int(labels[i])) for i in indices]
            records.append(dict(trip_id=trip.trip_id, observations=observations,
                mean_reference_offset_m=float(offsets.mean()), ambiguous_fraction=float(ambiguous.mean())))
            print(partition, trip.trip_id, len(trip.track), 'observations', flush=True)
        partitions[partition] = records
    joblib.dump(partitions, cache)


def flatten(records):
    x, y, groups = [], [], []
    for record in records:
        for row, label in record['observations']:
            for c in row:
                x.append(c['features']); y.append(int(c['arc_id']==label)); groups.append(record['trip_id'])
    return np.array(x), np.array(y), np.array(groups)


def evaluate(model, records):
    truth, predicted, losses, misses = [], [], [], 0
    x, _, _ = flatten(records)
    probabilities = np.clip(model.predict_proba(x)[:, 1], 1e-9, 1)
    offset = 0
    for record in records:
        for row, label in record['observations']:
            p = probabilities[offset:offset+len(row)]; offset += len(row)
            p = p/p.sum()
            ids = [c['arc_id'] for c in row]
            # Pair (trip, arc) preserves trip-local identity in macro F1.
            truth.append(record['trip_id']+':'+str(label))
            predicted.append(record['trip_id']+':'+str(ids[int(np.argmax(p))]))
            if label in ids:
                losses.append(-np.log(p[ids.index(label)]))
            else:
                misses += 1
    return dict(top1_accuracy=accuracy_score(truth, predicted),
        observation_arc_macro_f1=f1_score(truth, predicted, average='macro', zero_division=0),
        conditional_candidate_log_loss=float(np.mean(losses)) if losses else None,
        candidate_miss_rate=misses/len(truth), observations=len(truth))


models = {
    'logistic': make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, random_state=42)),
    'gaussian_nb': make_pipeline(StandardScaler(), GaussianNB()),
    'decision_tree': DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=42),
    'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, random_state=42),
}
x, y, groups = flatten(partitions['train'])
xt, yt, _ = flatten(partitions['student_test'])
splits = list(GroupKFold(n_splits=args.folds).split(x, y, groups))
results = dict(label_method='nearest reference-route geometry surrogate; not verified pointwise ground truth',
    seed=42, max_observations_per_trip=args.max_observations, feature_names=FEATURE_NAMES,
    folds=[dict(train_trips=sorted(set(groups[a])), validation_trips=sorted(set(groups[b]))) for a,b in splits],
    models={}, data_quality={p:[{k:v for k,v in r.items() if k!='observations'} for r in records]
                             for p,records in partitions.items()})
best, best_score = None, -1
for name, model in models.items():
    start = time.perf_counter()
    cv = []
    for a,b in splits:
        fitted = clone(model).fit(x[a],y[a])
        cv.append(f1_score(y[b], fitted.predict(x[b]), average='macro', zero_division=0))
    model.fit(x,y)
    train_seconds = time.perf_counter()-start
    start = time.perf_counter()
    metrics = evaluate(model,partitions['student_test'])
    inference_seconds = time.perf_counter()-start
    p = model.predict_proba(xt)[:,1]
    metrics.update(candidate_binary_macro_f1=f1_score(yt,p>=0.5,average='macro'),
        candidate_binary_log_loss=log_loss(yt,p), cv_candidate_binary_macro_f1=cv,
        mean_cv_candidate_binary_macro_f1=float(np.mean(cv)), training_including_cv_seconds=train_seconds,
        scoring_seconds=inference_seconds, configuration=repr(model))
    results['models'][name] = metrics
    target = Path('artifacts/models')/name
    target.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(model=model, feature_names=FEATURE_NAMES, label_method=results['label_method']),target/'model.joblib')
    if np.mean(cv)>best_score:
        best, best_score = name,float(np.mean(cv))
    results['selected_by_group_cv'] = best
    Path('results/model_comparison.json').write_text(json.dumps(results,indent=2))
    print(name, metrics, flush=True)
print('Selected by group CV:', best, flush=True)
if args.trials:
    import optuna
    def configure(trial):
        model = clone(models[best])
        if best == 'logistic':
            model.set_params(logisticregression__C=trial.suggest_float('C', 0.01, 100, log=True))
        elif best == 'gaussian_nb':
            model.set_params(gaussiannb__var_smoothing=trial.suggest_float('var_smoothing', 1e-12, 1e-3, log=True))
        elif best in ('decision_tree', 'random_forest'):
            model.set_params(max_depth=trial.suggest_int('max_depth', 5, 20),
                             min_samples_leaf=trial.suggest_int('min_samples_leaf', 2, 40))
        else:
            model.set_params(learning_rate=trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
                max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 7, 31),
                l2_regularization=trial.suggest_float('l2_regularization', 1e-4, 10, log=True))
        return model
    def objective(trial):
        model = configure(trial)
        scores = []
        for a,b in splits:
            fitted = clone(model).fit(x[a], y[a])
            scores.append(f1_score(y[b], fitted.predict(x[b]), average='macro', zero_division=0))
        trial.set_user_attr('fold_scores', scores)
        return float(np.mean(scores))
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42),
        storage='sqlite:///results/tuning.sqlite3',
        study_name=f'candidate-{best}-cap{args.max_observations}-folds{args.folds}', load_if_exists=True)
    study.optimize(objective, n_trials=args.trials)
    tuned = configure(study.best_trial)
    start = time.perf_counter()
    tuned.fit(x,y)
    fit_seconds = time.perf_counter()-start
    start = time.perf_counter()
    metrics = evaluate(tuned, partitions['student_test'])
    metrics.update(scoring_seconds=time.perf_counter()-start, fit_seconds=fit_seconds,
                   best_params=study.best_params, mean_cv_candidate_binary_macro_f1=study.best_value)
    results['tuning'] = metrics
    target = Path('artifacts/models')/'tuned'
    target.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(model=tuned, feature_names=FEATURE_NAMES, label_method=results['label_method']),target/'model.joblib')
    Path('results/tuning_trials.json').write_text(json.dumps([
        dict(number=t.number, params=t.params, value=t.value, state=t.state.name, fold_scores=t.user_attrs)
        for t in study.trials], indent=2))
    Path('results/model_comparison.json').write_text(json.dumps(results,indent=2))
    print('Tuned:', metrics, flush=True)
