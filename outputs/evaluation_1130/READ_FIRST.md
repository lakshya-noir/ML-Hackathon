# Lakshya and Nikhil: read this before 11:30

This folder contains measured development results and readable PNG images. Open PNG files directly in Preview/Finder or your editor's image viewer. HTML opened in the editor shows source code; to use an interactive map, open its HTML file in Safari/Chrome using Finder → Open With.

## What is finished

We audited every one of the 90 supplied development trips, checked the road graphs, created EDA/model/tuning plots, inspected three trip maps and three close-ups, and calculated model precision/recall and training-versus-test diagnostics. No raw data was deleted. No model was retrained in this session. No GitHub pushes were made.

## Nikhil's data explanation

There are 80 training trips containing **208,417 GPS observations**, and 10 student-test trips containing **19,160 observations**: **227,577 GPS observations in total**. The training experiment sampled up to 200 observations per trip to save time; this did not remove any raw observations.

Every trip has four files: GPS coordinates/times (`track`), road endpoint coordinates (`nodes`), road connections (`arcs`), and an ordered list of reference roads (`route`). IDs are local to a trip. Road graph totals count records across trip-specific graphs, not unique roads across the whole dataset.

Audit findings from `trip_audit.csv` and `data_summary.json`:

- All 90 trips loaded with expected column counts; node/route IDs passed range checks. No nonfinite GPS/node values, invalid GPS coordinate ranges, duplicate timestamps or non-increasing timestamps were found.
- **7,550 timestamp gaps exceed one second**. Training trip `00000090` has a **153,064-second gap**, between observations 534 and 535 (timestamps 970 and 154034). Student-test trip `00000065` has a **210-second gap**, between observations 114 and 115. We must not assume continuous movement through these gaps.
- **106 zero-length arcs** exist: 100 in training, six in student testing. Their endpoints have the same position, so direction is not informative there. Current features remain numerically finite, but these arcs need an explicit decoder policy.
- **89 of 90 trip graphs have multiple weakly connected components**. This does not mean the vehicle's route crosses components; it means we must check reachability when linking candidates. No isolated nodes were found.
- **14 consecutive reference-arc pairs do not connect in the stated direction**: 13 in training and one in student testing. This is a reference-data/interpretation concern; do not silently claim every supplied route is a perfect directed path.
- Shortest/longest training trips: `00000026` / `00000054`. Shortest/longest student-test trips: `00000052` / `00000056`.

Show `01_data_overview.png` while explaining this. The largest-gap point is an outlier; it is a measured finding, not a plotting error.

## Lakshya's feature explanation

We score candidate roads rather than predict a universal road ID. A candidate is labelled 1 when it equals the approximate reference road for that observation, otherwise 0.

1. Distance to road: closer roads are plausible candidates.
2. Signed direction agreement: does vehicle motion follow the arc's direction?
3. Absolute direction agreement: does movement line up with the road, regardless of direction?
4. Road length: helps distinguish different segment shapes/sizes.
5. Position along road: indicates whether the GPS projection is near an endpoint or in the middle.
6. Local speed: adds movement context from nearby timestamps.
7. Previous-point distance to this road: tests whether the previous observation also fits.
8. Next-point distance to this road: tests whether the following observation fits.
9. Distance rank: identifies the nearest, second nearest and later candidates.

Coordinates are projected approximately into local metres before distance calculations. Because next observations are used, this is offline matching of a recorded trip, not a strictly live streaming predictor.

## Three map findings we can actually explain

Open `04_disagreement_closeups.png`. These are agent-reviewed images and measured arc comparisons; they do not establish the vehicle's independently verified position.

1. **Trip 52, point 4:** the classifier selects short arc 2407 near the GPS point; the approximate reference selects arc 1680, about 11.18m away. The reference is in the candidates, so this is a scoring disagreement. The close-up shows why closeness alone can favour a small nearby segment instead of the reference road. Sequential context may help; improvement must be measured after decoding.
2. **Trip 56, point 214:** reference arc 13975 is about 12.24m away but absent from our eight candidates. A direct all-road distance check finds seven strictly closer arcs plus two tied at the reference distance. The top-eight cutoff can omit one of the tied roads. This point is also flagged ambiguous by reference geometry. The classifier cannot select a road that is missing from its input.
3. **Trip 65, point 267:** reference arc 1426 is about 21.79m away, with nine strictly closer road arcs. The top-eight cutoff therefore excludes it. The image shows different nearby road segments. This is a candidate-generation limitation, not solely a classifier error. Local-projection distance varies slightly with the projection origin.

Decision for the first evaluation: retain the existing reproducible baseline and state these limitations. Do not edit known test-trip predictions manually or claim labels are physically verified. A future general candidate-budget/alignment change requires regenerated caches, training and comparison. Keep reference roads out of candidate generation and inference features.

## Models: what to say

Show `02_model_comparison.png`. All five models used the same features and trip-grouped validation. Student-test Top-1 agreement with approximate labels:

- Logistic Regression: 85.15%. A simple regularised linear probability model; minimises logistic loss with L2 regularisation. It provides a useful simple baseline but has limited nonlinear interactions.
- Gaussian Naive Bayes: 84.10%. Fits class probabilities with Gaussian feature likelihoods. Its conditional-independence assumption is questionable because distance-related features are correlated; measured conditional log loss is also worse than the other models.
- Decision Tree: 85.45%. Gini-based splits learn nonlinear rules; depth and minimum leaf size constrain complexity.
- Random Forest: 85.55%. Averages many Gini-based trees. It models interactions but is larger and slower to train than the selected boosting model in our initial run.
- Histogram Gradient Boosting: 85.85%. Fits boosted nonlinear trees with log loss. It narrowly leads the chosen grouped-CV objective and is the selected family.

The plot's left panel measures **candidate binary macro F1**, averaging the two candidate classes. The middle panel measures **observation Top-1 agreement and trip-namespaced road-arc macro F1**. Those are different tasks and should not be described as the same F1 score. The right panel's loss excludes points whose approximate reference road is missing from candidates; candidate miss rate must be reported alongside it.

`model_diagnostics.csv` adds candidate-binary macro precision/recall and train/test F1 for all five models plus the tuned model. Random Forest's training F1 is about 0.931 versus 0.912 on student testing: some optimism is visible, but this alone does not prove severe overfitting. Gaussian NB is similarly low on train and test (~0.787/~0.786), suggesting limited fit under this representation. Other models have small or reversed train/test gaps. Different trip difficulty and approximate labels limit strong conclusions. We have no fabricated epoch curves.

## Tuning explanation

Show `03_tuning_comparison.png`. We ran 12 Optuna TPE trials with three trip-grouped folds on training data. Search ranges: learning rate 0.02–0.3 (log scale), leaf nodes 7–31 (integer), L2 regularisation 0.0001–10 (log scale).

Best parameters: learning rate 0.136014, leaf nodes 27, L2 regularisation 1.486758. The chosen CV binary macro F1 rose from 0.90158 to 0.90252. Student-test Top-1 declined slightly from 85.85% to 85.70%; conditional candidate log loss improved from 0.39682 to 0.39275. Tuning gave mixed results, not an improvement in every metric. Neither evaluator data nor evaluator labels were used.

## Scope

This retained pack describes the initial sampled classifier experiment and raw-data audit. Final full-trip HMM/API results are recorded in `results/decoder_summary_student_test.json` and `docs/report/FINAL_RESULTS.md`. Initial model samples and complete raw-data counts refer to different populations.
