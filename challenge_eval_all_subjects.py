"""
Evaluates model performance on population-level data one second after word onset.
Processes all available subject/trial pairs and accumulates results in a single file per evaluation type per model.

Parameters
----------
eval_names : str
    Comma-separated list of evaluation names (default: 'onset')
verbose : bool
    If True, prints progress information (default: False)
save_dir : str
    Directory path to save evaluation results (default: 'analyses/eval_results')
features_dir : str
    Directory containing pre-computed features (default: 'analyses/eval_features')
model_dir : str
    Directory containing model files (required)
model_epoch : int
    Model epoch to evaluate (default: 100)
timebins_aggregation : str
    Method to aggregate time bins - either 'mean' or 'concat' (default: 'mean')
overwrite : bool
    If True, overwrites existing results (default: False)
n_splits : int
    Number of cross-validation splits (default: 5)

Returns
-------
None
    Saves evaluation results as JSON files in the specified save_dir
"""

import argparse, json, os, time
import numpy as np

from decoding import auroc, discover_subject_trials, feature_path, fit_decoder, load_features, make_kfold, population_features

parser = argparse.ArgumentParser()
parser.add_argument('--eval_names', type=str, default='onset', help='Evaluation names list, separated by commas')
parser.add_argument('--verbose', action='store_true', help='Whether to print progress')
parser.add_argument('--save_dir', type=str, default='analyses/eval_results', help='Directory to save results')
parser.add_argument('--features_dir', type=str, default='analyses/eval_features', help='Directory containing pre-computed features')
parser.add_argument('--model_dir', type=str, required=True, help='Directory containing model files')
parser.add_argument('--model_epoch', type=int, default=100, help='Model epoch to evaluate')
parser.add_argument('--timebins_aggregation', type=str, default='mean', help='Timebins aggregation method (mean, concat)')
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing results')
parser.add_argument('--n_splits', type=int, default=5, help='Number of cross-validation splits')
args = parser.parse_args()

eval_names = args.eval_names.split(',')
verbose = bool(args.verbose)
save_dir = args.save_dir
features_dir = args.features_dir
model_dir = args.model_dir
model_epoch = args.model_epoch
timebins_aggregation = args.timebins_aggregation
overwrite = bool(args.overwrite)
n_splits = args.n_splits

# Moving here for addition to each file
all_results = {}
for eval_name in eval_names:
    all_results[eval_name] = {
        "model_name": f"Andrii's model ({model_dir}_epoch{model_epoch})",
        "author": "Andrii Zahorodnii",
        "description": "Linear regression on top of the output of the transformer model.",
        "organization": "MIT",
        "organization_url": "https://azaho.org/",
        "timestamp": time.time(),
        "evaluation_results": {}
    }

# Get all available feature files for this model and epoch
model_features_dir = os.path.join(features_dir, model_dir+f"_epoch{model_epoch}")

# Find all subject/trial pairs --> to iterate through for each model
subject_trial_pairs = discover_subject_trials(model_features_dir)

# now going through each eval_name
for eval_name in eval_names:
    # Create epoch-specific directory
    epoch_dir = os.path.join(save_dir, f"epoch{model_epoch}")
    save_file_path = os.path.join(epoch_dir, f"population_{eval_name}.json")
    
    if not overwrite and os.path.exists(save_file_path):
        if verbose:
            print(f"Results already exist at {save_file_path}, skipping...")
        continue

    # Going through every subject/trial pair
    for subject_id, trial_id in subject_trial_pairs:
        features_file_path = feature_path(model_features_dir, subject_id, trial_id, eval_name)

        if verbose:
            print(f"\nSubject: {subject_id}, Trial: {trial_id}, Eval: {eval_name}...")

        # Load pre-computed features
        X_all_bins, y = load_features(features_file_path)  # shape: (n_samples, n_timebins, 1, d_model)

        results_population = {
            "one_second_after_onset": {
                "time_bin_start": 0.0,
                "time_bin_end": 1.0,
                "folds": []
            }
        }

        # Create KFold cross-validator
        # changed to 4 for speaker due to issues/error with class imbalance
        kf = make_kfold(eval_name, n_splits) # -1 b/c speaker has 4 classes

        # Loop over all folds
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_all_bins)):
            X_train_all_bins = X_all_bins[train_idx]
            X_test_all_bins = X_all_bins[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Get one second after onset (bins 4-12, given that each bin is 0.125 seconds and they start at -0.5 seconds)
            X_train = population_features(X_train_all_bins, timebins_aggregation)
            X_test = population_features(X_test_all_bins, timebins_aggregation)

            # Scale features
            # Train logistic regression
            clf, X_test_scaled = fit_decoder(X_train, y_train, X_test)

            # Multi-class classification for speaker
            if eval_name == 'speaker':
                y_test_pred = clf.predict(X_test_scaled)
                # Binarize labels for multi-class classification
                test_roc = auroc(clf, X_test_scaled, y_test, eval_name, classes=np.unique(y_train))
            # same code as before    
            else:
                # Binary classification for onset, speech, volume
                # Train logistic regression
                y_test_pred = clf.predict(X_test_scaled)
                test_roc = auroc(clf, X_test_scaled, y_test, eval_name)

            # Calculate metrics
            test_accuracy = (y_test_pred == y_test).mean()

            fold_result = {
                "test_accuracy": float(test_accuracy),
                "test_roc_auc": float(test_roc),
                # added weights for decoder similarity analysis in task 2
                "weights": clf.coef_.flatten().tolist()
            }

            results_population["one_second_after_onset"]["folds"].append(fold_result)
            
            if verbose:
                print(f"Subject {subject_id}, Trial {trial_id}, Fold {fold_idx+1}, Test accuracy: {test_accuracy:.3f}, Test ROC AUC: {test_roc:.3f}")

        # Add this subject/trial's results to overall results
        all_results[eval_name]["evaluation_results"][f"btbank{subject_id}_{trial_id}"] = {
            "population": results_population
        }
        # Save accumulated results after each subject/trial
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        with open(save_file_path, "w") as f:
            json.dump(all_results[eval_name], f, indent=4)
        if verbose:
            print(f"Saved to {save_file_path}\n\n")
