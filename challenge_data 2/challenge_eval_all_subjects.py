import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to path

"""
Evaluates model performance on population-level data one second after word onset.
Processes all available subject/trial pairs and accumulates results in a single file per evaluation type.

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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import numpy as np
import argparse, json, os, time
from sklearn.preprocessing import label_binarize # added for speaker classification

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

bins_start_before_word_onset_seconds = 0.5
bins_end_after_word_onset_seconds = 2.5

# Putting all results in a single file, description at the top
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

# Find all subject/trial pairs
subject_trial_pairs = []
for filename in os.listdir(model_features_dir):
    parts = filename.replace("frozen_population_btbank", "").replace(".npy", "").split("_")
    subject = parts[0]
    trial = parts[1]
    subject_trial_pairs.append((subject, trial))

if verbose:
    print(f"Found {len(subject_trial_pairs)} subject/trial pairs to process")

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
        features_file_path = os.path.join(model_features_dir, f"frozen_population_btbank{subject_id}_{trial_id}_{eval_name}.npy")
        
        if not os.path.exists(features_file_path):
            if verbose:
                print(f"Features file not found: {features_file_path}, skipping...")
            continue

        if verbose:
            print(f"\nProcessing subject {subject_id} trial {trial_id} for {eval_name}...")

        # Load pre-computed features
        features_dict = np.load(features_file_path, allow_pickle=True).item()
        X_all_bins = features_dict['X']  # shape: (n_samples, n_timebins, 1, d_model)
        y = features_dict['y']

        results_population = {
            "one_second_after_onset": {
                "time_bin_start": 0.0,
                "time_bin_end": 1.0,
                "folds": []
            }
        }

        # Create KFold cross-validator
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Loop over all folds
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_all_bins)):
            X_train_all_bins = X_all_bins[train_idx]
            X_test_all_bins = X_all_bins[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Get one second after onset (bins 4-12, given that each bin is 0.125 seconds and they start at -0.5 seconds)
            if timebins_aggregation == "concat":
                X_train = X_train_all_bins[:, 4:12, 0, :].reshape(len(train_idx), -1)
                X_test = X_test_all_bins[:, 4:12, 0, :].reshape(len(test_idx), -1)
            elif timebins_aggregation == "mean":
                X_train = X_train_all_bins[:, 4:12, 0, :].mean(axis=1)
                X_test = X_test_all_bins[:, 4:12, 0, :].mean(axis=1)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train logistic regression
            if eval_name == 'speaker':
                # Multi-class classification for speaker
                clf = LogisticRegression(random_state=42)
                clf.fit(X_train_scaled, y_train)
                y_test_pred = clf.predict(X_test_scaled)
                y_test_prob = clf.predict_proba(X_test_scaled)
                # Binarize the labels
                y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
                test_roc = roc_auc_score(y_test_bin, y_test_prob)
                # was checking if auc is sensitive to shuffling, which it was (came down to 0.5)
                # print(f"shuffled auc: {roc_auc_score(label_binarize(y_test, classes=np.unique(y_train)), clf.predict_proba(X_test_scaled[np.random.permutation(len(X_test_scaled))]))}")
                
            else:
                # Binary classification for onset, speech, volume
                # Train logistic regression
                clf = LogisticRegression(random_state=42)
                clf.fit(X_train_scaled, y_train)

                # Get predictions
                y_test_pred = clf.predict(X_test_scaled)
                y_test_prob = clf.predict_proba(X_test_scaled)[:, 1]
                test_roc = roc_auc_score(y_test, y_test_prob)

            # Calculate metrics
            test_accuracy = (y_test_pred == y_test).mean()

            fold_result = {
                "test_accuracy": float(test_accuracy),
                "test_roc_auc": float(test_roc)
            }

            results_population["one_second_after_onset"]["folds"].append(fold_result)
            
            if verbose:
                print(f"Subject {subject_id} trial {trial_id}, Fold {fold_idx+1}, Test accuracy: {test_accuracy:.3f}, Test ROC AUC: {test_roc:.3f}")

        # Add this subject/trial's results to the overall results
        all_results[eval_name]["evaluation_results"][f"btbank{subject_id}_{trial_id}"] = {
            "population": results_population
        }

        # Save the accumulated results after each subject/trial
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        with open(save_file_path, "w") as f:
            json.dump(all_results[eval_name], f, indent=4)
        if verbose:
            print(f"Updated results saved to {save_file_path}\n\n") 