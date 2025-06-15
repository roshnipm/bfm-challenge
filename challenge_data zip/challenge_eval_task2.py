"""
Evaluates model generalization across different subject/trial combinations and movie conditions.

1. SS-SM (Same Subject, Same Movie): Cross-validation within a subject/trial
2. DS-SM (Different Subject, Same Movie): Train on one subject, test on another, same movie
3. DS-DM (Different Subject, Different Movie): Train on one subject/movie, test on another subject/movie

The script computes AUROC scores for each condition and task type (onset, speech, volume, speaker).
Results are saved to a JSON file and can be visualized using plot_task2_decoder_similarity.py.
"""

import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import KFold
from collections import defaultdict

# mapping from subject_trial to movie name
subject_trial_to_movie = {
    "1_2": "Thor: Ragnarok",
    "2_6": "Aquaman",
    "3_0": "Cars 2",
    "7_0": "Cars 2",
    "10_0": "Cars 2",
}

# configuration
features_dir = "analyses/eval_features/M_nst8_dm192_nh12_nl5_5_eaM_projN_eeL_fb1_cls_lr0.003_rCONT_EXP_CLS1_epoch100"
tasks = ["onset", "speech", "volume", "speaker"]
groups = ["SS-SM", "DS-SM", "DS-DM"]
bar_colors = ["#355C8D", "#2CA6A4", "#6EC664"]

means = []
sems = []

for task in tasks:
    # get all subject/trial pairs for the task
    subject_trial_pairs = []
    for filename in os.listdir(features_dir):
        if filename.endswith(f"_{task}.npy"):
            parts = filename.replace("frozen_population_btbank", "").replace(f"_{task}.npy", "").split("_")
            subject = parts[0]
            trial = parts[1]
            key = f"{subject}_{trial}"
            if key in subject_trial_to_movie:
                subject_trial_pairs.append((subject, trial))

    # load all data
    data = {}
    for subject, trial in subject_trial_pairs:
        key = f"{subject}_{trial}"
        filepath = os.path.join(features_dir, f"frozen_population_btbank{subject}_{trial}_{task}.npy")
        d = np.load(filepath, allow_pickle=True).item()
        X = d['X'][:, 4:12, 0, :].mean(axis=1)
        y = d['y']
        # storing X and y for each subject/trial pair
        # easier to iterate through later
        data[key] = (X, y)

    # SS-SM
    sssm_per_subject_means = []
    for subject, trial in subject_trial_pairs:
        key = f"{subject}_{trial}"
        X, y = data[key]
        
        # same splits due to data imbalance/errors
        if task == "speaker":
            kf = KFold(n_splits=4, shuffle=False)
        else:
            kf = KFold(n_splits=5, shuffle=False)

        fold_aurocs = []
        
        for train_idx, test_idx in kf.split(X):
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            
            # Train logistic regression
            clf = LogisticRegression(random_state=42)
            clf.fit(X_train, y[train_idx])
            
            # Get predictions and calculate AUROC
            # again split up for speaker due to multiclass classification
            if task == "speaker":
                classes = np.unique(y)
                y_test_bin = label_binarize(y[test_idx], classes=classes)
                y_prob = clf.predict_proba(X_test)
                auroc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
                fold_aurocs.append(auroc)
            else:
                y_prob = clf.predict_proba(X_test)[:, 1]
                auroc = roc_auc_score(y[test_idx], y_prob)
                fold_aurocs.append(auroc)
        
        # do means across folds first, as before
        sssm_per_subject_means.append(np.mean(fold_aurocs))

    # do means/SEMs across subject/trials
    sssm_mean = np.mean(sssm_per_subject_means)
    sssm_sem = np.std(sssm_per_subject_means) / np.sqrt(len(sssm_per_subject_means))

    # DS-SM
    dssm_aurocs = []
    # mapping from movie to subject/trial pairs
    movie_to_keys = defaultdict(list)
    # going through each subject/trial pair
    for subject, trial in subject_trial_pairs:
        # construct key for mapping use
        key = f"{subject}_{trial}"
        # add to map
        movie_to_keys[subject_trial_to_movie[key]].append(key)
    
    # for each movie, compare different subjects watching that same movie
    for movie, keys in movie_to_keys.items():
        # keys here contains *ONLY* subject/trial pairs from the *SAME* movie
        for i in range(len(keys)):
            for j in range(len(keys)):
                # check if different subject/trial pair and same movie (see outer loop)
                if i != j and keys[i].split('_')[0] != keys[j].split('_')[0]:
                    # train on one subject/trial pair, test on another
                    X_train, y_train = data[keys[i]]
                    X_test, y_test = data[keys[j]]
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    clf = LogisticRegression(random_state=42)
                    clf.fit(X_train_scaled, y_train)
                    if task == "speaker":
                        classes = np.unique(y_train)
                        y_test_bin = label_binarize(y_test, classes=classes)
                        y_prob = clf.predict_proba(X_test_scaled)
                        auroc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
                    else:
                        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
                        auroc = roc_auc_score(y_test, y_prob)
                    dssm_aurocs.append(auroc)
    dssm_mean = np.mean(dssm_aurocs)
    dssm_sem = np.std(dssm_aurocs) / np.sqrt(len(dssm_aurocs))

    # DS-DM: Train on one subject+movie, test on another subject+movie, different movies
    dsdm_aurocs = []
    for i in range(len(subject_trial_pairs)):
        for j in range(len(subject_trial_pairs)):
            # check if different subject/trial combo
            if i != j:
                # figuring out mappings
                subj_i, trial_i = subject_trial_pairs[i]
                subj_j, trial_j = subject_trial_pairs[j]
                key_i = f"{subj_i}_{trial_i}"
                key_j = f"{subj_j}_{trial_j}"
                movie_i = subject_trial_to_movie[key_i]
                movie_j = subject_trial_to_movie[key_j]

                # check if different subject/trial combo and different movie
                if subj_i != subj_j and movie_i != movie_j:
                    # same code as before, but train on one subject/trial pair, test on another
                    X_train, y_train = data[key_i]
                    X_test, y_test = data[key_j]
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    clf = LogisticRegression(random_state=42)
                    clf.fit(X_train_scaled, y_train)
                    # again, speaker different due to multiclass classification
                    if task == "speaker":
                        classes = np.unique(y_train)
                        y_test_bin = label_binarize(y_test, classes=classes)
                        y_prob = clf.predict_proba(X_test_scaled)
                        auroc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
                    else:
                        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
                        auroc = roc_auc_score(y_test, y_prob)
                    dsdm_aurocs.append(auroc)

    dsdm_mean = np.mean(dsdm_aurocs)
    dsdm_sem = np.std(dsdm_aurocs) / np.sqrt(len(dsdm_aurocs))

    # aggregate results
    means.append([sssm_mean, dssm_mean, dsdm_mean])
    sems.append([sssm_sem, dssm_sem, dsdm_sem])

# turn to array for overall analysis
means = np.array(means)
sems = np.array(sems)

# compute overall means/SEMs
overall_means = np.nanmean(means, axis=0)
overall_sems = np.sqrt(np.nansum(sems**2, axis=0)) / 4

# store results in a JSON file
results = {
    "tasks": tasks,
    "groups": groups,
    "means": means.tolist(),
    "sems": sems.tolist(),
    "overall_means": overall_means.tolist(),
    "overall_sems": overall_sems.tolist()
}

with open("task2_decoder_similarity_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to task2_decoder_similarity_results.json")
