"""
Evaluates model generalization across different subject/trial combinations and movie conditions.

1. SS-SM (Same Subject, Same Movie): Cross-validation within a subject/trial
2. DS-SM (Different Subject, Same Movie): Train on one subject, test on another, same movie
3. DS-DM (Different Subject, Different Movie): Train on one subject/movie, test on another subject/movie

The script computes AUROC scores for each condition and task type (onset, speech, volume, speaker).
Results are saved to a JSON file and can be visualized using plot_task2_decoder_similarity.py.
"""

import json
from collections import defaultdict

import numpy as np

from decoding import (
    MOVIE_BY_SUBJECT_TRIAL,
    TASKS,
    auroc,
    discover_subject_trials,
    feature_path,
    fit_decoder,
    load_features,
    make_kfold,
    mean_sem,
    overall_mean_sem,
    population_features,
)

# mapping from subject_trial to movie name
subject_trial_to_movie = MOVIE_BY_SUBJECT_TRIAL

# configuration
features_dir = "analyses/eval_features/M_nst8_dm192_nh12_nl5_5_eaM_projN_eeL_fb1_cls_lr0.003_rCONT_EXP_CLS1_epoch100"
tasks = TASKS
groups = ["SS-SM", "DS-SM", "DS-DM"]


def decoder_auroc(X_train, y_train, X_test, y_test, task, classes=None):
    # Scale features
    # Train logistic regression
    clf, X_test_scaled = fit_decoder(X_train, y_train, X_test)
    # again split up for speaker due to multiclass classification
    if task == "speaker":
        if classes is None:
            classes = np.unique(y_train)
        return auroc(clf, X_test_scaled, y_test, task, classes=classes, multi_class="ovr")
    return auroc(clf, X_test_scaled, y_test, task)


means = []
sems = []

for task in tasks:
    # get all subject/trial pairs for the task
    subject_trial_pairs = []
    for subject, trial in discover_subject_trials(features_dir, task=task):
        key = f"{subject}_{trial}"
        if key in subject_trial_to_movie:
            subject_trial_pairs.append((subject, trial))

    # load all data
    data = {}
    for subject, trial in subject_trial_pairs:
        key = f"{subject}_{trial}"
        filepath = feature_path(features_dir, subject, trial, task)
        X_bins, y = load_features(filepath)
        X = population_features(X_bins)
        # storing X and y for each subject/trial pair
        # easier to iterate through later
        data[key] = (X, y)

    # SS-SM
    sssm_per_subject_means = []
    for subject, trial in subject_trial_pairs:
        key = f"{subject}_{trial}"
        X, y = data[key]

        # same splits due to data imbalance/errors
        kf = make_kfold(task, 5)

        fold_aurocs = []

        for train_idx, test_idx in kf.split(X):
            # Get predictions and calculate AUROC
            # again split up for speaker due to multiclass classification
            classes = np.unique(y) if task == "speaker" else None
            fold_aurocs.append(decoder_auroc(X[train_idx], y[train_idx], X[test_idx], y[test_idx], task, classes=classes))

        # do means across folds first, as before
        sssm_per_subject_means.append(np.mean(fold_aurocs))

    # do means/SEMs across subject/trials
    sssm_mean, sssm_sem = mean_sem(sssm_per_subject_means)

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
                    dssm_aurocs.append(decoder_auroc(X_train, y_train, X_test, y_test, task))
    dssm_mean, dssm_sem = mean_sem(dssm_aurocs)

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
                    # again, speaker different due to multiclass classification
                    X_train, y_train = data[key_i]
                    X_test, y_test = data[key_j]
                    dsdm_aurocs.append(decoder_auroc(X_train, y_train, X_test, y_test, task))

    dsdm_mean, dsdm_sem = mean_sem(dsdm_aurocs)

    # aggregate results
    means.append([sssm_mean, dssm_mean, dsdm_mean])
    sems.append([sssm_sem, dssm_sem, dsdm_sem])

# turn to array for overall analysis
means = np.array(means)
sems = np.array(sems)

# compute overall means/SEMs
overall_means, overall_sems = overall_mean_sem(means, sems)

# store results in a JSON file
results = {
    "tasks": tasks,
    "groups": groups,
    "means": means.tolist(),
    "sems": sems.tolist(),
    "overall_means": overall_means.tolist(),
    "overall_sems": overall_sems.tolist()
}

with open("analyses/eval_results/task2_decoder_similarity_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to analyses/eval_results/task2_decoder_similarity_results.json")
