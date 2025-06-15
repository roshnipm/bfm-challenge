import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# Mapping from subject_trial to movie (fill in all from your table)
subject_trial_to_movie = {
    "1_2": "Thor: Ragnarok",
    "2_6": "Aquaman",
    "3_0": "Cars 2",
    "7_0": "Cars 2",
    "10_0": "Cars 2",
    # Add more as needed
}

# Settings
features_dir = "analyses/eval_features/M_nst8_dm192_nh12_nl5_5_eaM_projN_eeL_fb1_cls_lr0.003_rCONT_EXP_CLS1_epoch100"  # adjust as needed
TASKS = ["onset", "speech", "volume", "speaker"]
GROUPS = ["SS-SM", "DS-SM", "DS-DM"]
BAR_COLORS = ["#355C8D", "#2CA6A4", "#6EC664"]

means = []  # shape: (num_tasks, 3)
sems = []

for TASK in TASKS:
    # Gather all available subject_trial pairs for the task
    subject_trial_pairs = []
    for filename in os.listdir(features_dir):
        if filename.endswith(f"_{TASK}.npy"):
            parts = filename.replace("frozen_population_btbank", "").replace(f"_{TASK}.npy", "").split("_")
            subject = parts[0]
            trial = parts[1]
            key = f"{subject}_{trial}"
            if key in subject_trial_to_movie:
                subject_trial_pairs.append((subject, trial))

    # Load all data
    data = {}
    for subject, trial in subject_trial_pairs:
        key = f"{subject}_{trial}"
        fpath = os.path.join(features_dir, f"frozen_population_btbank{subject}_{trial}_{TASK}.npy")
        d = np.load(fpath, allow_pickle=True).item()
        X = d['X'][:, 4:12, 0, :].mean(axis=1)  # mean aggregation, 1s after onset
        y = d['y']
        data[key] = (X, y)

    # SS-SM: mean AUROC across folds for each subject/trial, then mean/SEM across subject/trials
    sssm_per_subject_means = []
    for subject, trial in subject_trial_pairs:
        key = f"{subject}_{trial}"
        X, y = data[key]
        
        # Use same number of splits as in challenge_eval_all_subjects
        if TASK == "speaker":
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
            if TASK == "speaker":
                classes = np.unique(y)
                y_test_bin = label_binarize(y[test_idx], classes=classes)
                y_prob = clf.predict_proba(X_test)
                auroc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
                fold_aurocs.append(auroc)
            else:
                y_prob = clf.predict_proba(X_test)[:, 1]
                auroc = roc_auc_score(y[test_idx], y_prob)
                fold_aurocs.append(auroc)
        
        sssm_per_subject_means.append(np.mean(fold_aurocs))

    sssm_mean = np.mean(sssm_per_subject_means) if len(sssm_per_subject_means) > 0 else np.nan
    sssm_sem = np.std(sssm_per_subject_means) / np.sqrt(len(sssm_per_subject_means)) if len(sssm_per_subject_means) > 0 else np.nan

    # DS-SM: Train on one subject, test on another, same movie
    dssm_aurocs = []
    movie_to_keys = defaultdict(list)
    for subject, trial in subject_trial_pairs:
        key = f"{subject}_{trial}"
        movie_to_keys[subject_trial_to_movie[key]].append(key)
    for movie, keys in movie_to_keys.items():
        if len(keys) < 2:
            continue
        for i in range(len(keys)):
            for j in range(len(keys)):
                if i != j and keys[i].split('_')[0] != keys[j].split('_')[0]:
                    X_train, y_train = data[keys[i]]
                    X_test, y_test = data[keys[j]]
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    clf = LogisticRegression(random_state=42)
                    clf.fit(X_train_scaled, y_train)
                    if TASK == "speaker":
                        classes = np.unique(y_train)
                        y_test_bin = label_binarize(y_test, classes=classes)
                        y_prob = clf.predict_proba(X_test_scaled)
                        auroc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
                    else:
                        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
                        auroc = roc_auc_score(y_test, y_prob)
                    dssm_aurocs.append(auroc)
    dssm_mean = np.mean(dssm_aurocs) if len(dssm_aurocs) > 0 else np.nan
    dssm_sem = np.std(dssm_aurocs) / np.sqrt(len(dssm_aurocs)) if len(dssm_aurocs) > 0 else np.nan

    # DS-DM: Train on one subject+movie, test on another subject+movie, different movies
    dsdm_aurocs = []
    for i in range(len(subject_trial_pairs)):
        for j in range(len(subject_trial_pairs)):
            if i != j:
                subj_i, trial_i = subject_trial_pairs[i]
                subj_j, trial_j = subject_trial_pairs[j]
                key_i = f"{subj_i}_{trial_i}"
                key_j = f"{subj_j}_{trial_j}"
                movie_i = subject_trial_to_movie[key_i]
                movie_j = subject_trial_to_movie[key_j]
                if subj_i != subj_j and movie_i != movie_j:
                    X_train, y_train = data[key_i]
                    X_test, y_test = data[key_j]
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    clf = LogisticRegression(random_state=42)
                    clf.fit(X_train_scaled, y_train)
                    if TASK == "speaker":
                        classes = np.unique(y_train)
                        y_test_bin = label_binarize(y_test, classes=classes)
                        y_prob = clf.predict_proba(X_test_scaled)
                        auroc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')
                    else:
                        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
                        auroc = roc_auc_score(y_test, y_prob)
                    dsdm_aurocs.append(auroc)
    dsdm_mean = np.mean(dsdm_aurocs) if len(dsdm_aurocs) > 0 else np.nan
    dsdm_sem = np.std(dsdm_aurocs) / np.sqrt(len(dsdm_aurocs)) if len(dsdm_aurocs) > 0 else np.nan

    means.append([sssm_mean, dssm_mean, dsdm_mean])
    sems.append([sssm_sem, dssm_sem, dsdm_sem])

means = np.array(means)  # shape: (num_tasks, 3)
sems = np.array(sems)

# Compute overall means/SEMs
overall_means = np.nanmean(means, axis=0)  # Include all 4 tasks
overall_sems = np.sqrt(np.nansum(sems**2, axis=0)) / 4  # Divide by 4 tasks

# Store results in a JSON file
results = {
    "tasks": TASKS,
    "groups": GROUPS,
    "means": means.tolist(),
    "sems": sems.tolist(),
    "overall_means": overall_means.tolist(),
    "overall_sems": overall_sems.tolist()
}

with open("task2_decoder_similarity_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to task2_decoder_similarity_results.json")
