"""Shared pieces used by the challenge evaluation scripts.

Feature files are named frozen_population_btbank{subject}_{trial}_{task}.npy
and store a dict with X of shape (samples, time bins, 1, d_model) and y.
"""

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, label_binarize

# mapping from subject_trial to movie name
MOVIE_BY_SUBJECT_TRIAL = {
    "1_2": "Thor: Ragnarok",
    "2_6": "Aquaman",
    "3_0": "Cars 2",
    "7_0": "Cars 2",
    "10_0": "Cars 2",
}

TASKS = ["onset", "speech", "volume", "speaker"]


def discover_subject_trials(features_dir, task=None):
    # get all subject/trial pairs for the task
    prefix = "frozen_population_btbank"
    pairs = []
    seen = set()
    for filename in sorted(os.listdir(features_dir)):
        if not filename.startswith(prefix) or not filename.endswith(".npy"):
            continue
        parts = filename[len(prefix):-len(".npy")].split("_")
        if len(parts) < 3:
            continue
        subject, trial, file_task = parts[0], parts[1], parts[2]
        if task is not None and file_task != task:
            continue
        pair = (subject, trial)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def feature_path(features_dir, subject, trial, task):
    filename = f"frozen_population_btbank{subject}_{trial}_{task}.npy"
    return os.path.join(features_dir, filename)


def load_features(path):
    # Load pre-computed features
    data = np.load(path, allow_pickle=True).item()
    return data["X"], data["y"]  # X shape: (n_samples, n_timebins, 1, d_model)


def population_features(X_bins, aggregation="mean"):
    # Get one second after onset (bins 4-12, given that each bin is 0.125 seconds and they start at -0.5 seconds)
    window = X_bins[:, 4:12, 0, :]
    if aggregation == "concat":
        return window.reshape(len(X_bins), -1)
    elif aggregation == "mean":
        return window.mean(axis=1)
    raise ValueError(f"Unknown timebins_aggregation: {aggregation}")


def make_kfold(task, n_splits):
    # changed to 4 for speaker due to issues/error with class imbalance
    splits = n_splits - 1 if task == "speaker" else n_splits  # -1 b/c speaker has 4 classes
    return KFold(n_splits=splits, shuffle=False)


def fit_decoder(X_train, y_train, X_test):
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_scaled, y_train)
    return clf, X_test_scaled


def auroc(clf, X_test_scaled, y_test, task, classes=None, multi_class=None):
    # Multi-class classification for speaker
    if task == "speaker":
        if classes is None:
            classes = clf.classes_
        # Binarize labels for multi-class classification
        y_bin = label_binarize(y_test, classes=classes)
        y_prob = clf.predict_proba(X_test_scaled)
        if multi_class is None:
            return roc_auc_score(y_bin, y_prob)
        return roc_auc_score(y_bin, y_prob, multi_class=multi_class)
    # Binary classification for onset, speech, volume
    return roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1])


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    sem = np.std(values) / np.sqrt(len(values))
    return float(np.mean(values)), float(sem)


def overall_mean_sem(means, sems):
    # Overall (mean of the four evaluation type means for each bar)
    means = np.asarray(means, dtype=float)
    sems = np.asarray(sems, dtype=float)
    overall_means = np.nanmean(means, axis=0)
    overall_sems = np.sqrt(np.nansum(sems**2, axis=0)) / means.shape[0]
    return overall_means, overall_sems
