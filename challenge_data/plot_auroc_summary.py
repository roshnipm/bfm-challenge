import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Settings
BAR_LABELS = [
    "Linear (raw voltage)",
    "Linear (spectrogram)",
    "Our Model (epoch 0)",
    "Our Model (epoch 40)",
    "Our Model (epoch 100)"
]
BAR_COLORS = [
    "#355C8D",   # dark blue
    "#2CA6A4",   # teal
    "#6EC664",   # light green
    "#3CB371",   # medium green
    "#228B22"    # dark green
]
EVAL_TYPES = ["onset", "speech", "volume", "speaker"]
EVAL_LABELS = ["Sentence Onset", "Speech", "Audio Volume", "Speaker Identity"]
RESULTS_DIR = "analyses/eval_results"

# File patterns for each bar
FILE_PATTERNS = [
    os.path.join(RESULTS_DIR, "linear_voltage", "population_{eval_type}.json"),
    os.path.join(RESULTS_DIR, "linear_spectrogram", "population_{eval_type}.json"),
    os.path.join(RESULTS_DIR, "epoch0", "population_{eval_type}.json"),
    os.path.join(RESULTS_DIR, "epoch40", "population_{eval_type}.json"),
    os.path.join(RESULTS_DIR, "epoch100", "population_{eval_type}.json")
]

# Collect AUROC values
# For each eval_type, for each bar, store a list of per-subject/trial mean AUROCs
auroc_data = {eval_type: [[] for _ in range(len(BAR_LABELS))] for eval_type in EVAL_TYPES}

for eval_idx, eval_type in enumerate(EVAL_TYPES):
    for bar_idx, file_pattern in enumerate(FILE_PATTERNS):
        json_path = file_pattern.format(eval_type=eval_type)
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found.")
            continue
        with open(json_path, "r") as f:
            results = json.load(f)
        # For each subject/trial, average across folds
        for subj_trial, res in results["evaluation_results"].items():
            folds = res["population"]["one_second_after_onset"]["folds"]
            fold_aurocs = [fold["test_roc_auc"] for fold in folds if fold["test_roc_auc"] is not None and not np.isnan(fold["test_roc_auc"])]
            if len(fold_aurocs) > 0:
                subj_trial_mean = np.mean(fold_aurocs)
                auroc_data[eval_type][bar_idx].append(subj_trial_mean)

# Compute means and SEMs across subject/trials
means = []
sems = []
for eval_type in EVAL_TYPES:
    m = []
    s = []
    for bar_idx in range(len(BAR_LABELS)):
        vals = auroc_data[eval_type][bar_idx]
        m.append(np.mean(vals))
        s.append(np.std(vals) / np.sqrt(len(vals)))

    means.append(m)
    sems.append(s)
means = np.array(means)  # shape: (eval_types, bars)
sems = np.array(sems)

# Overall (mean of the four evaluation type means for each bar)
overall_means = np.nanmean(means, axis=0)
overall_sems = np.sqrt(np.nansum(sems[:3, :]**2, axis=0)) / 3

# Plot
fig, axes = plt.subplots(1, len(EVAL_TYPES)+1, figsize=(14, 4), sharey=True)
bar_width = 1
x = np.arange(len(BAR_LABELS))

# Overall panel
bar_handles = []
for i in range(len(BAR_LABELS)):
    bar = axes[0].bar(i, overall_means[i], yerr=overall_sems[i], color=BAR_COLORS[i], width=bar_width, capsize=5)
    bar_handles.append(bar[0])
axes[0].set_title("Overall")
axes[0].set_ylabel("AUROC")
axes[0].set_ylim(0.45, 0.9)
axes[0].set_xticks([])
axes[0].axhline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.5)

# Individual eval types
for idx, (eval_label, mean, sem) in enumerate(zip(EVAL_LABELS, means, sems)):
    ax = axes[idx+1]
    for i in range(len(BAR_LABELS)):
        ax.bar(i, mean[i], yerr=sem[i], color=BAR_COLORS[i], width=bar_width, capsize=5)
    ax.set_title(eval_label)
    ax.set_xticks([])
    ax.set_ylim(0.45, 0.9)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.5)

plt.tight_layout(rect=[0, 0.07, 1, 1])  # leave space at bottom for legend
fig.legend(bar_handles, BAR_LABELS, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(BAR_LABELS), frameon=False)

# Save to PDF
fig.savefig("auroc_barplot.pdf", bbox_inches="tight", dpi=300)

plt.show()