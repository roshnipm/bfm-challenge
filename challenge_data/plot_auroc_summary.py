import os
import json
import numpy as np
import matplotlib.pyplot as plt

# labels for all features of the plot
bar_labels = [
    "Linear (raw voltage)",
    "Linear (spectrogram)",
    "Our Model (epoch 0)",
    "Our Model (epoch 40)",
    "Our Model (epoch 100)"
]
bar_colors = [
    "#355C8D",   # dark blue
    "#2CA6A4",   # teal
    "#6EC664",   # light green
    "#3CB371",   # medium green
    "#228B22"    # dark green
]
eval_types = ["onset", "speech", "volume", "speaker"]
eval_labels = ["Sentence Onset", "Speech", "Audio Volume", "Speaker Identity"]
results_dir = "analyses/eval_results"

# Retrieve results for each model and eval
file_patterns = [
    os.path.join(results_dir, "linear_voltage", "population_{eval_type}.json"),
    os.path.join(results_dir, "linear_spectrogram", "population_{eval_type}.json"),
    os.path.join(results_dir, "epoch0", "population_{eval_type}.json"),
    os.path.join(results_dir, "epoch40", "population_{eval_type}.json"),
    os.path.join(results_dir, "epoch100", "population_{eval_type}.json")
]

# Collect AUROC values
# For each eval_type, for each bar, store a list of per-subject/trial mean AUROCs
auroc_data = {eval_type: [[] for _ in range(len(bar_labels))] for eval_type in eval_types}

# going through each evaluation type
for eval_idx, eval_type in enumerate(eval_types):
    # going through each model (bar in plot)
    for bar_idx, file_pattern in enumerate(file_patterns):
        # format the file path for the current eval_type
        json_path = file_pattern.format(eval_type=eval_type)
        # load results
        with open(json_path, "r") as f:
            results = json.load(f)
        # for each subject/trial pair, average across folds as listed in challenge description
        for subj_trial, res in results["evaluation_results"].items():
            # retrieving folds
            folds = res["population"]["one_second_after_onset"]["folds"]
            # retrieving aurocs
            fold_aurocs = [fold["test_roc_auc"] for fold in folds if fold["test_roc_auc"] is not None and not np.isnan(fold["test_roc_auc"])]
            subj_trial_mean = np.mean(fold_aurocs)
            auroc_data[eval_type][bar_idx].append(subj_trial_mean)

# Compute means + SEMs across subject/trials
means = []
sems = []
for eval_type in eval_types:
    m = []
    s = []
    # going through each bar/model
    for bar_idx in range(len(bar_labels)):
        # retrieving aurocs for each bar/model
        vals = auroc_data[eval_type][bar_idx]
        # appending mean and sem for each bar/model
        # taking mean after done across folds previously
        m.append(np.mean(vals))
        s.append(np.std(vals) / np.sqrt(len(vals)))

    means.append(m)
    sems.append(s)
    
# converting to numpy arrays for overall analysis
means = np.array(means)
sems = np.array(sems)

# Overall (mean of the four evaluation type means for each bar)
overall_means = np.nanmean(means, axis=0)
overall_sems = np.sqrt(np.nansum(sems**2, axis=0)) / 4

# Plot
fig, axes = plt.subplots(1, len(eval_types)+1, figsize=(14, 4), sharey=True) # +1 for overall graph
x = np.arange(len(bar_labels))

# Overall panel
bar_handles = []
for i in range(len(bar_labels)):
    bar = axes[0].bar(i, overall_means[i], yerr=overall_sems[i], color=bar_colors[i], width=1, capsize=5)
    bar_handles.append(bar[0])
axes[0].set_title("Overall")
axes[0].set_ylabel("AUROC")
axes[0].set_ylim(0.45, 0.9)
axes[0].set_xticks([])
axes[0].axhline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.5)

# Individual eval types
# zip to iterate through each eval_label, mean, and sem all at once
for idx, (eval_label, mean, sem) in enumerate(zip(eval_labels, means, sems)):
    # everything +1 b/c overall graph is first
    ax = axes[idx+1]
    for i in range(len(bar_labels)):
        ax.bar(i, mean[i], yerr=sem[i], color=bar_colors[i], width=1, capsize=5)
    ax.set_title(eval_label)
    ax.set_xticks([])
    ax.set_ylim(0.45, 0.9)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.5)

plt.tight_layout(rect=[0, 0.07, 1, 1])  # leave space at bottom for legend
fig.legend(bar_handles, bar_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(bar_labels), frameon=False)

# save to PDF
fig.savefig("auroc_barplot.pdf", bbox_inches="tight", dpi=300)

plt.show()