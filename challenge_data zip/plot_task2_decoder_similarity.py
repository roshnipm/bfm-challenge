"""
Visualizes decoder similarity results across different conditions and tasks.

This script creates a bar plot showing AUROC scores for three conditions:
1. SS-SM (Same Subject, Same Movie)
2. DS-SM (Different Subject, Same Movie)
3. DS-DM (Different Subject, Different Movie)
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# load results as stored from challenge_eval_task2.py
with open("task2_decoder_similarity_results.json", "r") as f:
    results = json.load(f)

# extract data from results
tasks = results["tasks"]
groups = results["groups"]
means = np.array(results["means"])
sems = np.array(results["sems"])
overall_means = np.array(results["overall_means"])
overall_sems = np.array(results["overall_sems"])

bar_colors = ["#355C8D", "#2CA6A4", "#6EC664"]
# x-axis for groups, not in legend
x = np.arange(len(groups))

fig, axes = plt.subplots(1, len(tasks)+1, figsize=(16, 4), sharey=True)

# plot overall panel
axes[0].bar(x, overall_means, yerr=overall_sems, color=bar_colors, width=1, capsize=5)
axes[0].set_title("Overall", fontsize=12, fontname='Arial')
axes[0].set_ylabel("AUROC", fontsize=12, fontname='Arial')
axes[0].set_xticks(x)
axes[0].set_xticklabels(groups, fontsize=12, fontname='Arial')
axes[0].set_ylim(0.45, 0.9)
axes[0].axhline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.5)  # chance level

# plot individual task panels
for idx, task in enumerate(tasks):
    ax = axes[idx+1]
    ax.bar(x, means[idx], yerr=sems[idx], color=bar_colors, width=1, capsize=5)
    ax.set_title(task.capitalize(), fontsize=12, fontname='Arial')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12, fontname='Arial')
    ax.set_ylim(0.45, 0.9)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.5)  # chance level

# save plots
plt.tight_layout()
plt.savefig('task2_decoder_similarity_alltasks.pdf')
plt.show()