import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open("task2_decoder_similarity_results.json", "r") as f:
    results = json.load(f)

TASKS = results["tasks"]
GROUPS = results["groups"]
means = np.array(results["means"])  # shape: (num_tasks, 3)
sems = np.array(results["sems"])
overall_means = np.array(results["overall_means"])
overall_sems = np.array(results["overall_sems"])

BAR_COLORS = ["#355C8D", "#2CA6A4", "#6EC664"]
x = np.arange(len(GROUPS))

fig, axes = plt.subplots(1, len(TASKS)+1, figsize=(16, 4), sharey=True)

# Overall panel
axes[0].bar(x, overall_means, yerr=overall_sems, color=BAR_COLORS, width=1, capsize=5)
axes[0].set_title("Overall", fontsize=12, fontname='Arial')
axes[0].set_ylabel("AUROC", fontsize=12, fontname='Arial')
axes[0].set_xticks(x)
axes[0].set_xticklabels(GROUPS, fontsize=12, fontname='Arial')
axes[0].set_ylim(0.45, 0.9)
axes[0].axhline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.5)

# Individual tasks
for idx, task in enumerate(TASKS):
    ax = axes[idx+1]
    ax.bar(x, means[idx], yerr=sems[idx], color=BAR_COLORS, width=1, capsize=5)
    ax.set_title(task.capitalize(), fontsize=12, fontname='Arial')
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, fontsize=12, fontname='Arial')
    ax.set_ylim(0.45, 0.9)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('task2_decoder_similarity_alltasks.pdf')
plt.show()