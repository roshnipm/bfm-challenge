import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.backends.backend_pdf import PdfPages


# ---- User settings ----
RESULTS_DIR = "analyses/eval_results/epoch100"  # Change if needed
EVAL_TYPES = ["onset", "speech", "volume", "speaker"]
EVAL_LABELS = ["Sentence Onset", "Speech", "Audio Volume", "Speaker Identity"]
MOVIE_MAP = {
    "btbank1_2": "Thor: Ragnarok",
    "btbank2_6": "Aquaman",
    "btbank3_0": "Cars 2",
    "btbank7_0": "Cars 2",
    "btbank10_0": "Cars 2",
}
BAR_COLORS = ["#355C8D", "#2CA6A4"]
# -----------------------

def load_weights(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
    weights_dict = {}
    for subj_trial, res in results["evaluation_results"].items():
        folds = res["population"]["one_second_after_onset"]["folds"]
        # Take mean weights across folds for each subject/trial
        weights = np.array([fold["weights"] for fold in folds])
        mean_weights = np.mean(weights, axis=0)
        weights_dict[subj_trial] = mean_weights
    return weights_dict

def compute_pairwise_metrics(weights_dict, movie_map):
    keys = list(weights_dict.keys())
    pearson = {"DS-SM": [], "DS-DM": []}
    cosine = {"DS-SM": [], "DS-DM": []}
    for i, k1 in enumerate(keys):
        subj1 = k1.split("_")[0]
        mov1 = movie_map.get(k1, None)
        for j, k2 in enumerate(keys):
            if j <= i:
                continue
            subj2 = k2.split("_")[0]
            mov2 = movie_map.get(k2, None)
            w1 = weights_dict[k1]
            w2 = weights_dict[k2]
            # Normalize for cosine
            w1_norm = w1 / np.linalg.norm(w1) if np.linalg.norm(w1) > 0 else w1
            w2_norm = w2 / np.linalg.norm(w2) if np.linalg.norm(w2) > 0 else w2
            # Pearson
            try:
                pear, _ = pearsonr(w1, w2)
            except Exception:
                pear = np.nan
            cos = cosine_similarity(w1_norm.reshape(1, -1), w2_norm.reshape(1, -1))[0, 0]
            # Group
            if subj1 != subj2 and mov1 == mov2:
                pearson["DS-SM"].append(pear)
                cosine["DS-SM"].append(cos)
            elif subj1 != subj2 and mov1 != mov2:
                pearson["DS-DM"].append(pear)
                cosine["DS-DM"].append(cos)
    return pearson, cosine

def plot_metric(metric_dict, metric_name, eval_label, ax=None):
    groups = ["DS-SM", "DS-DM"]
    means = [np.nanmean(metric_dict[g]) for g in groups]
    sems = [
        np.nanstd(metric_dict[g], ddof=1) / np.sqrt(np.sum(~np.isnan(metric_dict[g])))
        if np.sum(~np.isnan(metric_dict[g])) > 1 else np.nan
        for g in groups
    ]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = None

    ax.bar(groups, means, yerr=sems, capsize=5, color=BAR_COLORS)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.set_title(f"{metric_name}\n{eval_label}", fontsize=10)
    
    if fig:
        fig.tight_layout()
        return fig


def main():
    all_results = {}
    pearson_metrics = []
    cosine_metrics = []
    eval_labels_used = []

    for eval_type, eval_label in zip(EVAL_TYPES, EVAL_LABELS):
        results_file = os.path.join(RESULTS_DIR, f"population_{eval_type}.json")
        if not os.path.exists(results_file):
            print(f"Warning: {results_file} not found. Skipping.")
            continue
        weights_dict = load_weights(results_file)
        pearson, cosine = compute_pairwise_metrics(weights_dict, MOVIE_MAP)
        all_results[eval_type] = {"pearson": pearson, "cosine": cosine}
        pearson_metrics.append((pearson, eval_label))
        cosine_metrics.append((cosine, eval_label))
        eval_labels_used.append(eval_label)

        # Save raw results
        with open(f"analyses/eval_results/weights/weight_similarity_{eval_type}.json", "w") as f:
            json.dump({"pearson": pearson, "cosine": cosine}, f, indent=2)

    # Make composite PDF
    with PdfPages("weight_similarity_alltasks.pdf") as pdf:
        num_tasks = len(pearson_metrics)
        fig, axs = plt.subplots(nrows=2, ncols=num_tasks, figsize=(5 * num_tasks, 8))
        
        for i, (metric, label) in enumerate(pearson_metrics):
            plot_metric(metric, "Pearson Correlation", label, ax=axs[0, i])
        for i, (metric, label) in enumerate(cosine_metrics):
            plot_metric(metric, "Cosine Similarity", label, ax=axs[1, i])
        
        axs[0, 0].set_ylabel("Pearson", fontsize=12)
        axs[1, 0].set_ylabel("Cosine", fontsize=12)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # Save combined results
    with open("analyses/eval_results/weights/weight_similarity_alltasks.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main() 