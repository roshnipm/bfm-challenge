"""
Analyzes decoder weight similarity across subjects and movies.

1. Different subjects watching the same movie (DS-SM)
2. Different subjects watching different movies (DS-DM)

Similarity is measured using both Pearson correlation and cosine similarity.
Results are saved as JSON files (for future plotting) and visualized in a PDF.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.backends.backend_pdf import PdfPages

from decoding import MOVIE_BY_SUBJECT_TRIAL, TASKS

# configuration
results_dir = "analyses/eval_results/epoch100"
eval_types = TASKS
eval_labels = ["Sentence Onset", "Speech", "Audio Volume", "Speaker Identity"]

# noted slightly different from before
movie_map = {f"btbank{subject_trial}": movie for subject_trial, movie in MOVIE_BY_SUBJECT_TRIAL.items()}
bar_colors = ["#355C8D", "#2CA6A4"]  # blue, teal

# load weights from results file
def load_weights(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # create dictionary mapping subject/trial to weights
    weights_dict = {}
    for subj_trial, res in results["evaluation_results"].items():
        folds = res["population"]["one_second_after_onset"]["folds"]
        # take mean weights across folds for each subject/trial
        weights = np.array([fold["weights"] for fold in folds])
        mean_weights = np.mean(weights, axis=0)
        weights_dict[subj_trial] = mean_weights
    
    return weights_dict

def compute_pairwise_metrics(weights_dict, movie_map):
    keys = list(weights_dict.keys())
    # excluding SS-SM since train/test splits were done differently
    pearson = {"DS-SM": [], "DS-DM": []}
    cosine = {"DS-SM": [], "DS-DM": []}
    
    # compare all pairs of weights
    for i, k1 in enumerate(keys):
        # 0 is subject, 1 is trial
        subj1 = k1.split("_")[0]
        mov1 = movie_map.get(k1, None)

        # going through each pair of weights
        for j, k2 in enumerate(keys):
            # skip if same subject/trial pair/already done
            if j <= i: 
                continue
            
            # get subject/trial/movie for second pair
            subj2 = k2.split("_")[0]
            mov2 = movie_map.get(k2, None)
            w1 = weights_dict[k1]
            w2 = weights_dict[k2]
            
            # normalize weights before calculating cosine similarity
            w1_norm = w1 / np.linalg.norm(w1) if np.linalg.norm(w1) > 0 else w1
            w2_norm = w2 / np.linalg.norm(w2) if np.linalg.norm(w2) > 0 else w2
            
            # compute similarities
            pear, _ = pearsonr(w1, w2)
            cos = cosine_similarity(w1_norm.reshape(1, -1), w2_norm.reshape(1, -1))[0, 0]
            
            # group by condition (DS-SM or DS-DM)
            # check if different subject/same movie
            if subj1 != subj2 and mov1 == mov2:
                pearson["DS-SM"].append(pear)
                cosine["DS-SM"].append(cos)
            # check if different subject/different movie
            elif subj1 != subj2 and mov1 != mov2:
                pearson["DS-DM"].append(pear)
                cosine["DS-DM"].append(cos)
    
    return pearson, cosine

def plot_metric(metric_dict, metric_name, eval_label, ax=None):
    groups = ["DS-SM", "DS-DM"]
    # compute means/SEMs across groups
    # metric_dict has pearson/cosine for each group
    means = [np.nanmean(metric_dict[g]) for g in groups]
    sems = [
        # ddof=1 for sample standard deviation
        np.nanstd(metric_dict[g], ddof=1) / np.sqrt(np.sum(~np.isnan(metric_dict[g])))
        for g in groups
    ]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()  # get figure from the provided axis
    
    ax.bar(groups, means, yerr=sems, capsize=5, color=bar_colors)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name}\n{eval_label}", fontsize=12)
    
    return fig

def main():
    all_results = {}
    pearson_metrics = []
    cosine_metrics = []
    
    # Process each evaluation type
    for eval_type, eval_label in zip(eval_types, eval_labels):
        results_file = os.path.join(results_dir, f"population_{eval_type}.json")
        
        # Load weights and compute similarities
        weights_dict = load_weights(results_file)
        pearson, cosine = compute_pairwise_metrics(weights_dict, movie_map)
        
        # Store results
        all_results[eval_type] = {"pearson": pearson, "cosine": cosine}
        pearson_metrics.append((pearson, eval_label))
        cosine_metrics.append((cosine, eval_label))
        
        # Save individual task results
        with open(f"analyses/eval_results/weights/weight_similarity_{eval_type}.json", "w") as f:
            json.dump({"pearson": pearson, "cosine": cosine}, f, indent=2)
    
    # Create composite PDF with all plots
    os.makedirs("figures", exist_ok=True)
    with PdfPages("figures/weight_similarity_alltasks.pdf") as pdf:
        num_tasks = len(pearson_metrics)
        fig, axs = plt.subplots(nrows=2, ncols=num_tasks, figsize=(5 * num_tasks, 8))
        
        # Plot Pearson correlations
        for i, (metric, label) in enumerate(pearson_metrics):
            plot_metric(metric, "Pearson Correlation", label, ax=axs[0, i])
        
        # Plot cosine similarities
        for i, (metric, label) in enumerate(cosine_metrics):
            plot_metric(metric, "Cosine Similarity", label, ax=axs[1, i])
        
        # Add labels
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