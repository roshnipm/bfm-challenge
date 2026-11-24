This is part of an internship project I did in 2025, building off of [Neuroprobe](https://neuroprobe.dev/), a benchmark for models trained on intracranial recordings of people watching movies.

Andrii Zahorodnii provided the frozen features, the linear voltage and spectrogram baselines, and a starter script that loads the features and takes the second after word onset. The saved results still use his name in the file header, which came from that script.

I wrote the logistic regression and AUROC on top of that. The tasks are whether a sentence just started, whether someone is talking, how loud the audio is, and who is speaking. Speaker identity is multiclass. The others are binary. I compared the decoder to the linear baselines and to the model at epoch 0, epoch 40, and epoch 100. I also checked whether a decoder trained on one person still works for someone else, for the same movie and for a different movie, and whether the decoder weights look alike in those cases.

The code for that is:

- `decoding.py`: shared feature loading and the decoder
- `challenge_eval_task2.py`: SS-SM, DS-SM, and DS-DM transfer
- `challenge_eval_task2_weights.py`: Pearson correlation and cosine similarity of the decoder weights
- `plot_auroc_summary.py` and `plot_task2_decoder_similarity.py`: the summary plots in `figures/`

Run the commands below from this directory.

### 1. Evaluate Cross-Subject/Movie Generalization

Next, analyze model generalization across different conditions:

```bash
python challenge_eval_task2.py
```

This will:

- Evaluate SS-SM (Same Subject, Same Movie)
- Evaluate DS-SM (Different Subject, Same Movie)
- Evaluate DS-DM (Different Subject, Different Movie)
- Save results to `analyses/eval_results/task2_decoder_similarity_results.json`

### 2. Analyze Decoder Weight Similarity

Analyze the similarity of decoder weights across conditions:

```bash
python challenge_eval_task2_weights.py
```

This will:

- Compute Pearson correlation and cosine similarity between weights
- Compare DS-SM and DS-DM conditions
- Save results to `analyses/eval_results/weights/`
- Generate `figures/weight_similarity_alltasks.pdf`

### 3. Generate Visualizations

Finally, create summary plots:

```bash
# Plot AUROC summary across all models and tasks
python plot_auroc_summary.py

# Plot decoder similarity results
python plot_task2_decoder_similarity.py
```

This will generate:

- `figures/auroc_barplot.pdf`: Performance comparison across models
- `figures/task2_decoder_similarity_alltasks.pdf`: Decoder similarity visualization

## Output Files

- `analyses/eval_results/epoch100/population_*.json`: Individual subject results
- `analyses/eval_results/task2_decoder_similarity_results.json`: Cross-subject/movie results
- `analyses/eval_results/weights/weight_similarity_*.json`: Weight similarity results
- `figures/auroc_barplot.pdf`: Performance visualization
- `figures/weight_similarity_alltasks.pdf`: Weight similarity visualization
- `figures/task2_decoder_similarity_alltasks.pdf`: Decoder similarity visualization
