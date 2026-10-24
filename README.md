This is a portion of an internship project I did in 2025 on building off of [Neuroprobe](https://neuroprobe.dev/), a benchmark for models trained on intracranial recordings of people watching movies.

The frozen features and the linear voltage and spectrogram baselines were provided by Andrii Zahorodnii. I used them to test whether the model's features carry information about what the person was hearing: whether a sentence just started, whether someone is talking, how loud the audio is, and who is speaking. I fit a logistic regression on the second after each word, scored it with AUROC, and compared that to the linear baselines and to the model at epoch 0, epoch 40, and epoch 100. Speaker identity is multiclass, the others are binary.

I also checked whether a decoder trained on one person still works for someone else, for the same movie and for a different movie, and whether the decoder weights look alike in those cases.

The code for that is:

- `decoding.py`: shared feature loading and the decoder
- `challenge_eval_all_subjects.py`: runs the decoder on every subject/trial and saves the weights
- `challenge_eval_task2.py`: SS-SM, DS-SM, and DS-DM transfer
- `challenge_eval_task2_weights.py`: Pearson correlation and cosine similarity of the decoder weights
- `plot_auroc_summary.py` and `plot_task2_decoder_similarity.py`: the summary plots in `figures/`

Run the commands below from this directory.

### 1. Evaluate model performance for each subject/trial combination:

Run this separately for all epochs

```bash
python challenge_eval_all_subjects.py \
    --eval_names onset,speech,volume,speaker \
    --model_dir M_nst8_dm192_nh12_nl5_5_eaM_projN_eeL_fb1_cls_lr0.003_rCONT_EXP_CLS1 \
    --model_epoch 100 \
    --verbose
```

This will:

- Process all available subject/trial pairs
- Evaluate performance on all tasks (onset, speech, volume, speaker)
- Save results to `analyses/eval_results/epoch100/`

### 2. Evaluate Cross-Subject/Movie Generalization

Next, analyze model generalization across different conditions:

```bash
python challenge_eval_task2.py
```

This will:

- Evaluate SS-SM (Same Subject, Same Movie)
- Evaluate DS-SM (Different Subject, Same Movie)
- Evaluate DS-DM (Different Subject, Different Movie)
- Save results to `analyses/eval_results/task2_decoder_similarity_results.json`

### 3. Analyze Decoder Weight Similarity

Analyze the similarity of decoder weights across conditions:

```bash
python challenge_eval_task2_weights.py
```

This will:

- Compute Pearson correlation and cosine similarity between weights
- Compare DS-SM and DS-DM conditions
- Save results to `analyses/eval_results/weights/`
- Generate `figures/weight_similarity_alltasks.pdf`

### 4. Generate Visualizations

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
