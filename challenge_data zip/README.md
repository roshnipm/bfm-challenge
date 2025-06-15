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
- Save results to `task2_decoder_similarity_results.json`

### 3. Analyze Decoder Weight Similarity

Analyze the similarity of decoder weights across conditions:

```bash
python challenge_eval_task2_weights.py
```

This will:

- Compute Pearson correlation and cosine similarity between weights
- Compare DS-SM and DS-DM conditions
- Save results to `analyses/eval_results/weights/`
- Generate `weight_similarity_alltasks.pdf`

### 4. Generate Visualizations

Finally, create summary plots:

```bash
# Plot AUROC summary across all models and tasks
python plot_auroc_summary.py

# Plot decoder similarity results
python plot_task2_decoder_similarity.py
```

This will generate:

- `auroc_barplot.pdf`: Performance comparison across models
- `task2_decoder_similarity_alltasks.pdf`: Decoder similarity visualization

## Output Files

- `analyses/eval_results/epoch100/population_*.json`: Individual subject results
- `task2_decoder_similarity_results.json`: Cross-subject/movie results
- `analyses/eval_results/weights/weight_similarity_*.json`: Weight similarity results
- `auroc_barplot.pdf`: Performance visualization
- `weight_similarity_alltasks.pdf`: Weight similarity visualization
- `task2_decoder_similarity_alltasks.pdf`: Decoder similarity visualization
