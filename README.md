# PP-social-reasoning
This is the repository for the paper "Persona Prompts as a Lens on LLM Social Reasoning"

## Repository Structure

```
.
├── personas_samples_generation/    # Persona generation scripts
├── datasets/                       # Processed datasets (HateXplain, SST-2, CoS-E)
├── generate_persona_explanations/  # Model inference scripts
├── analysis/                       # Analysis code for each dataset
└── results/                        # Experimental results
```

## Datasets

The repository includes three datasets:

- **HatEXplain**: Hate speech detection
- **SST (Stanford Sentiment Treebank)**: Sentiment classification
- **CoS-E**: Commonsense reasoning task

SST-2 and CoS-E datasets are re-annotated subsets from BRWRR ("Being Right for Whose Right Reasons" by Jakobsen et al.). Dataset files are organized by demographic groups (e.g., `BY_processed.json`, `WY_processed.json`, `LO_processed.json`).

## Models

The experiments use three large language models:

- **GPT-OSS-120B** (gptoss)
- **Mistral-Medium** (mistral)
- **Qwen3-32B** (qwen)

## Reproducing Results

Follow these steps to reproduce the paper's results:

### 1. Data and Persona Preparation

Generate personas and prepare datasets using scripts in `personas_samples_generation/`:

- `generate_single_attribute_personas.py`: Generate single-attribute personas
- `generate_group_personas_cose.py`: Generate group-aligned composite personas for CoS-E and SST-2 (the two datasets share the same personas)
- `questions_hatexplain_500.py`: Select HateXplain 500 samples

### 2. Generate Labels and Rationales

Run model inference using scripts in `generate_persona_explanations/`:

**Baseline explanations:**
- `{model}_sst_baseline_explanations.py`
- `{model}_cose_baseline_explanations.py`
- `{model}_hatexplain_baseline_explanations.py`

**Persona-based explanations:**
- `{model}_sst_persona_explanations.py`
- `{model}_cose_persona_explanations.py`
- `{model}_hatexplain_persona_explanations.py`

Replace `{model}` with `gptoss`, `mistral`, or `qwen`.

### 3. Analysis

Run analysis scripts for each dataset under `analysis/`:

**HatEXplain Analysis (`analysis/hatexplain_analysis/code/`):**
- `persona_accuracy.py`: All personas and baseline task performance (accuracy, Macro-F1, ME, over-flagging rate)
- `persona_accuracy_by_subgroup.py`: Subgroup-specific task performance (supgroups are divided by annotated targets)
- `compute_persona_agreement.py`: Inter-persona agreement on labels
- `compute_persona_rationale_inter_agreement.py`: Inter-persona agreement on rationales
- `extract_ground_truth_rationales.py`: Extract ground truth rationales
- `create_subgroup_f1_table.py`: F1 scores by subgroup (divided by annotated targets)
- `cot_reasoning_analysis.ipynb`: Analyzing models' reasoning output
- `plot_f1_macro_results.py`: Macro-F1 visualizations
- `plot_flagging_rates_heatmaps.py`: Flagging rate heatmaps
- `plot_rationale_baseline_vs_personas_filtered.py`: Token-F1 visualizations

**SST-2 Analysis (`analysis/sst_analysis/code/`):**
- `baseline_accuracy_per_run.py`: Baseline task performance (accuracy, Macro-F1)
- `persona_accuracy_per_run.py`: Persona task performance (accuracy, Macro-F1)
- `compute_persona_label_iaa_per_run.py`: Inter-persona agreement on labels
- `compute_persona_rationale_iaa_per_run.py`: Inter-persona agreement on rationales
- `compute_rationale_token_iou_f1.py`: Token-F1 and IoU-F1 scores
- `plot_sst_binary_f1_and_token_f1_transposed.py`: Generate plots

**CoS-E Analysis (`analysis/cose_analysis/code/`):**
- `baseline_accuracy_per_run.py`: Baseline task performance (accuracy, Macro-F1)
- `persona_accuracy_per_run.py`: Persona task performance (accuracy, Macro-F1)
- `compute_persona_label_iaa_per_run.py`: Inter-persona agreement on labels
- `compute_persona_rationale_iaa_per_run.py`: Inter-persona agreement on rationales
- `compute_rationale_token_iou_f1.py`: Token-F1 and IoU-F1 scores
- `plot_cose_accuracy_and_f1_transposed.py`: Generate plots

## Results

Experimental results are stored in `results/` with the naming convention:
```
results_{model}_{dataset}_r{run_number}/
```

For example:
- `results_gpt_oss_120b_sst_r1/`
- `results_mistral_medium_hatexplain_r2/`
- `results_qwen_cose_r3/`

Each experiment is run 3 times (r1, r2, r3) for robustness.