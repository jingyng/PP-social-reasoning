#!/usr/bin/env python3

import json
import shutil
from pathlib import Path

# Valid labels 
VALID_LABELS = {"Hate speech", "Offensive language", "Normal"} #HateXplain
# VALID_LABELS = {"Positive", "No sentiment", "Negative"} #SST

def clean_jsonl_file(file_path, label_field="label"):
    """Remove entries with invalid labels from a JSONL file"""
    file_path = Path(file_path)

    valid_entries = []
    invalid_count = 0
    total_count = 0

    # Read and filter entries
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total_count += 1
            try:
                data = json.loads(line)
                label = data.get(label_field)

                if label not in VALID_LABELS:
                    invalid_count += 1
                    persona_id = data.get("persona_id", "unknown")
                    question_id = data.get("question_id", "unknown")
                    print(f"  Removing invalid entry: {persona_id} -> {question_id} ({label_field}: {label})")
                else:
                    valid_entries.append(line)

            except json.JSONDecodeError:
                invalid_count += 1
                print(f"  Removing malformed JSON at line {line_num}")
            except Exception as e:
                invalid_count += 1
                print(f"  Removing problematic entry at line {line_num}: {e}")

    # Only rewrite file if there were invalid entries
    if invalid_count > 0:
        # Create backup first
        backup_path = file_path.with_suffix('.jsonl.backup')
        shutil.copy2(file_path, backup_path)
        print(f"  Created backup: {backup_path}")

        # Write cleaned file
        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in valid_entries:
                f.write(entry + '\n')

    print(f"  {file_path.name}: {total_count} total, {invalid_count} removed, {len(valid_entries)} kept")
    return invalid_count

def main():
    """Clean all combo persona result files"""
    base_path = Path("../results")

    # Define directories and their label fields
    directory_configs = [
        # Qwen uses "label"
        (base_path / "results_qwen3_32b_hatexplain_r1", "label"),
        (base_path / "results_qwen3_32b_hatexplain_r2", "label"),
        (base_path / "results_qwen3_32b_hatexplain_r3", "label"),
        # Mistral uses "label"
        (base_path / "results_mistral_medium_hatexplain_r1", "label"),
        (base_path / "results_mistral_medium_hatexplain_r2", "label"),
        (base_path / "results_mistral_medium_hatexplain_r3", "label"),
        # GPT-OSS uses "label_agreement"
        (base_path / "results_gpt_oss_120b_hatexplain_r1", "label"),
        (base_path / "results_gpt_oss_120b_hatexplain_r2", "label"),
        (base_path / "results_gpt_oss_120b_hatexplain_r3", "label"),
    ]

    # Define directories and their label fields
    # directory_configs = [
    #     # Qwen uses "label"
    #     (base_path / "results_qwen3_32b_sst_r1", "model_answer"),
    #     (base_path / "results_qwen3_32b_sst_r2", "model_answer"),
    #     (base_path / "results_qwen3_32b_sst_r3", "model_answer"),
    #     # Mistral uses "label"
    #     (base_path / "results_mistral_medium_sst_r1", "model_answer"),
    #     (base_path / "results_mistral_medium_sst_r2", "model_answer"),
    #     (base_path / "results_mistral_medium_sst_r3", "model_answer"),
    #     # # GPT-OSS uses "label_agreement"
    #     (base_path / "results_gpt_oss_120b_sst_r1", "model_answer"),
    #     (base_path / "results_gpt_oss_120b_sst_r2", "model_answer"),
    #     (base_path / "results_gpt_oss_120b_sst_r3", "model_answer"),
    # ]

    print("Cleaning combo persona results...")
    print(f"Valid labels: {', '.join(sorted(VALID_LABELS))}")
    print("Automatically detecting and removing entries with invalid labels...")

    total_removed = 0
    files_processed = 0

    for directory, label_field in directory_configs:
        if not directory.exists():
            print(f"Directory not found: {directory}")
            continue

        print(f"\nProcessing directory: {directory.name} (using field: {label_field})")

        # Process all JSONL files in the directory
        for jsonl_file in sorted(directory.glob("s*.jsonl")):
            removed = clean_jsonl_file(jsonl_file, label_field)
            total_removed += removed
            files_processed += 1

    print(f"\nCleaning completed!")
    print(f"Files processed: {files_processed}")
    print(f"Total invalid entries removed: {total_removed}")

    if total_removed > 0:
        print(f"You can now re-run the generation script to fill in the missing entries.")

if __name__ == "__main__":
    main()