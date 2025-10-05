import json
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def load_hatexplain_500_samples():
    """Load the 500 selected samples from hatexplain_500_hybrid.jsonl"""
    samples_file = Path("datasets/personas_&_questions/hatexplain_500_hybrid.jsonl")
    samples = {}

    with open(samples_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            samples[item['post_id']] = item

    return samples


def load_original_hatexplain_dataset():
    """Load the original HatEXplain dataset"""
    dataset_file = Path("datasets/personas_&_questions/cache_hatexplain_dataset.json")

    with open(dataset_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_majority_rationale(rationales: List[List[int]], majority_label: str, token_length: int) -> List[int]:
    """
    Compute majority rationale based on the rules:
    - If majority_label is 'Normal', return all zeros
    - Otherwise, compute majority voting on rationales (>=2 out of 3)
    - When there are 2 rationales, the missing one is all zeros (Normal label)
    """
    if majority_label == 'Normal':
        # For normal labels, rationale should be all zeros
        return [0] * token_length

    if not rationales:
        return [0] * token_length

    # Pad rationales to always have 3 (missing ones are all zeros for Normal labels)
    padded_rationales = rationales[:]
    while len(padded_rationales) < 3:
        padded_rationales.append([0] * token_length)

    # Get rationale length
    rationale_length = token_length
    majority_rationale = []

    for pos in range(rationale_length):
        # Count how many annotators marked this position as important
        count = sum(1 for rationale in padded_rationales if rationale[pos] == 1)
        # Majority voting: >=2 out of 3
        majority_rationale.append(1 if count >= 2 else 0)

    return majority_rationale


def extract_ground_truth_rationales():
    """Extract ground truth rationales for the 500 selected samples"""
    # Load data
    selected_samples = load_hatexplain_500_samples()
    original_dataset = load_original_hatexplain_dataset()

    ground_truth_rationales = {}

    print(f"Processing {len(selected_samples)} selected samples...")

    for post_id, sample_info in selected_samples.items():
        if post_id not in original_dataset:
            print(f"Warning: {post_id} not found in original dataset")
            continue

        original_data = original_dataset[post_id]

        # Use the majority label from the 500 samples file
        majority_label = sample_info['majority_label']

        # Extract rationales
        rationales = original_data.get('rationales', [])

        # Get token length
        token_length = len(original_data['post_tokens'])

        # Compute majority rationale
        majority_rationale = compute_majority_rationale(rationales, majority_label, token_length)

        ground_truth_rationales[post_id] = {
            'post_id': post_id,
            'id': sample_info['id'],
            'majority_label': majority_label,
            'post_tokens': original_data['post_tokens'],
            'annotator_labels': [ann['label'] for ann in original_data['annotators']],
            'rationales_all': rationales,
            'majority_rationale': majority_rationale
        }

    return ground_truth_rationales


def save_ground_truth_rationales(ground_truth_rationales: Dict):
    """Save ground truth rationales to file"""
    output_dir = Path("3_analysis/3_hatexplain_analysis")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "ground_truth_rationales.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth_rationales, f, indent=2, ensure_ascii=False)

    print(f"Saved ground truth rationales to: {output_file}")

    # Also save as JSONL for easier processing
    output_jsonl = output_dir / "ground_truth_rationales.jsonl"
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for data in ground_truth_rationales.values():
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Saved ground truth rationales to: {output_jsonl}")


def main():
    print("Extracting ground truth rationales from original HatEXplain dataset...")

    ground_truth_rationales = extract_ground_truth_rationales()

    # Print statistics
    label_counts = Counter()
    rationale_stats = {'has_rationale': 0, 'no_rationale': 0}

    for data in ground_truth_rationales.values():
        label_counts[data['majority_label']] += 1
        if any(data['majority_rationale']):
            rationale_stats['has_rationale'] += 1
        else:
            rationale_stats['no_rationale'] += 1

    print(f"\nProcessed {len(ground_truth_rationales)} samples")
    print(f"Label distribution: {dict(label_counts)}")
    print(f"Rationale statistics: {rationale_stats}")

    # Save results
    save_ground_truth_rationales(ground_truth_rationales)

    print("\nGround truth rationale extraction completed!")


if __name__ == "__main__":
    main()