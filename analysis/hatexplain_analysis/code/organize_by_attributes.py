"""
Organize HatEXplain 500 hybrid dataset by demographic attributes.

This script separates the hatexplain 500 samples into different subsets
according to the target groups they belong to.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Set

# Define target mappings to demographic attributes
TARGET_MAPPINGS = {
    # 'age': [],
    # 'education': ['Economic'],
    'gender': ['Men', 'Women', 'Homosexual', 'Heterosexual'],
    'race_ethnicity': ['African', 'Caucasian', 'Asian', 'Hispanic', 'Arab',
                       'Indigenous', 'Indian', 'Minority'],
    'religion': ['Islam', 'Jewish', 'Christian', 'Hindu', 'Nonreligious'],
    'other': ['Other', 'Refugee', 'Disability'],
    'none': ['None']
}

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of entries."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries

def save_jsonl(entries: List[Dict], file_path: str):
    """Save entries to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

def get_targets_from_entry(entry: Dict) -> Set[str]:
    """Extract all unique targets from an entry's annotators."""
    targets = set()
    for annotator in entry.get('annotators', []):
        for target in annotator.get('target', []):
            targets.add(target)
    return targets

def map_targets_to_attributes(targets: Set[str]) -> Dict[str, Set[str]]:
    """Map targets to their corresponding demographic attributes."""
    attribute_mapping = defaultdict(set)

    for target in targets:
        for attribute, target_list in TARGET_MAPPINGS.items():
            if target in target_list:
                attribute_mapping[attribute].add(target)

    return attribute_mapping

def organize_by_attributes(input_file: str, output_dir: str):
    """
    Organize dataset by demographic attributes.

    Creates:
    - Main attribute files (attribute_*.jsonl)
    - Detailed subset files in detailed/ directory (only subsets with ≥3 entries)
    """
    # Load data
    print(f"Loading data from {input_file}...")
    entries = load_jsonl(input_file)
    print(f"Loaded {len(entries)} entries")

    # Data structures for organizing
    attribute_entries = defaultdict(list)  # Main attribute categories
    detailed_entries = defaultdict(list)   # Detailed subsets

    # Process each entry
    for entry in entries:
        targets = get_targets_from_entry(entry)
        target_to_attributes = map_targets_to_attributes(targets)

        # Add to main attribute files
        for attribute, target_subset in target_to_attributes.items():
            attribute_entries[attribute].append(entry)

            # Create detailed subset key (sorted for consistency)
            if attribute != 'none':
                detailed_key = f"{attribute}_" + "_".join(sorted([t.lower() for t in target_subset]))
                detailed_entries[detailed_key].append(entry)

    # Save main attribute files
    print("\nSaving main attribute files...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for attribute, entries_list in sorted(attribute_entries.items()):
        output_file = output_path / f"attribute_{attribute}.jsonl"
        save_jsonl(entries_list, str(output_file))
        print(f"  {attribute}: {len(entries_list)} entries -> {output_file.name}")

    # Save detailed subset files (only those with ≥3 entries)
    print("\nSaving detailed subset files (≥3 entries)...")
    detailed_path = output_path / "detailed"
    detailed_path.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for subset_key, entries_list in sorted(detailed_entries.items()):
        if len(entries_list) >= 3:
            output_file = detailed_path / f"{subset_key}.jsonl"
            save_jsonl(entries_list, str(output_file))
            print(f"  {subset_key}: {len(entries_list)} entries")
            saved_count += 1

    print(f"\nSaved {saved_count} detailed subset files")

    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)

    print("\nMain Attribute Distribution:")
    for attribute, entries_list in sorted(attribute_entries.items(),
                                         key=lambda x: len(x[1]), reverse=True):
        percentage = (len(entries_list) / len(entries)) * 100
        print(f"  {attribute}: {len(entries_list)} entries ({percentage:.1f}%)")

    # Attribute combination statistics
    combination_counts = Counter()
    for entry in entries:
        targets = get_targets_from_entry(entry)
        target_to_attributes = map_targets_to_attributes(targets)
        attributes = tuple(sorted(target_to_attributes.keys()))
        combination_counts[attributes] += 1

    print("\nMost Common Attribute Combinations:")
    for combo, count in combination_counts.most_common(10):
        if combo:
            combo_str = " + ".join(combo)
        else:
            combo_str = "No targets"
        print(f"  {combo_str}: {count} entries")

    print("\n" + "="*60)
    print(f"Total entries processed: {len(entries)}")
    print(f"Main attribute files: {len(attribute_entries)}")
    print(f"Detailed subset files (≥3): {saved_count}")
    print("="*60)

if __name__ == "__main__":
    # Set paths (relative to repository root)
    base_dir = Path(__file__).parent.parent.parent
    input_file = base_dir / "datasets/personas_&_questions/hatexplain_500_hybrid.jsonl"
    output_dir = base_dir / "datasets/personas_&_questions/attribute_subsets"

    # Run organization
    organize_by_attributes(str(input_file), str(output_dir))
    print("\nOrganization complete!")
