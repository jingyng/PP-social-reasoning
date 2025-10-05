# HatEXplain 500 Hybrid Dataset - Organized by Demographic Attributes

This directory contains the HatEXplain 500 hybrid dataset organized by demographic attributes as requested.

## Organization Structure

### Main Attribute Categories

The dataset has been organized into the following demographic attributes:

1. **Age**: No explicit age-related targets found in the dataset
2. **Education**: 3 entries (Economic status-related)
3. **Gender**: 142 entries (Men, Women, Homosexual, Heterosexual)
4. **Race/Ethnicity**: 171 entries (African, Caucasian, Asian, Hispanic, Arab, Indigenous, Indian, Minority)
5. **Religion**: 101 entries (Islam, Jewish, Christian, Hindu, Nonreligious)
6. **Other**: 85 entries (Other, Refugee, Disability)
7. **None**: 144 entries (No specific demographic targets)

### File Structure

#### Main Attribute Files
- `attribute_race_ethnicity.jsonl` (171 entries)
- `attribute_gender.jsonl` (142 entries)
- `attribute_none.jsonl` (144 entries)
- `attribute_religion.jsonl` (101 entries)
- `attribute_other.jsonl` (85 entries)
- `attribute_education.jsonl` (3 entries)

#### Detailed Subsets (`detailed/` directory)
More granular subsets within each attribute category (only subsets with ≥3 entries):

**Gender Subsets:**
- `gender_homosexual.jsonl` (44 entries)
- `gender_women.jsonl` (66 entries)
- `gender_homosexual_women.jsonl` (7 entries)
- `gender_homosexual_men.jsonl` (6 entries)
- `gender_men_women.jsonl` (9 entries)
- `gender_men.jsonl` (5 entries)

**Race/Ethnicity Subsets:**
- `race_ethnicity_african.jsonl` (80 entries)
- `race_ethnicity_caucasian.jsonl` (23 entries)
- `race_ethnicity_arab.jsonl` (18 entries)
- `race_ethnicity_hispanic.jsonl` (12 entries)
- `race_ethnicity_african_caucasian.jsonl` (12 entries)
- `race_ethnicity_asian.jsonl` (5 entries)
- `race_ethnicity_minority.jsonl` (3 entries)

**Religion Subsets:**
- `religion_islam.jsonl` (51 entries)
- `religion_jewish.jsonl` (33 entries)
- `religion_islam_jewish.jsonl` (7 entries)
- `religion_christian.jsonl` (4 entries)
- `religion_nonreligious.jsonl` (3 entries)

## Target Mapping

The following targets from the original dataset were mapped to demographic attributes:

- **Age**: [] (no age-related targets identified)
- **Education**: ['Economic']
- **Gender**: ['Men', 'Women', 'Homosexual', 'Heterosexual']
- **Race/Ethnicity**: ['African', 'Caucasian', 'Asian', 'Hispanic', 'Arab', 'Indigenous', 'Indian', 'Minority']
- **Religion**: ['Islam', 'Jewish', 'Christian', 'Hindu', 'Nonreligious']
- **Other**: ['Other', 'Refugee', 'Disability']
- **None**: ['None']

## Distribution Summary

### Most Common Attribute Combinations:
1. None only: 144 entries
2. Race/ethnicity only: 82 entries
3. Gender only: 76 entries
4. Religion only: 44 entries
5. Other only: 35 entries
6. Gender + Race/ethnicity: 33 entries
7. Race/ethnicity + Religion: 20 entries

### Key Statistics:
- **Total entries**: 500 (original dataset size preserved)
- **Entries with no specific targets**: 144 (28.8%)
- **Entries targeting race/ethnicity**: 171 (34.2%)
- **Entries targeting gender**: 142 (28.4%)
- **Entries targeting religion**: 101 (20.2%)

## Data Format

Each JSONL file contains entries with the complete original structure:
- `id`: Sequential identifier
- `post_id`: Original post identifier
- `input_text`: The text content
- `post_tokens`: Tokenized text
- `annotator_ids_sorted`: Sorted annotator IDs
- `annotator_labels_sorted`: Corresponding labels
- `annotators`: Detailed annotator information including targets
- `rationales_all`: Individual annotator rationales
- `merged_rationale_or/and`: Merged rationale masks
- `majority_label`: Final classification

## Usage Notes

- Entries can appear in multiple attribute files if they target multiple demographic categories
- The sum of main attribute files (646) exceeds 500 because entries with multiple targets are counted in each relevant category
- Only detailed subsets with 3 or more entries were created to ensure statistical significance
- All original annotator information and target specifications are preserved