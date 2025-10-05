import json
import os
import pandas as pd
import re
import textstat
from functools import reduce
from tqdm import tqdm


def extract_cot_reasoning(root_dir):
    """
    Extracts Chain-of-Thought reasoning from .jsonl files in a directory structure.

    Args:
        root_dir (str): The path to the main results directory (e.g., 'datasets').

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted information.
    """
    extracted_data = []

    # Walk through the entire directory structure
    for dirpath, _, filenames in tqdm(os.walk(root_dir), desc="Processing directories"):
        if "merged" not in dirpath and "hybrid" not in dirpath and "_500_" in dirpath:
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    file_path = os.path.join(dirpath, filename)

                    try:
                        run_name = os.path.basename(dirpath)  # e.g., "results_mistral_medium_sst_r1"
                    except Exception:
                        run_name = "unknown"

                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())

                                persona_text = data.get("persona_text", data.get("persona_id", "N/A"))
                                input_text = data.get("input_text",
                                                      data.get("sentence", "N/A"))  # Handle different key names
                                model_answer = data.get("model_answer", "N/A")
                                label_agreement = data.get("label_agreement", "N/A")

                                raw_response = data.get("raw_response", "")
                                reasoning_output = data.get("reasoning_output", "")

                                cot_reasoning = ""

                                # **Primary Strategy: Find in-character <think> tags**
                                # The re.DOTALL flag is crucial to match across multiple lines.
                                match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)

                                if match:
                                    cot_reasoning = match.group(1).strip()
                                else:
                                    # **Fallback Strategy: Use the 'reasoning_output' field**
                                    # This is often meta-commentary but still useful.
                                    cot_reasoning = reasoning_output.strip()

                                extracted_data.append({
                                    "run_name": run_name + "/" + filename,
                                    "persona_text": persona_text,
                                    "input_text": input_text,
                                    "model_answer": model_answer,
                                    "label_agreement": label_agreement,
                                    "cot_reasoning": cot_reasoning,
                                })

                            except json.JSONDecodeError:
                                # print(f"Warning: Could not decode JSON line in {file_path}")
                                continue

    return pd.DataFrame(extracted_data)


# Set the root directory containing all results folders
DATASETS_ROOT = '../datasets'
df_cot = extract_cot_reasoning(DATASETS_ROOT)

# Display the first few rows of the extracted data
print(df_cot.head())

# --- Education Personas ---
# Filter for rows where 'persona_text' contains "_nfe"
df_nfe = df_cot[df_cot['persona_text'].str.contains('_nfe', na=False)].copy()

# Filter for rows where 'persona_text' contains "_he"
df_he = df_cot[df_cot['persona_text'].str.contains('_he', na=False)].copy()


# --- Political View Personas ---
# Filter for rows where 'persona_text' contains "_l"
df_l = df_cot[df_cot['persona_text'].str.contains('_l', na=False)].copy()

# Filter for rows where 'persona_text' contains "_r"
df_r = df_cot[df_cot['persona_text'].str.contains('_r', na=False)].copy()

# --- Verification (Optional but recommended) ---
print("--- Education Personas ---")
print(f"Found {len(df_nfe)} 'no formal education' entries.")
print(f"Found {len(df_he)} 'higher education' entries.")
print("\n--- Political View Personas ---")
print(f"Found {len(df_l)} 'left-wing' entries.")
print(f"Found {len(df_r)} 'right-wing' entries.")

# --- 1. Prepare each DataFrame for merging ---
# We select the key columns and rename the 'cot_reasoning' column to be unique.
# This prevents column name clashes during the merge.

df_nfe_renamed = df_nfe[['run_name', 'input_text', 'cot_reasoning']].rename(
    columns={'cot_reasoning': 'cot_reasoning_nfe'}
)
df_he_renamed = df_he[['run_name', 'input_text', 'cot_reasoning']].rename(
    columns={'cot_reasoning': 'cot_reasoning_he'}
)
df_l_renamed = df_l[['run_name', 'input_text', 'cot_reasoning']].rename(
    columns={'cot_reasoning': 'cot_reasoning_l'}
)
df_r_renamed = df_r[['run_name', 'input_text', 'cot_reasoning']].rename(
    columns={'cot_reasoning': 'cot_reasoning_r'}
)

# --- 2. Create a list of the prepared DataFrames ---
dataframes_to_merge = [
    df_nfe_renamed,
    df_he_renamed,
    df_l_renamed,
    df_r_renamed
]

# --- 3. Use functools.reduce to merge all DataFrames sequentially ---
# The 'reduce' function applies the pd.merge operation iteratively.
# 'on' specifies the columns to join by (our unique key for an input).
# 'how="outer"' ensures that we keep all rows, even if one persona is missing
# a result for a specific input (it will show as NaN).

df_merged_personas = reduce(
    lambda left, right: pd.merge(left, right, on=['run_name', 'input_text'], how='outer'),
    dataframes_to_merge
)

# --- 4. Display the results ---
print("Successfully merged the persona DataFrames.")
print(f"The final DataFrame has {len(df_merged_personas)} rows and {len(df_merged_personas.columns)} columns.")

print("\nColumns in the final DataFrame:")
print(df_merged_personas.columns)

print("\nSample of the merged DataFrame:")
print(df_merged_personas.head())


# Save the results to a CSV for easier analysis
#df_merged_personas.to_csv('extracted_cot_reasoning.csv', index=False)
#print(f"\nExtracted {len(df_merged_personas)} CoT entries and saved to extracted_cot_reasoning.csv")

# --- 1. Define Helper Functions ---

def get_model_family(run_name):
    """Extracts the base model family from the run_name string."""
    if "gpt_oss_120b" in run_name:
        return "gpt_oss_120b"
    if "mistral_medium" in run_name:
        return "mistral_medium"
    if "qwen3_32b" in run_name:
        return "qwen3_32b"
    return "unknown"

def calculate_word_count(text):
    """Calculates word count, handling potential NaN values."""
    if isinstance(text, str):
        return len(text.split())
    return 0

def calculate_flesch_ease(text):
    """Calculates Flesch Reading Ease, handling NaN and short texts."""
    if isinstance(text, str) and len(text.split()) > 10: # Readability on very short text is unstable
        try:
            return textstat.flesch_reading_ease(text)
        except:
            return None # Handle potential errors in the library
    return None

def calculate_ttr(text):
    """Calculates Type-Token Ratio, a measure of lexical diversity."""
    if isinstance(text, str) and text:
        tokens = text.lower().split()
        if len(tokens) == 0:
            return 0
        return len(set(tokens)) / len(tokens)
    return 0

# --- 2. Prepare DataFrame for Analysis ---

# Create a new column for the model family for easy grouping
df_merged_personas['model_family'] = df_merged_personas['run_name'].apply(get_model_family)

# Identify the CoT columns to analyze
cot_columns = [
    'cot_reasoning_nfe', 'cot_reasoning_he',
    'cot_reasoning_l', 'cot_reasoning_r'
]

# --- 3. Calculate Metrics for Each CoT Column ---

print("\n--- Calculating Automated Metrics ---")
for col in tqdm(cot_columns, desc="Analyzing CoT columns"):
    # Calculate Word Count
    word_count_col_name = col.replace('cot_reasoning', 'word_count')
    df_merged_personas[word_count_col_name] = df_merged_personas[col].apply(calculate_word_count)

    # Calculate Flesch Reading Ease
    flesch_col_name = col.replace('cot_reasoning', 'flesch_ease')
    df_merged_personas[flesch_col_name] = df_merged_personas[col].apply(calculate_flesch_ease)

    # Calculate Type-Token ratio
    ttr_col_name = col.replace('cot_reasoning', 'ttr')
    df_merged_personas[ttr_col_name] = df_merged_personas[col].apply(calculate_ttr)

# --- 4. Aggregate Results by Model Family and Print ---

metric_columns = [col for col in df_merged_personas.columns if 'word_count' in col or 'flesch_ease' in col]

print("\n--- Aggregated Analysis Results by Model Family ---")
for metric in metric_columns:
    print(f"\nAverage '{metric}':")
    # Group by model family and calculate the mean for the current metric
    agg_result = df_merged_personas.groupby('model_family')[metric].mean().round(2)
    print(agg_result)
    print("-" * 40)

