"""
Create a PDF table showing macro-F1 performance by subgroup.

This script generates a table comparing baseline and persona performance
across different demographic subgroups using macro-F1 scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

def get_persona_display_name(persona_id):
    """Convert persona ID to display name"""
    name_map = {
        '15': 'Age 15',
        '35': 'Age 35',
        '65': 'Age 65',
        'no_formal_education': 'No Formal Edu',
        'high_school_education': 'High School',
        'higher_education': 'Higher Edu',
        'male': 'Male',
        'female': 'Female',
        'not_lonely': 'Not Lonely',
        'somewhat_lonely': 'Lonely',
        'left-wing': 'Left-wing',
        'right-wing': 'Right-wing',
        'centrist': 'Centrist',
        'white': 'White',
        'black': 'Black',
        'asian': 'Asian',
        'christian': 'Christian',
        'muslim': 'Muslim',
        'jewish': 'Jewish',
        'atheist': 'Atheist',
        'hindu': 'Hindu'
    }
    return name_map.get(persona_id, persona_id)

def create_latex_table(df, output_file):
    """Create a LaTeX table showing top-3 personas for each model-subgroup combination"""

    # Subgroup display names and order
    subgroup_order = ['all', 'gender', 'race_ethnicity', 'religion', 'none']
    subgroup_names = {
        'all': 'All Samples',
        'gender': 'Gender',
        'race_ethnicity': 'Race/Ethnicity',
        'religion': 'Religion',
        'none': 'No Target'
    }

    model_order = ['gpt_oss_120b', 'mistral_medium', 'qwen3_32b']
    model_names = {
        'gpt_oss_120b': 'GPT-4o-mini',
        'mistral_medium': 'Mistral Medium',
        'qwen3_32b': 'Qwen 2.5 32B'
    }

    latex_lines = []
    latex_lines.append(r'\begin{table*}[t]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Top-3 Performing Personas by Macro-F1 for Each Model and Subgroup}')
    latex_lines.append(r'\label{tab:subgroup_f1_top3}')
    latex_lines.append(r'\scriptsize')
    latex_lines.append(r'\begin{tabular}{lllccc}')
    latex_lines.append(r'\hline')
    latex_lines.append(r'\textbf{Subgroup} & \textbf{Model} & \textbf{Persona} & \textbf{Macro-F1} & \textbf{Std} & \textbf{n} \\')
    latex_lines.append(r'\hline')

    for subgroup in subgroup_order:
        subgroup_data = df[df['subgroup'] == subgroup]

        if subgroup_data.empty:
            continue

        # Get baseline data
        baseline_data = subgroup_data[subgroup_data['persona_type'] == 'baseline']

        # Get persona data
        persona_data = subgroup_data[subgroup_data['persona_type'] == 'persona']

        # Get sample size
        if not baseline_data.empty:
            n_samples = int(baseline_data.iloc[0]['n_samples_mean'])
        else:
            n_samples = 0

        first_model = True
        for model in model_order:
            # Get top-3 personas for this specific model
            model_personas = persona_data[persona_data['model'] == model]

            if model_personas.empty:
                continue

            # Sort by F1 score
            model_personas_sorted = model_personas.sort_values('f1_macro_mean', ascending=False)
            top_3 = model_personas_sorted.head(3)

            # Baseline for this model
            model_baseline = baseline_data[baseline_data['model'] == model]

            if not model_baseline.empty:
                baseline_f1 = model_baseline.iloc[0]['f1_macro_mean']
                baseline_std = model_baseline.iloc[0]['f1_macro_std']
            else:
                baseline_f1 = 0.0
                baseline_std = 0.0

            # Calculate number of rows for this model (baseline + top 3)
            num_rows = 4

            # Add subgroup name only for first model
            if first_model:
                latex_lines.append(r'\multirow{12}{*}{\textbf{' + subgroup_names[subgroup] + r'}}')
                first_model = False

            # Add model name
            latex_lines.append(r' & \multirow{' + str(num_rows) + r'}{*}{\textit{' +
                             model_names[model] + r'}}')

            # Baseline row
            latex_lines.append(r' & Baseline & ' +
                             f'{baseline_f1:.3f} & {baseline_std:.3f} & {n_samples}' + r' \\')

            # Top 3 persona rows
            for idx, (_, row) in enumerate(top_3.iterrows(), 1):
                persona_display = get_persona_display_name(row['persona_id'])
                f1 = row['f1_macro_mean']
                std = row['f1_macro_std']

                latex_lines.append(r' & & ' + f'{idx}. {persona_display}' + r' & ' +
                                 f'{f1:.3f} & {std:.3f} & {n_samples}' + r' \\')

        if subgroup != subgroup_order[-1]:
            latex_lines.append(r'\hline')

    latex_lines.append(r'\hline')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\vspace{0.2cm}')
    latex_lines.append(r'\begin{flushleft}')
    latex_lines.append(r'\scriptsize')
    latex_lines.append(r'\textit{Note:} Values show mean macro-F1 and standard deviation across three independent runs. ')
    latex_lines.append(r'Top-3 personas are ranked separately for each model within each subgroup. ')
    latex_lines.append(r'n indicates the number of samples in each subgroup.')
    latex_lines.append(r'\end{flushleft}')
    latex_lines.append(r'\end{table*}')

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"LaTeX table saved to {output_file}")

def create_visual_table(df, output_file):
    """Create a visual PDF table using matplotlib showing top-3 personas per model"""

    # Subgroup display names and order
    subgroup_order = ['all', 'gender', 'race_ethnicity', 'religion', 'none']
    subgroup_names = {
        'all': 'All',
        'gender': 'Gender',
        'race_ethnicity': 'Race/Eth.',
        'religion': 'Religion',
        'none': 'No Target'
    }

    model_order = ['gpt_oss_120b', 'mistral_medium', 'qwen3_32b']
    model_names = {
        'gpt_oss_120b': 'GPT-4o-mini',
        'mistral_medium': 'Mistral Medium',
        'qwen3_32b': 'Qwen 2.5 32B'
    }

    # Prepare data for table
    table_data = []
    row_colors = []

    for subgroup in subgroup_order:
        subgroup_data = df[df['subgroup'] == subgroup]

        if subgroup_data.empty:
            continue

        # Get baseline data
        baseline_data = subgroup_data[subgroup_data['persona_type'] == 'baseline']

        # Get persona data
        persona_data = subgroup_data[subgroup_data['persona_type'] == 'persona']

        # Get sample size
        if not baseline_data.empty:
            n_samples = int(baseline_data.iloc[0]['n_samples_mean'])
        else:
            n_samples = 0

        # Add subgroup header row
        table_data.append([f'{subgroup_names[subgroup]} (n={n_samples})', '', '', ''])
        row_colors.append('#E8E8E8')

        for model in model_order:
            # Get top-3 personas for this specific model
            model_personas = persona_data[persona_data['model'] == model]

            if model_personas.empty:
                continue

            # Sort by F1 score
            model_personas_sorted = model_personas.sort_values('f1_macro_mean', ascending=False)
            top_3 = model_personas_sorted.head(3)

            # Baseline for this model
            model_baseline = baseline_data[baseline_data['model'] == model]

            if not model_baseline.empty:
                baseline_f1 = model_baseline.iloc[0]['f1_macro_mean']
                baseline_std = model_baseline.iloc[0]['f1_macro_std']
            else:
                baseline_f1 = 0.0
                baseline_std = 0.0

            # Baseline row
            table_data.append([f'  {model_names[model]}', 'Baseline',
                              f'{baseline_f1:.3f}', f'±{baseline_std:.3f}'])
            row_colors.append('#F0F0F0')

            # Top 3 persona rows
            for idx, (_, row) in enumerate(top_3.iterrows(), 1):
                persona_display = get_persona_display_name(row['persona_id'])
                f1 = row['f1_macro_mean']
                std = row['f1_macro_std']

                table_data.append(['', f'  {idx}. {persona_display}',
                                  f'{f1:.3f}', f'±{std:.3f}'])
                row_colors.append('white')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 18))
    ax.axis('tight')
    ax.axis('off')

    # Column headers
    col_labels = ['Subgroup / Model', 'Persona', 'Macro-F1', 'Std']

    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     cellLoc='left',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Color header
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Color rows and set alignment
    for i, color in enumerate(row_colors):
        for j in range(len(col_labels)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(color)

            # Bold subgroup names
            if j == 0 and color == '#E8E8E8':
                cell.set_text_props(weight='bold')

            # Right align numeric columns
            if j >= 2:
                cell.set_text_props(ha='right')

    plt.title('Top-3 Performing Personas by Macro-F1 for Each Model and Subgroup',
              fontsize=12, fontweight='bold', pad=20)

    # Add note
    note_text = ('Note: Top-3 personas ranked separately for each model within each subgroup.\n'
                'Values show mean macro-F1 and standard deviation across three independent runs.')
    plt.figtext(0.5, 0.01, note_text, ha='center', fontsize=7, style='italic', wrap=True)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Visual table saved to {output_file}")

def create_detailed_table(df, output_file):
    """Create a transposed table with mean±std in one cell"""

    # Subgroup display names and order
    subgroup_order = ['all', 'gender', 'race_ethnicity', 'religion', 'none']
    subgroup_names = {
        'all': 'All',
        'gender': 'Gender',
        'race_ethnicity': 'Race/Eth.',
        'religion': 'Religion',
        'none': 'No Target'
    }

    model_order = ['gpt_oss_120b', 'mistral_medium', 'qwen3_32b']
    model_names = {
        'gpt_oss_120b': 'GPT-4o-mini',
        'mistral_medium': 'Mistral Medium',
        'qwen3_32b': 'Qwen 2.5 32B'
    }

    latex_lines = []
    latex_lines.append(r'\begin{table*}[t]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Top-3 Personas Macro-F1 Performance by Subgroup (Transposed)}')
    latex_lines.append(r'\label{tab:subgroup_f1_transposed}')
    latex_lines.append(r'\scriptsize')
    latex_lines.append(r'\begin{adjustbox}{max width=\textwidth}')

    # Create column headers: Model | All (Persona + F1) | Gender (Persona + F1) | etc.
    # Each subgroup gets 2 columns: persona name and F1 score
    latex_lines.append(r'\begin{tabular}{l|' + 'lc' * len(subgroup_order) + '}')
    latex_lines.append(r'\hline')

    # Header row with multicolumn for each subgroup
    header = r'\textbf{Model}'
    for subgroup in subgroup_order:
        subgroup_data = df[df['subgroup'] == subgroup]
        if not subgroup_data.empty:
            baseline_data = subgroup_data[subgroup_data['persona_type'] == 'baseline']
            if not baseline_data.empty:
                n_samples = int(baseline_data.iloc[0]['n_samples_mean'])
            else:
                n_samples = 0
            header += r' & \multicolumn{2}{c}{\textbf{' + subgroup_names[subgroup] + r'} \textit{(n=' + str(n_samples) + r')}}'
        else:
            header += r' & \multicolumn{2}{c}{\textbf{' + subgroup_names[subgroup] + r'}}'

    latex_lines.append(header + r' \\')

    # Sub-header row for Persona and F1 columns
    subheader = r''
    for _ in subgroup_order:
        subheader += r' & Persona & F1\%'
    latex_lines.append(subheader + r' \\')
    latex_lines.append(r'\hline')

    # For each model
    for model_idx, model in enumerate(model_order):
        # First pass: collect all F1 values for each subgroup to find max
        subgroup_max_f1 = {}

        for subgroup in subgroup_order:
            subgroup_data = df[df['subgroup'] == subgroup]

            # Collect baseline F1
            baseline_data = subgroup_data[(subgroup_data['persona_type'] == 'baseline') &
                                         (subgroup_data['model'] == model)]
            f1_values = []
            if not baseline_data.empty:
                f1_values.append(baseline_data.iloc[0]['f1_macro_mean'])

            # Collect top-3 persona F1 values
            persona_data = subgroup_data[(subgroup_data['persona_type'] == 'persona') &
                                        (subgroup_data['model'] == model)]
            if not persona_data.empty:
                persona_sorted = persona_data.sort_values('f1_macro_mean', ascending=False)
                for idx in range(min(3, len(persona_sorted))):
                    f1_values.append(persona_sorted.iloc[idx]['f1_macro_mean'])

            # Find max F1 for this subgroup
            if f1_values:
                subgroup_max_f1[subgroup] = max(f1_values)

        # Top 3 persona rows - collect all top personas for each subgroup
        top_personas_by_subgroup = {}
        for subgroup in subgroup_order:
            subgroup_data = df[df['subgroup'] == subgroup]
            persona_data = subgroup_data[(subgroup_data['persona_type'] == 'persona') &
                                        (subgroup_data['model'] == model)]

            if not persona_data.empty:
                # Sort by F1 score
                persona_sorted = persona_data.sort_values('f1_macro_mean', ascending=False)
                top_personas_by_subgroup[subgroup] = persona_sorted.head(3)

        # Generate 4 rows (baseline + 3 personas) with model name spanning all rows
        for row_idx in range(4):
            if row_idx == 0:
                # First row: model name with baseline data
                row_content = r'\multirow{4}{*}{\textit{' + model_names[model] + r'}}'

                for subgroup in subgroup_order:
                    subgroup_data = df[df['subgroup'] == subgroup]
                    baseline_data = subgroup_data[(subgroup_data['persona_type'] == 'baseline') &
                                                 (subgroup_data['model'] == model)]

                    if not baseline_data.empty:
                        mean = baseline_data.iloc[0]['f1_macro_mean']
                        mean_pct = mean * 100  # Convert to percentage
                        std = baseline_data.iloc[0]['f1_macro_std'] * 100    # Convert to percentage

                        # Check if this is the max value for this subgroup
                        if subgroup in subgroup_max_f1 and abs(mean - subgroup_max_f1[subgroup]) < 1e-6:
                            row_content += f' & Baseline & \\textbf{{{mean_pct:.1f}}}$_{{\pm{std:.1f}}}$'
                        else:
                            row_content += f' & Baseline & {mean_pct:.1f}$_{{\pm{std:.1f}}}$'
                    else:
                        row_content += r' & Baseline & ---'

                latex_lines.append(row_content + r' \\')
            else:
                # Persona rows (rank 1, 2, 3)
                rank = row_idx  # row_idx 1, 2, 3 corresponds to rank 1, 2, 3
                rank_row = r''

                # Add persona name and F1 scores for each subgroup (in separate columns)
                for subgroup in subgroup_order:
                    if subgroup in top_personas_by_subgroup and len(top_personas_by_subgroup[subgroup]) >= rank:
                        top_persona = top_personas_by_subgroup[subgroup].iloc[rank - 1]
                        persona_name = get_persona_display_name(top_persona['persona_id'])
                        f1 = top_persona['f1_macro_mean']
                        f1_pct = f1 * 100  # Convert to percentage
                        std = top_persona['f1_macro_std'] * 100  # Convert to percentage

                        # Shorten persona name if needed
                        if len(persona_name) > 12:
                            persona_name = persona_name[:10] + '.'

                        # Check if this is the max value for this subgroup
                        if subgroup in subgroup_max_f1 and abs(f1 - subgroup_max_f1[subgroup]) < 1e-6:
                            rank_row += f' & {persona_name} & \\textbf{{{f1_pct:.1f}}}$_{{\pm{std:.1f}}}$'
                        else:
                            rank_row += f' & {persona_name} & {f1_pct:.1f}$_{{\pm{std:.1f}}}$'
                    else:
                        rank_row += r' & --- & ---'

                latex_lines.append(rank_row + r' \\')

        # Add separator between models
        if model != model_order[-1]:
            latex_lines.append(r'\hline')

    latex_lines.append(r'\hline')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{adjustbox}')
    latex_lines.append(r'\vspace{0.2cm}')
    latex_lines.append(r'\begin{flushleft}')
    latex_lines.append(r'\scriptsize')
    latex_lines.append(r'\textit{Note:} F1\% shows macro-F1 percentage mean$_{\pm\text{std}}$ (e.g., 75.0$_{\pm1.2}$ = 75.0\%). ')
    latex_lines.append(r'\textbf{Bold values} indicate the highest F1 score for each model-subgroup combination. ')
    latex_lines.append(r'Top-3 personas ranked separately for each model-subgroup combination. ')
    latex_lines.append(r'Requires \textbackslash usepackage\{adjustbox\} in preamble.')
    latex_lines.append(r'\end{flushleft}')
    latex_lines.append(r'\end{table*}')

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"Detailed LaTeX table saved to {output_file}")

def main():
    # Load averaged results
    csv_file = THIS_DIR / "csv" / "accuracy_by_subgroup_averaged.csv"

    if not csv_file.exists():
        print(f"Error: {csv_file} not found!")
        print("Please run persona_accuracy_by_subgroup.py first to generate the data.")
        return

    df = pd.read_csv(csv_file)

    # Create output directory for tables
    output_dir = THIS_DIR / "tables"
    output_dir.mkdir(exist_ok=True)

    # Generate LaTeX table
    latex_file = output_dir / "subgroup_f1_table.tex"
    create_latex_table(df, latex_file)

    # Generate detailed LaTeX table
    detailed_latex_file = output_dir / "subgroup_f1_table_detailed.tex"
    create_detailed_table(df, detailed_latex_file)

    # Generate visual PDF table
    pdf_file = output_dir / "subgroup_f1_table.pdf"
    create_visual_table(df, pdf_file)

    print("\nTable generation completed!")
    print(f"Files saved in: {output_dir}")

if __name__ == "__main__":
    main()
