import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as ticker


THIS_DIR = Path(__file__).resolve().parent
CSV_DIR = THIS_DIR / "csv"
PLOTS_DIR = THIS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Group information for SST
GROUPS = ["BO", "BY", "LO", "LY", "WO", "WY"]
GROUP_LABELS = {
    "BO": "45yo African American",
    "BY": "25yo African American",
    "LO": "45yo Hispanic",
    "LY": "25yo Hispanic",
    "WO": "45yo Caucasian",
    "WY": "25yo Caucasian"
}

# Model order
MODEL_ORDER = ["gpt_oss_120b", "mistral_medium", "qwen3_32b"]

# Set style
plt.style.use('default')
sns.set_palette("tab10")


def extract_gender_from_persona(persona_text):
    """Extract gender from persona text"""
    if 'Male' in persona_text:
        return 'Male'
    elif 'Female' in persona_text:
        return 'Female'
    else:
        return 'Unknown'


def load_accuracy_data():
    """Load baseline and persona accuracy data"""
    baseline_file = CSV_DIR / "baseline_accuracy_sst_per_run.csv"
    persona_file = CSV_DIR / "persona_accuracy_sst_per_run.csv"

    baseline_df = pd.read_csv(baseline_file)
    persona_df = pd.read_csv(persona_file)

    # Filter for matched personas only
    persona_df = persona_df[persona_df['matches_group'] == True]

    # Add gender information to persona data
    persona_df['gender'] = persona_df['persona_text'].apply(extract_gender_from_persona)

    # Compute baseline statistics (mean and CI across runs)
    baseline_stats = baseline_df.groupby(['model', 'group']).agg({
        'accuracy': ['mean', 'std', 'count']
    }).reset_index()
    baseline_stats.columns = ['model', 'group', 'baseline_mean', 'baseline_std', 'baseline_count']

    # Compute 95% CI for baseline
    baseline_stats['baseline_ci'] = 1.96 * baseline_stats['baseline_std'] / np.sqrt(baseline_stats['baseline_count'])

    # Compute persona statistics by gender (mean and CI across runs)
    persona_stats = persona_df.groupby(['model', 'group', 'gender']).agg({
        'accuracy': ['mean', 'std', 'count']
    }).reset_index()
    persona_stats.columns = ['model', 'group', 'gender', 'persona_mean', 'persona_std', 'persona_count']

    # Compute 95% CI for personas
    persona_stats['persona_ci'] = 1.96 * persona_stats['persona_std'] / np.sqrt(persona_stats['persona_count'])

    # Merge baseline and persona stats
    combined_stats = pd.merge(baseline_stats, persona_stats, on=['model', 'group'], how='outer')

    # Compute delta (persona - baseline)
    combined_stats['delta'] = combined_stats['persona_mean'] - combined_stats['baseline_mean']

    # Convert accuracy values to percentages
    combined_stats['baseline_mean'] *= 100
    combined_stats['baseline_ci'] *= 100
    combined_stats['persona_mean'] *= 100
    combined_stats['persona_ci'] *= 100
    combined_stats['delta'] *= 100

    return combined_stats


def load_rationale_data():
    """Load baseline and persona rationale data"""
    baseline_file = CSV_DIR / "baseline_rationale_agreement_sst_per_run.csv"
    persona_file = CSV_DIR / "persona_rationale_agreement_sst_per_run.csv"

    baseline_df = pd.read_csv(baseline_file)
    persona_df = pd.read_csv(persona_file)

    # Filter for matched personas only
    persona_df = persona_df[persona_df['matches_group'] == True]

    # Add gender information to persona data
    persona_df['gender'] = persona_df['persona_text'].apply(extract_gender_from_persona)

    # Compute baseline statistics (mean and CI across runs)
    baseline_stats = baseline_df.groupby(['model', 'group']).agg({
        'rationale_f1_mean': ['mean', 'std', 'count']
    }).reset_index()
    baseline_stats.columns = ['model', 'group', 'baseline_mean', 'baseline_std', 'baseline_count']

    # Compute 95% CI for baseline
    baseline_stats['baseline_ci'] = 1.96 * baseline_stats['baseline_std'] / np.sqrt(baseline_stats['baseline_count'])

    # Compute persona statistics by gender (mean and CI across runs)
    persona_stats = persona_df.groupby(['model', 'group', 'gender']).agg({
        'rationale_f1_mean': ['mean', 'std', 'count']
    }).reset_index()
    persona_stats.columns = ['model', 'group', 'gender', 'persona_mean', 'persona_std', 'persona_count']

    # Compute 95% CI for personas
    persona_stats['persona_ci'] = 1.96 * persona_stats['persona_std'] / np.sqrt(persona_stats['persona_count'])

    # Merge baseline and persona stats
    combined_stats = pd.merge(baseline_stats, persona_stats, on=['model', 'group'], how='outer')

    # Compute delta (persona - baseline)
    combined_stats['delta'] = combined_stats['persona_mean'] - combined_stats['baseline_mean']

    return combined_stats


def plot_accuracy_and_f1_combined():
    """Create combined plot with accuracy (top) and F1 (bottom), models in columns"""
    accuracy_data = load_accuracy_data()
    f1_data = load_rationale_data()

    # Create figure with 2 rows, 3 columns
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.15, height_ratios=[1, 1])

    # Order groups by overall baseline accuracy
    group_baseline_order = accuracy_data.groupby('group')['baseline_mean'].mean().sort_values(ascending=False).index

    # Color scheme
    colors = {
        'baseline': '#d62728',  # red
        'Male': '#1f77b4',      # blue
        'Female': '#2ca02c'     # green
    }

    # Plot accuracy (top row)
    for col, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[0, col])

        model_data = accuracy_data[accuracy_data['model'] == model].copy()
        if model_data.empty:
            continue

        # Calculate mean deltas for title annotation
        model_gender_deltas = model_data.groupby('gender')['delta'].mean()
        male_delta = model_gender_deltas.get('Male', 0)
        female_delta = model_gender_deltas.get('Female', 0)

        # Prepare data for plotting - order by baseline accuracy
        plot_data = []
        for group in group_baseline_order:
            group_data = model_data[model_data['group'] == group]
            if not group_data.empty:
                baseline_row = group_data.iloc[0]
                for _, row in group_data.iterrows():
                    if pd.notna(row['persona_mean']):
                        plot_data.append({
                            'group': group,
                            'baseline_mean': baseline_row['baseline_mean'],
                            'baseline_ci': baseline_row['baseline_ci'],
                            'persona_mean': row['persona_mean'],
                            'persona_ci': row['persona_ci'],
                            'gender': row['gender'],
                            'delta': row['delta']
                        })

        if not plot_data:
            continue

        plot_df = pd.DataFrame(plot_data)

        # Create dumbbell plot
        y_positions = {}
        y_pos = 0

        for group in group_baseline_order:
            group_data = plot_df[plot_df['group'] == group]
            if group_data.empty:
                continue

            baseline_val = group_data['baseline_mean'].iloc[0]
            baseline_ci = group_data['baseline_ci'].iloc[0]

            # Plot baseline point (hollow circle)
            ax.scatter(baseline_val, y_pos, s=100, facecolors='none',
                      edgecolors=colors['baseline'], linewidth=2, marker='o',
                      label='Baseline' if col == 0 and y_pos == 0 else "")

            # Add baseline CI
            if pd.notna(baseline_ci):
                ax.errorbar(baseline_val, y_pos, xerr=baseline_ci,
                           color=colors['baseline'], alpha=0.6, capsize=3, capthick=1)

            # Plot persona points and connecting lines
            for _, row in group_data.iterrows():
                gender = row['gender']
                persona_val = row['persona_mean']
                persona_ci = row['persona_ci']

                # Slight y-offset for gender separation
                y_offset = 0.1 if gender == 'Male' else -0.1
                y_persona = y_pos + y_offset

                # Connecting line (dumbbell stick)
                ax.plot([baseline_val, persona_val], [y_pos, y_persona],
                       color='gray', alpha=0.6, linewidth=1.5, zorder=1)

                # Persona point (filled circle/triangle)
                marker = 'o' if gender == 'Male' else '^'
                ax.scatter(persona_val, y_persona, s=100,
                          color=colors[gender], marker=marker, alpha=0.8,
                          label=f'{gender} Personas' if col == 0 and y_pos == 0 else "", zorder=3)

                # Add persona CI
                if pd.notna(persona_ci):
                    ax.errorbar(persona_val, y_persona, xerr=persona_ci,
                               color=colors[gender], alpha=0.6, capsize=3, capthick=1)

            y_positions[group] = y_pos
            y_pos += 0.6

        # Format model name
        if model == 'gpt_oss_120b':
            model_display = 'GPT OSS 120B'
        else:
            model_display = model.replace('_', ' ').title()

        # Set up accuracy plot
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([g for g in y_positions.keys()])
        ax.set_xlabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{model_display}\nMean diff: male {male_delta:+.1f}%, female {female_delta:+.1f}%',
                     fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

        # Set reasonable x-limits for SST accuracy
        ax.set_xlim(65, 95)

        # Add legend only to first subplot
        if col == 0:
            ax.legend(loc='upper right', fontsize=10)

    # Plot F1 (bottom row) - now with both baseline and persona data
    for col, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[1, col])

        model_data = f1_data[f1_data['model'] == model].copy()
        if model_data.empty:
            continue

        # Calculate mean deltas for title annotation
        model_gender_deltas = model_data.groupby('gender')['delta'].mean()
        male_delta = model_gender_deltas.get('Male', 0)
        female_delta = model_gender_deltas.get('Female', 0)

        # Prepare data for plotting - order by baseline F1
        plot_data = []
        for group in group_baseline_order:  # Use same order as accuracy
            group_data = model_data[model_data['group'] == group]
            if not group_data.empty:
                baseline_row = group_data.iloc[0]
                for _, row in group_data.iterrows():
                    if pd.notna(row['persona_mean']):
                        plot_data.append({
                            'group': group,
                            'baseline_mean': baseline_row['baseline_mean'],
                            'baseline_ci': baseline_row['baseline_ci'],
                            'persona_mean': row['persona_mean'],
                            'persona_ci': row['persona_ci'],
                            'gender': row['gender'],
                            'delta': row['delta']
                        })

        if not plot_data:
            continue

        plot_df = pd.DataFrame(plot_data)

        # Create dumbbell plot for F1
        y_positions = {}
        y_pos = 0

        for group in group_baseline_order:
            group_data = plot_df[plot_df['group'] == group]
            if group_data.empty:
                continue

            baseline_val = group_data['baseline_mean'].iloc[0]
            baseline_ci = group_data['baseline_ci'].iloc[0]

            # Plot baseline point (hollow circle)
            ax.scatter(baseline_val, y_pos, s=100, facecolors='none',
                      edgecolors=colors['baseline'], linewidth=2, marker='o',
                      label='Baseline' if col == 0 and y_pos == 0 else "")

            # Add baseline CI
            if pd.notna(baseline_ci):
                ax.errorbar(baseline_val, y_pos, xerr=baseline_ci,
                           color=colors['baseline'], alpha=0.6, capsize=3, capthick=1)

            # Plot persona points and connecting lines
            for _, row in group_data.iterrows():
                gender = row['gender']
                persona_val = row['persona_mean']
                persona_ci = row['persona_ci']

                # Slight y-offset for gender separation
                y_offset = 0.1 if gender == 'Male' else -0.1
                y_persona = y_pos + y_offset

                # Connecting line (dumbbell stick)
                ax.plot([baseline_val, persona_val], [y_pos, y_persona],
                       color='gray', alpha=0.6, linewidth=1.5, zorder=1)

                # Persona point (filled circle/triangle)
                marker = 'o' if gender == 'Male' else '^'
                ax.scatter(persona_val, y_persona, s=100,
                          color=colors[gender], marker=marker, alpha=0.8,
                          label=f'{gender} Personas' if col == 0 and y_pos == 0 else "", zorder=3)

                # Add persona CI
                if pd.notna(persona_ci):
                    ax.errorbar(persona_val, y_persona, xerr=persona_ci,
                               color=colors[gender], alpha=0.6, capsize=3, capthick=1)

            y_positions[group] = y_pos
            y_pos += 0.6

        # Set up F1 plot
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([g for g in y_positions.keys()])
        ax.set_xlabel('Token-F1', fontsize=12)
        ax.set_title(f'male {male_delta:+.3f}, female {female_delta:+.3f}', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

        # Set reasonable x-limits for F1 scores
        all_vals = list(plot_df['baseline_mean'].dropna().values) + list(plot_df['persona_mean'].dropna().values)
        if len(all_vals) > 0:
            x_min, x_max = min(all_vals), max(all_vals)
            x_range = x_max - x_min
            ax.set_xlim(max(0, x_min - 0.1 * x_range), min(1, x_max + 0.1 * x_range))

        # Format x-axis to show only decimal places (e.g., .52 instead of 0.52)
        def format_decimal(x, pos):
            if x >= 1:
                return f'{x:.0f}'
            elif x <= 0:
                return '0'
            else:
                return f'{x:.2f}'[1:]  # Remove the leading '0'

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))

        # No legend needed in F1 section (already shown in accuracy section)

    # plt.suptitle('SST: Baseline vs Persona Performance (Accuracy & Token-F1)',
                # fontsize=18, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(PLOTS_DIR / 'sst_accuracy_and_f1_dumbbell.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created combined accuracy and F1 dumbbell plot: sst_accuracy_and_f1_dumbbell.png")


def main():
    """Generate combined accuracy and F1 dumbbell plot for SST"""
    print("Creating SST combined accuracy and F1 dumbbell plot...")

    PLOTS_DIR.mkdir(exist_ok=True)

    plot_accuracy_and_f1_combined()

    print(f"\nPlots saved to: {PLOTS_DIR}")
    print("✅ Combined accuracy and F1 visualization completed!")


if __name__ == "__main__":
    main()