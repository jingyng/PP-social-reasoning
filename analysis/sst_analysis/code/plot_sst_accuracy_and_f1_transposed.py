import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter


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
MODEL_DISPLAY_NAMES = {
    "gpt_oss_120b": "GPT-OSS-120B",
    "mistral_medium": "Mistral-Medium",
    "qwen3_32b": "Qwen3-32B"
}

# Soft background colors for each persona group to aid scanning in swapped plot
GROUP_BACKGROUND_COLORS = {
    group: color
    for group, color in zip(GROUPS, sns.color_palette('pastel', len(GROUPS)))
}

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


def format_decimal(value: float, decimals: int = 2, strip_leading_zero: bool = False) -> str:
    """Format a float with configurable decimals and optional leading zero removal."""
    formatted = f"{value:.{decimals}f}"
    if strip_leading_zero:
        if formatted.startswith('-0'):
            formatted = '-' + formatted[2:]
        elif formatted.startswith('0'):
            formatted = formatted[1:]
    return formatted


def get_group_order_by_baseline(accuracy_df, reference_model="gpt_oss_120b"):
    """Order persona groups by baseline accuracy for a reference model (ascending)."""
    model_data = accuracy_df[accuracy_df['model'] == reference_model]
    ordered = (
        model_data.dropna(subset=['baseline_mean'])
        .sort_values('baseline_mean', ascending=True)
        .drop_duplicates('group', keep='first')['group']
        .tolist()
    )
    for group in GROUPS:
        if group not in ordered:
            ordered.append(group)
    return ordered


def plot_sst_accuracy_and_f1_swapped():
    """Create swapped-axis scatter plot for SST accuracy and rationale F1 with personas and baselines."""
    accuracy_data = load_accuracy_data()
    f1_data = load_rationale_data()

    gender_order = ['Male', 'Female']
    baseline_color = '#ff7f0e'
    gender_colors = {'Male': '#1f77b4', 'Female': '#e377c2'}

    # Arrange persona groups (BO, BY, ...) top-to-bottom
    reference_order = get_group_order_by_baseline(accuracy_data)
    ordered_groups = reference_order
    y_spacing = 1.2
    group_y_positions = {group: idx * y_spacing for idx, group in enumerate(ordered_groups)}
    y_ticks = list(group_y_positions.values())
    y_min = -0.6
    y_max = (len(ordered_groups) - 1) * y_spacing + 0.6

    fig, axes = plt.subplots(len(MODEL_ORDER), 2, figsize=(13, 11), sharey=True)
    if len(MODEL_ORDER) == 1:
        axes = axes.reshape(1, 2)

    metric_configs = [
        {
            'label': 'Accuracy (%)',
            'data': accuracy_data,
            'formatter': FuncFormatter(lambda x, _: format_decimal(x, decimals=1)),
            'margin': 0.6
        },
        {
            'label': 'Token-F1',
            'data': f1_data,
            'formatter': FuncFormatter(lambda x, _: format_decimal(x, decimals=2, strip_leading_zero=True)),
            'margin': 0.01
        }
    ]

    y_offsets = {
        'baseline': 0.3,
        'Male': 0.1,
        'Female': -0.2
    }

    x_limits = [[float('inf'), float('-inf')] for _ in range(2)]

    for row_idx, model in enumerate(MODEL_ORDER):
        model_display = MODEL_DISPLAY_NAMES.get(model, model.replace('_', ' ').title())
        axes[row_idx, 0].text(-0.08, 0.5, model_display, transform=axes[row_idx, 0].transAxes,
                              rotation=90, va='center', ha='center', fontsize=14, fontweight='bold')

        for col_idx, config in enumerate(metric_configs):
            ax = axes[row_idx, col_idx]
            metric_df = config['data']
            formatter = config['formatter']
            margin = config['margin']
            model_data = metric_df[metric_df['model'] == model]
            if model_data.empty:
                ax.axis('off')
                continue

            for group in ordered_groups:
                group_subset = model_data[model_data['group'] == group]
                if group_subset.empty:
                    continue

                y_center = group_y_positions[group]

                baseline_row = group_subset.iloc[0]
                baseline_value = baseline_row['baseline_mean']
                baseline_ci = baseline_row['baseline_ci'] if pd.notna(baseline_row['baseline_ci']) else 0.0

                if pd.notna(baseline_value):
                    baseline_y = y_center + y_offsets['baseline']
                    ax.errorbar(baseline_value, baseline_y, xerr=baseline_ci, fmt='o', color=baseline_color,
                                markerfacecolor='white', markeredgecolor=baseline_color, markersize=7,
                                capsize=3, capthick=1, zorder=3)
                    x_limits[col_idx][0] = min(x_limits[col_idx][0], baseline_value - baseline_ci)
                    x_limits[col_idx][1] = max(x_limits[col_idx][1], baseline_value + baseline_ci)

                for gender in gender_order:
                    persona_row = group_subset[group_subset['gender'] == gender]
                    if persona_row.empty:
                        continue

                    value = persona_row['persona_mean'].iloc[0]
                    persona_ci_value = persona_row['persona_ci'].iloc[0]
                    persona_ci = persona_ci_value if pd.notna(persona_ci_value) else 0.0

                    if pd.notna(value):
                        persona_y = y_center + y_offsets.get(gender, 0)
                        ax.errorbar(value, persona_y, xerr=persona_ci,
                                    fmt='o', color=gender_colors[gender], alpha=0.9,
                                    markerfacecolor=gender_colors[gender], markeredgecolor='black', markersize=7,
                                    capsize=3, capthick=1, zorder=4)

                        x_limits[col_idx][0] = min(x_limits[col_idx][0], value - persona_ci)
                        x_limits[col_idx][1] = max(x_limits[col_idx][1], value + persona_ci)

            ax.set_yticks(y_ticks)
            if col_idx == 0:
                ax.set_yticklabels(ordered_groups, fontsize=13)
            else:
                ax.tick_params(axis='y', labelleft=False)

            ax.set_ylim(y_min, y_max)
            if row_idx == len(MODEL_ORDER) - 1:
                ax.set_xlabel(config['label'], fontsize=15)
            else:
                ax.set_xlabel('')
            ax.grid(True, axis='x', alpha=0.3)
            ax.xaxis.set_major_formatter(formatter)
            ax.tick_params(axis='x', labelsize=12)

    for col_idx, (xmin, xmax) in enumerate(x_limits):
        if xmin == float('inf') or xmax == float('-inf'):
            continue
        xmin -= metric_configs[col_idx]['margin']
        xmax += metric_configs[col_idx]['margin']
        for row_idx in range(len(MODEL_ORDER)):
            ax = axes[row_idx, col_idx]
            if ax.has_data():
                ax.set_xlim(xmin, xmax)

    # Shared y-axis label
    fig.text(0.04, 0.5, 'Persona Groups', rotation='vertical', va='center', fontsize=15, fontweight='bold')

    # Add column headers
    column_titles = ['Accuracy', 'Token-F1']
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].text(0.5, 1.08, title, transform=axes[0, col_idx].transAxes,
                              ha='center', fontsize=16, fontweight='bold')

    # Legend (baseline + genders)
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color=baseline_color, linestyle='None', markersize=7,
                   markerfacecolor='white', markeredgecolor=baseline_color, label='Baseline'),
        plt.Line2D([0], [0], marker='o', color=gender_colors['Male'], linestyle='None', markersize=7,
                   markerfacecolor=gender_colors['Male'], markeredgecolor='black', label='Persona (Male)'),
        plt.Line2D([0], [0], marker='o', color=gender_colors['Female'], linestyle='None', markersize=7,
                   markerfacecolor=gender_colors['Female'], markeredgecolor='black', label='Persona (Female)')
    ]

    axes[0, 1].legend(handles=legend_handles, loc='upper right', fontsize=13, frameon=True)

    fig.suptitle('SST Accuracy and Token-F1 by Persona Group (Swapped Axes)', fontsize=18, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.92)
    plt.savefig(PLOTS_DIR / 'sst_accuracy_and_f1_swapped.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_and_f1_transposed():
    """Create transposed plot: models in rows, accuracy and F1 in columns"""
    accuracy_data = load_accuracy_data()
    f1_data = load_rationale_data()

    # Create figure with 3 rows (models), 2 columns (accuracy, F1)
    fig = plt.figure(figsize=(10, 9))  # Adjusted for new layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    # Order groups by baseline accuracy of reference model
    group_baseline_order = get_group_order_by_baseline(accuracy_data)

    # Color scheme
    colors = {
        'baseline': '#ff7f0e',  # orange
        'Male': '#1f77b4',      # blue
        'Female': '#e377c2'     # pink
    }

    # Plot each model as a row
    for row, model in enumerate(MODEL_ORDER):
        # Format model name
        if model == 'gpt_oss_120b':
            model_display = 'GPT OSS 120B'
        else:
            model_display = model.replace('_', ' ').title()

        # Left column: Accuracy
        ax_acc = fig.add_subplot(gs[row, 0])

        model_acc_data = accuracy_data[accuracy_data['model'] == model].copy()
        if not model_acc_data.empty:
            # Calculate mean deltas for title annotation
            model_gender_deltas = model_acc_data.groupby('gender')['delta'].mean()
            male_delta = model_gender_deltas.get('Male', 0)
            female_delta = model_gender_deltas.get('Female', 0)

            plot_accuracy_subplot(ax_acc, model_acc_data, group_baseline_order, colors,
                                model_display, male_delta, female_delta, show_legend=(row == 0), is_bottom_row=(row == 2))

            # Add column header for first row
            if row == 0:
                ax_acc.text(0.5, 1.15, 'Accuracy', transform=ax_acc.transAxes,
                           horizontalalignment='center', fontsize=14, fontweight='bold')

        # Add rotated model name on the left side
        if not model_acc_data.empty:
            ax_acc.text(-0.15, 0.5, model_display, transform=ax_acc.transAxes,
                       rotation=90, verticalalignment='center', horizontalalignment='center',
                       fontsize=12, fontweight='bold')

        # Right column: F1
        ax_f1 = fig.add_subplot(gs[row, 1])

        model_f1_data = f1_data[f1_data['model'] == model].copy()
        if not model_f1_data.empty:
            # Calculate mean deltas for title annotation
            model_gender_deltas = model_f1_data.groupby('gender')['delta'].mean()
            male_delta = model_gender_deltas.get('Male', 0)
            female_delta = model_gender_deltas.get('Female', 0)

            plot_f1_subplot(ax_f1, model_f1_data, group_baseline_order, colors,
                          model_display, male_delta, female_delta, is_bottom_row=(row == 2))

            # Add column header for first row
            if row == 0:
                ax_f1.text(0.5, 1.15, 'Token-F1', transform=ax_f1.transAxes,
                          horizontalalignment='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'sst_accuracy_and_f1_dumbbell_transposed.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created transposed combined accuracy and F1 dumbbell plot: sst_accuracy_and_f1_dumbbell_transposed.png")


def plot_accuracy_subplot(ax, model_data, group_baseline_order, colors, model_display, male_delta, female_delta, show_legend=False, is_bottom_row=False):
    """Plot accuracy subplot for one model"""
    # Prepare data for plotting
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
        return

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

        # Plot baseline point
        ax.scatter(baseline_val, y_pos, s=60, facecolors='none',
                  edgecolors=colors['baseline'], linewidth=1.5, marker='o',
                  label='Baseline' if show_legend and y_pos == 0 else "")

        # Add baseline CI
        if pd.notna(baseline_ci):
            ax.errorbar(baseline_val, y_pos, xerr=baseline_ci,
                       color=colors['baseline'], alpha=0.6, capsize=2, capthick=0.8)

        # Plot persona points and connecting lines
        for _, row in group_data.iterrows():
            gender = row['gender']
            persona_val = row['persona_mean']
            persona_ci = row['persona_ci']

            y_offset = 0.08 if gender == 'Male' else -0.08
            y_persona = y_pos + y_offset

            # Connecting line
            ax.plot([baseline_val, persona_val], [y_pos, y_persona],
                   color='gray', alpha=0.6, linewidth=1, zorder=1)

            # Persona point
            marker = 'o' if gender == 'Male' else '^'
            ax.scatter(persona_val, y_persona, s=60,
                      color=colors[gender], marker=marker, alpha=0.8,
                      label=f'{gender} Personas' if show_legend and y_pos == 0 else "", zorder=3)

            # Add persona CI
            if pd.notna(persona_ci):
                ax.errorbar(persona_val, y_persona, xerr=persona_ci,
                           color=colors[gender], alpha=0.6, capsize=2, capthick=0.8)

        y_positions[group] = y_pos
        y_pos += 0.5

    # Format subplot
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([g for g in y_positions.keys()], fontsize=11)
    if is_bottom_row:
        ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'♂ {male_delta:+.1f}%, ♀ {female_delta:+.1f}%', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(65, 95)

    # Add legend only to first subplot
    if show_legend:
        ax.legend(loc='upper left', fontsize=10)


def plot_f1_subplot(ax, model_data, group_baseline_order, colors, model_display, male_delta, female_delta, is_bottom_row=False):
    """Plot F1 subplot for one model"""
    # Prepare data for plotting
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
        return

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

        # Plot baseline point
        ax.scatter(baseline_val, y_pos, s=60, facecolors='none',
                  edgecolors=colors['baseline'], linewidth=1.5, marker='o')

        # Add baseline CI
        if pd.notna(baseline_ci):
            ax.errorbar(baseline_val, y_pos, xerr=baseline_ci,
                       color=colors['baseline'], alpha=0.6, capsize=2, capthick=0.8)

        # Plot persona points
        for _, row in group_data.iterrows():
            gender = row['gender']
            persona_val = row['persona_mean']
            persona_ci = row['persona_ci']

            y_offset = 0.08 if gender == 'Male' else -0.08
            y_persona = y_pos + y_offset

            # Connecting line
            ax.plot([baseline_val, persona_val], [y_pos, y_persona],
                   color='gray', alpha=0.6, linewidth=1, zorder=1)

            # Persona point
            marker = 'o' if gender == 'Male' else '^'
            ax.scatter(persona_val, y_persona, s=60,
                      color=colors[gender], marker=marker, alpha=0.8, zorder=3)

            # Add persona CI
            if pd.notna(persona_ci):
                ax.errorbar(persona_val, y_persona, xerr=persona_ci,
                           color=colors[gender], alpha=0.6, capsize=2, capthick=0.8)

        y_positions[group] = y_pos
        y_pos += 0.5

    # Format subplot
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([g for g in y_positions.keys()], fontsize=11)
    if is_bottom_row:
        ax.set_xlabel('Token-F1', fontsize=12)
    ax.set_title(f'♂ {male_delta:+.3f}, ♀ {female_delta:+.3f}', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    # Set fixed x-limits for F1 scores to ensure alignment
    ax.set_xlim(0.20, 0.52)

    # Format x-axis to show only decimal places
    def format_decimal(x, pos):
        if x >= 1:
            return f'{x:.0f}'
        elif x <= 0:
            return '0'
        else:
            return f'{x:.2f}'[1:]

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))


def main():
    """Generate SST accuracy/F1 visualizations (transposed and swapped styles)."""
    print("Creating SST accuracy and Token-F1 visualizations...")

    PLOTS_DIR.mkdir(exist_ok=True)

    print("- Building swapped-axis accuracy/F1 scatter...")
    plot_sst_accuracy_and_f1_swapped()

    print("- Building transposed dumbbell accuracy/F1 plot...")
    plot_accuracy_and_f1_transposed()

    print(f"\nPlots saved to: {PLOTS_DIR}")
    print("✅ SST accuracy and Token-F1 visualizations completed!")


if __name__ == "__main__":
    main()
