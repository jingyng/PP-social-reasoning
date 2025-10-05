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

# Group information for CoSE
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


def extract_gender_from_persona_code(persona_code):
    """Extract gender from persona code (e.g., '25_m_b' -> 'Male', '45_f_w' -> 'Female')"""
    if '_m_' in persona_code:
        return 'Male'
    elif '_f_' in persona_code:
        return 'Female'
    else:
        return 'Unknown'


def persona_code_matches_group(persona_code, group):
    """Check if persona code matches the demographic group"""
    # Parse persona code: age_gender_ethnicity
    parts = persona_code.split('_')
    if len(parts) != 3:
        return False

    age_code, gender_code, eth_code = parts

    # Map to group codes
    age_mapping = {'25': 'Y', '45': 'O'}
    eth_mapping = {'b': 'B', 'l': 'L', 'w': 'W'}

    expected_age = age_mapping.get(age_code)
    expected_eth = eth_mapping.get(eth_code)

    if not expected_age or not expected_eth:
        return False

    expected_group = expected_eth + expected_age
    return expected_group == group


def load_accuracy_data():
    """Load baseline and persona accuracy data"""
    baseline_file = CSV_DIR / "baseline_accuracy_cose_per_run.csv"
    persona_file = CSV_DIR / "persona_accuracy_cose_per_run.csv"

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
    """Load baseline and persona Token-F1 data"""
    baseline_file = CSV_DIR / "baseline_token_iou_f1_cose_per_run.csv"
    persona_file = CSV_DIR / "persona_token_iou_f1_cose_per_run.csv"

    baseline_df = pd.read_csv(baseline_file)
    persona_df = pd.read_csv(persona_file)

    # Add gender information and matching status
    persona_df['gender'] = persona_df['persona_code'].apply(extract_gender_from_persona_code)
    persona_df['matches_group'] = persona_df.apply(
        lambda row: persona_code_matches_group(row['persona_code'], row['group']), axis=1
    )

    # Filter for matched personas only
    persona_df = persona_df[persona_df['matches_group'] == True]

    # Compute baseline statistics (mean and CI across runs)
    baseline_stats = baseline_df.groupby(['model', 'group']).agg({
        'token_f1': ['mean', 'std', 'count']
    }).reset_index()
    baseline_stats.columns = ['model', 'group', 'baseline_mean', 'baseline_std', 'baseline_count']

    # Compute 95% CI for baseline
    baseline_stats['baseline_ci'] = 1.96 * baseline_stats['baseline_std'] / np.sqrt(baseline_stats['baseline_count'])

    # Compute persona statistics by gender (mean and CI across runs)
    persona_stats = persona_df.groupby(['model', 'group', 'gender']).agg({
        'token_f1': ['mean', 'std', 'count']
    }).reset_index()
    persona_stats.columns = ['model', 'group', 'gender', 'persona_mean', 'persona_std', 'persona_count']

    # Compute 95% CI for personas
    persona_stats['persona_ci'] = 1.96 * persona_stats['persona_std'] / np.sqrt(persona_stats['persona_count'])

    # Merge baseline and persona stats
    combined_stats = pd.merge(baseline_stats, persona_stats, on=['model', 'group'], how='outer')

    # Compute delta (persona - baseline)
    combined_stats['delta'] = combined_stats['persona_mean'] - combined_stats['baseline_mean']

    return combined_stats


def load_iou_f1_data():
    """Load baseline and persona IOU-F1 data from per-run files"""
    baseline_file = CSV_DIR / "baseline_token_iou_f1_cose_per_run.csv"
    persona_file = CSV_DIR / "persona_token_iou_f1_cose_per_run.csv"

    baseline_df = pd.read_csv(baseline_file)
    persona_df = pd.read_csv(persona_file)

    # Add gender information and matching status
    persona_df['gender'] = persona_df['persona_code'].apply(extract_gender_from_persona_code)
    persona_df['matches_group'] = persona_df.apply(
        lambda row: persona_code_matches_group(row['persona_code'], row['group']), axis=1
    )

    # Filter for matched personas only
    persona_df = persona_df[persona_df['matches_group'] == True]

    # Compute baseline statistics (mean and CI across runs)
    baseline_stats = baseline_df.groupby(['model', 'group']).agg({
        'iou_f1': ['mean', 'std', 'count']
    }).reset_index()
    baseline_stats.columns = ['model', 'group', 'baseline_mean', 'baseline_std', 'baseline_count']

    # Compute 95% CI for baseline
    baseline_stats['baseline_ci'] = 1.96 * baseline_stats['baseline_std'] / np.sqrt(baseline_stats['baseline_count'])

    # Compute persona statistics by gender (mean and CI across runs)
    persona_stats = persona_df.groupby(['model', 'group', 'gender']).agg({
        'iou_f1': ['mean', 'std', 'count']
    }).reset_index()
    persona_stats.columns = ['model', 'group', 'gender', 'persona_mean', 'persona_std', 'persona_count']

    # Compute 95% CI for personas
    persona_stats['persona_ci'] = 1.96 * persona_stats['persona_std'] / np.sqrt(persona_stats['persona_count'])

    # Merge baseline and persona stats
    combined_stats = pd.merge(baseline_stats, persona_stats, on=['model', 'group'], how='outer')

    # Compute delta (persona - baseline)
    combined_stats['delta'] = combined_stats['persona_mean'] - combined_stats['baseline_mean']

    return combined_stats


def get_group_order_by_baseline(accuracy_df, reference_model="gpt_oss_120b"):
    """Order persona groups by baseline accuracy for a reference model (ascending)."""
    model_data = accuracy_df[accuracy_df['model'] == reference_model]
    ordered = (
        model_data.dropna(subset=['baseline_mean'])
        .sort_values('baseline_mean', ascending=True)
        .drop_duplicates('group', keep='first')['group']
        .tolist()
    )
    # Append any groups missing from the reference model in original order
    for group in GROUPS:
        if group not in ordered:
            ordered.append(group)
    return ordered


def format_decimal(value: float, decimals: int = 2, strip_leading_zero: bool = False) -> str:
    """Format float with optional leading zero removal (e.g., .23)."""
    formatted = f"{value:.{decimals}f}"
    if strip_leading_zero:
        if formatted.startswith('-0'):
            formatted = '-' + formatted[2:]
        elif formatted.startswith('0'):
            formatted = formatted[1:]
    return formatted


def plot_cose_accuracy_and_f1_swapped():
    """Create swapped-axis scatter plot for CoSE accuracy and rationale F1 results."""
    accuracy_data = load_accuracy_data()
    f1_data = load_rationale_data()

    gender_order = ['Male', 'Female']
    baseline_color = '#ff7f0e'
    gender_colors = {'Male': '#1f77b4', 'Female': '#e377c2'}  # Original colors

    ordered_groups = get_group_order_by_baseline(accuracy_data)
    y_spacing = 2.2  # Much larger spacing for better group separation
    group_y_positions = {group: idx * y_spacing for idx, group in enumerate(ordered_groups)}
    y_ticks = list(group_y_positions.values())
    y_min = -1.0
    y_max = (len(ordered_groups) - 1) * y_spacing + 1.0

    fig, axes = plt.subplots(len(MODEL_ORDER), 2, figsize=(16, 15), sharey=True)  # Taller figure for better spacing
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

    y_offsets = {'baseline': 0.4, 'Male': 0.15, 'Female': -0.25}  # Increased offsets for better separation
    x_limits = [[float('inf'), float('-inf')] for _ in range(2)]

    for row_idx, model in enumerate(MODEL_ORDER):
        model_display = MODEL_DISPLAY_NAMES.get(model, model.replace('_', ' ').title())
        axes[row_idx, 0].text(-0.12, 0.5, model_display, transform=axes[row_idx, 0].transAxes,
                              rotation=90, va='center', ha='center', fontsize=16, fontweight='bold')  # Larger font

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
                                markerfacecolor='white', markeredgecolor=baseline_color, markersize=12,
                                capsize=6, capthick=2.5, zorder=3, markeredgewidth=3, elinewidth=2.5)  # Much larger markers and error bars
                    x_limits[col_idx][0] = min(x_limits[col_idx][0], baseline_value - baseline_ci)
                    x_limits[col_idx][1] = max(x_limits[col_idx][1], baseline_value + baseline_ci)

                for gender in gender_order:
                    persona_row = group_subset[group_subset['gender'] == gender]
                    if persona_row.empty:
                        continue

                    value = persona_row['persona_mean'].iloc[0]
                    persona_ci_val = persona_row['persona_ci'].iloc[0]
                    persona_ci = persona_ci_val if pd.notna(persona_ci_val) else 0.0

                    if pd.notna(value):
                        persona_y = y_center + y_offsets.get(gender, 0)
                        ax.errorbar(value, persona_y, xerr=persona_ci,
                                    fmt='o', color=gender_colors[gender], alpha=0.9,
                                    markerfacecolor=gender_colors[gender], markeredgecolor='black', markersize=12,
                                    capsize=6, capthick=2.5, zorder=4, markeredgewidth=2.5, elinewidth=2.5)  # Much larger markers and error bars

                        x_limits[col_idx][0] = min(x_limits[col_idx][0], value - persona_ci)
                        x_limits[col_idx][1] = max(x_limits[col_idx][1], value + persona_ci)

            # Add subtle horizontal lines between groups for better separation
            for i in range(len(ordered_groups) - 1):
                y_separator = (y_ticks[i] + y_ticks[i+1]) / 2
                ax.axhline(y=y_separator, color='lightgray', linestyle='-', alpha=0.3, linewidth=0.8, zorder=0)

            ax.set_yticks(y_ticks)
            if col_idx == 0:
                ax.set_yticklabels(ordered_groups, fontsize=15, fontweight='bold')  # Larger, bold group labels
            else:
                ax.tick_params(axis='y', labelleft=False)

            ax.set_ylim(y_min, y_max)
            if row_idx == len(MODEL_ORDER) - 1:
                ax.set_xlabel(config['label'], fontsize=17, fontweight='bold')  # Larger axis labels
            else:
                ax.set_xlabel('')
            # Remove grid lines for cleaner look
            ax.xaxis.set_major_formatter(formatter)
            ax.tick_params(axis='x', labelsize=14)  # Larger tick labels

    for col_idx, (xmin, xmax) in enumerate(x_limits):
        if xmin == float('inf') or xmax == float('-inf'):
            continue
        xmin -= metric_configs[col_idx]['margin']
        xmax += metric_configs[col_idx]['margin']
        for row_idx in range(len(MODEL_ORDER)):
            ax = axes[row_idx, col_idx]
            if ax.has_data():
                ax.set_xlim(xmin, xmax)

    fig.text(0.02, 0.5, 'Persona Groups', rotation='vertical', va='center', fontsize=18, fontweight='bold')

    column_titles = ['Accuracy', 'Token-F1']
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].text(0.5, 1.12, title, transform=axes[0, col_idx].transAxes,
                              ha='center', fontsize=20, fontweight='bold')  # Larger column titles

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color=baseline_color, linestyle='None', markersize=12,
                   markerfacecolor='white', markeredgecolor=baseline_color, markeredgewidth=3, label='Baseline'),
        plt.Line2D([0], [0], marker='o', color=gender_colors['Male'], linestyle='None', markersize=12,
                   markerfacecolor=gender_colors['Male'], markeredgecolor='black', markeredgewidth=2.5, label='Persona (Male)'),
        plt.Line2D([0], [0], marker='o', color=gender_colors['Female'], linestyle='None', markersize=12,
                   markerfacecolor=gender_colors['Female'], markeredgecolor='black', markeredgewidth=2.5, label='Persona (Female)')
    ]

    legend = axes[0, 1].legend(handles=legend_handles, loc='upper right', fontsize=14, frameon=True,
                               framealpha=0.9, edgecolor='black', fancybox=True)  # Enhanced legend
    legend.get_frame().set_linewidth(1.5)

    fig.suptitle('CoSE Accuracy and Token-F1 by Persona Group', fontsize=22, y=0.96, fontweight='bold')  # Larger, cleaner title

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10, top=0.88, left=0.08, right=0.98, hspace=0.15, wspace=0.12)  # Better spacing
    plt.savefig(PLOTS_DIR / 'cose_accuracy_and_f1_swapped_improved.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_and_f1_transposed():
    """Create transposed plot: models in rows, accuracy and F1 in columns"""
    accuracy_data = load_accuracy_data()
    f1_data = load_rationale_data()

    # Derive persona group order from reference model baseline accuracy (ascending)
    group_baseline_order = get_group_order_by_baseline(accuracy_data)

    # Create figure with 3 rows (models), 2 columns (accuracy, F1)
    fig = plt.figure(figsize=(10, 9))  # Adjusted for new layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    # Color scheme
    colors = {
        'baseline': '#ff7f0e',  # orange
        'Male': '#1f77b4',      # blue
        'Female': '#e377c2'     # pink
    }

    # Plot each model as a row
    for row, model in enumerate(MODEL_ORDER):
        # Format model name
        model_display = MODEL_DISPLAY_NAMES.get(model, model.replace('_', ' ').title())

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
    plt.savefig(PLOTS_DIR / 'cose_accuracy_and_f1_dumbbell_transposed.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created transposed combined accuracy and F1 dumbbell plot: cose_accuracy_and_f1_dumbbell_transposed.png")


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
    ax.set_xlim(62, 78)

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
    ax.set_xlim(0.43, 0.60)

    # Format x-axis to show only decimal places
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: f'{x:.0f}' if x >= 1 else ('0' if x <= 0 else format_decimal(x, decimals=2, strip_leading_zero=True))
        )
    )


def plot_iou_f1_subplot(ax, model_data, group_baseline_order, colors, model_display, male_delta, female_delta, is_bottom_row=False):
    """Plot IOU-F1 subplot for one model"""
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

        # Add baseline CI (if available)
        if pd.notna(baseline_ci) and baseline_ci > 0:
            ax.errorbar(baseline_val, y_pos, xerr=baseline_ci,
                       color=colors['baseline'], alpha=0.8, capsize=4, capthick=1.2, elinewidth=1.2)

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

            # Add persona CI (make it more visible)
            if pd.notna(persona_ci):
                ax.errorbar(persona_val, y_persona, xerr=persona_ci,
                           color=colors[gender], alpha=0.8, capsize=4, capthick=1.2, elinewidth=1.2)

        y_positions[group] = y_pos
        y_pos += 0.5

    # Format subplot
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([g for g in y_positions.keys()], fontsize=11)
    if is_bottom_row:
        ax.set_xlabel('IOU-F1', fontsize=12)
    ax.set_title(f'♂ {male_delta:+.3f}, ♀ {female_delta:+.3f}', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    # Set fixed x-limits for IOU-F1 scores to ensure alignment
    ax.set_xlim(0.28, 0.56)

    # Format x-axis to show only decimal places
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: f'{x:.0f}' if x >= 1 else ('0' if x <= 0 else format_decimal(x, decimals=2, strip_leading_zero=True))
        )
    )


def plot_iou_f1_transposed():
    """Create transposed plot: models in rows, IOU-F1 only"""
    accuracy_data = load_accuracy_data()
    iou_f1_data = load_iou_f1_data()

    # Create figure with 3 rows (models), 1 column (IOU-F1)
    fig = plt.figure(figsize=(6, 9))
    gs = fig.add_gridspec(3, 1, hspace=0.3)

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
        model_display = MODEL_DISPLAY_NAMES.get(model, model.replace('_', ' ').title())

        ax_iou = fig.add_subplot(gs[row, 0])

        model_iou_data = iou_f1_data[iou_f1_data['model'] == model].copy()
        if not model_iou_data.empty:
            # Calculate mean deltas for title annotation
            model_gender_deltas = model_iou_data.groupby('gender')['delta'].mean()
            male_delta = model_gender_deltas.get('Male', 0)
            female_delta = model_gender_deltas.get('Female', 0)

            plot_iou_f1_subplot(ax_iou, model_iou_data, group_baseline_order, colors,
                          model_display, male_delta, female_delta, is_bottom_row=(row == 2))

            # Add column header for first row
            if row == 0:
                ax_iou.text(0.5, 1.15, 'IOU-F1', transform=ax_iou.transAxes,
                          horizontalalignment='center', fontsize=14, fontweight='bold')

        # Add rotated model name on the left side
        if not model_iou_data.empty:
            ax_iou.text(-0.15, 0.5, model_display, transform=ax_iou.transAxes,
                       rotation=90, verticalalignment='center', horizontalalignment='center',
                       fontsize=12, fontweight='bold')

            # Add legend only to first subplot
            if row == 0:
                legend_handles = [
                    plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='none', markeredgecolor=colors['baseline'],
                              markersize=8, markeredgewidth=1.5, label='Baseline'),
                    plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=colors['Male'], markersize=8, label='Male Personas'),
                    plt.Line2D([0], [0], marker='^', color='w',
                              markerfacecolor=colors['Female'], markersize=8, label='Female Personas')
                ]
                ax_iou.legend(handles=legend_handles, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'cose_iou_f1_dumbbell_transposed.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created transposed IOU-F1 dumbbell plot: cose_iou_f1_dumbbell_transposed.png")


def main():
    """Generate CoSE accuracy, Token-F1, and IOU-F1 visualizations."""
    print("Creating CoSE accuracy, Token-F1, and IOU-F1 visualizations...")

    PLOTS_DIR.mkdir(exist_ok=True)

    print("- Building swapped-axis accuracy/F1 scatter...")
    plot_cose_accuracy_and_f1_swapped()

    print("- Building transposed dumbbell accuracy/F1 plot...")
    plot_accuracy_and_f1_transposed()

    print("- Building transposed dumbbell IOU-F1 plot...")
    plot_iou_f1_transposed()

    print(f"\nPlots saved to: {PLOTS_DIR}")
    print("✅ CoSE accuracy, Token-F1, and IOU-F1 visualizations completed!")


if __name__ == "__main__":
    main()
