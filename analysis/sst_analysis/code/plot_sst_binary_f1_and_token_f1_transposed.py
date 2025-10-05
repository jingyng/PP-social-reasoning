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
    """Extract gender from persona code (e.g., '25_m_b' -> 'Male')"""
    parts = persona_code.split('_')
    if len(parts) >= 2:
        return 'Male' if parts[1] == 'm' else 'Female'
    return 'Unknown'


def persona_code_matches_group(persona_code, group):
    """Check if persona code matches the demographic group"""
    parts = persona_code.split('_')
    if len(parts) < 3:
        return False
    age = parts[0]  # '25' or '45'
    race = parts[2]  # 'b', 'l', or 'w'

    # Map group codes to age and race
    group_age = '45' if group.endswith('O') else '25'
    group_race = {'B': 'b', 'L': 'l', 'W': 'w'}.get(group[0], '')

    return age == group_age and race == group_race


def load_binary_f1_data():
    """Load baseline and persona binary F1 data"""
    baseline_file = CSV_DIR / "baseline_accuracy_sst_per_run.csv"
    persona_file = CSV_DIR / "persona_accuracy_sst_per_run.csv"

    baseline_df = pd.read_csv(baseline_file)
    persona_df = pd.read_csv(persona_file)

    # Filter for matched personas only
    persona_df = persona_df[persona_df['matches_group'] == True]

    # Add gender information to persona data
    persona_df['gender'] = persona_df['persona_text'].apply(extract_gender_from_persona)

    # Compute baseline statistics (mean and CI across runs) using binary F1
    baseline_stats = baseline_df.groupby(['model', 'group']).agg({
        'macro_f1_binary': ['mean', 'std', 'count']
    }).reset_index()
    baseline_stats.columns = ['model', 'group', 'baseline_mean', 'baseline_std', 'baseline_count']

    # Compute 95% CI for baseline
    baseline_stats['baseline_ci'] = 1.96 * baseline_stats['baseline_std'] / np.sqrt(baseline_stats['baseline_count'])

    # Compute persona statistics by gender (mean and CI across runs) using binary F1
    persona_stats = persona_df.groupby(['model', 'group', 'gender']).agg({
        'macro_f1_binary': ['mean', 'std', 'count']
    }).reset_index()
    persona_stats.columns = ['model', 'group', 'gender', 'persona_mean', 'persona_std', 'persona_count']

    # Compute 95% CI for personas
    persona_stats['persona_ci'] = 1.96 * persona_stats['persona_std'] / np.sqrt(persona_stats['persona_count'])

    # Merge baseline and persona stats
    combined_stats = pd.merge(baseline_stats, persona_stats, on=['model', 'group'], how='outer')

    # Compute delta (persona - baseline)
    combined_stats['delta'] = combined_stats['persona_mean'] - combined_stats['baseline_mean']

    # Convert F1 values to percentages
    combined_stats['baseline_mean'] *= 100
    combined_stats['baseline_ci'] *= 100
    combined_stats['persona_mean'] *= 100
    combined_stats['persona_ci'] *= 100
    combined_stats['delta'] *= 100

    return combined_stats


def load_rationale_data():
    """Load baseline and persona Token-F1 data"""
    baseline_file = CSV_DIR / "baseline_token_iou_f1_sst_per_run.csv"
    persona_file = CSV_DIR / "persona_token_iou_f1_sst_per_run.csv"

    baseline_df = pd.read_csv(baseline_file)
    persona_df = pd.read_csv(persona_file)

    # Add gender information to persona data
    persona_df['gender'] = persona_df['persona_code'].apply(extract_gender_from_persona_code)

    # Add matching status
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
    baseline_file = CSV_DIR / "baseline_token_iou_f1_sst_per_run.csv"
    persona_file = CSV_DIR / "persona_token_iou_f1_sst_per_run.csv"

    baseline_df = pd.read_csv(baseline_file)
    persona_df = pd.read_csv(persona_file)

    # Add gender information to persona data
    persona_df['gender'] = persona_df['persona_code'].apply(extract_gender_from_persona_code)

    # Add matching status
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


def format_decimal(value: float, decimals: int = 2, strip_leading_zero: bool = False) -> str:
    """Format a float with configurable decimals and optional leading zero removal."""
    formatted = f"{value:.{decimals}f}"
    if strip_leading_zero:
        if formatted.startswith('-0'):
            formatted = '-' + formatted[2:]
        elif formatted.startswith('0'):
            formatted = formatted[1:]
    return formatted


def get_group_order_by_baseline(binary_f1_df, reference_model="gpt_oss_120b"):
    """Order persona groups by baseline binary F1 for a reference model (ascending)."""
    model_data = binary_f1_df[binary_f1_df['model'] == reference_model]
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


def plot_binary_f1_and_token_f1_transposed():
    """Create transposed plot: models in rows, binary F1 and token F1 in columns"""
    binary_f1_data = load_binary_f1_data()
    token_f1_data = load_rationale_data()

    # Create figure with 3 rows (models), 2 columns (binary F1, token F1)
    fig = plt.figure(figsize=(10, 9))  # Adjusted for new layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    # Order groups by baseline binary F1 of reference model
    group_baseline_order = get_group_order_by_baseline(binary_f1_data)

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

        # Left column: Binary F1
        ax_bf1 = fig.add_subplot(gs[row, 0])

        model_bf1_data = binary_f1_data[binary_f1_data['model'] == model].copy()
        if not model_bf1_data.empty:
            # Calculate mean deltas for title annotation
            model_gender_deltas = model_bf1_data.groupby('gender')['delta'].mean()
            male_delta = model_gender_deltas.get('Male', 0)
            female_delta = model_gender_deltas.get('Female', 0)

            plot_binary_f1_subplot(ax_bf1, model_bf1_data, group_baseline_order, colors,
                                model_display, male_delta, female_delta, show_legend=(row == 0), is_bottom_row=(row == 2))

            # Add column header for first row
            if row == 0:
                ax_bf1.text(0.5, 1.15, 'Binary Macro F1', transform=ax_bf1.transAxes,
                           horizontalalignment='center', fontsize=14, fontweight='bold')

        # Add rotated model name on the left side
        if not model_bf1_data.empty:
            ax_bf1.text(-0.15, 0.5, model_display, transform=ax_bf1.transAxes,
                       rotation=90, verticalalignment='center', horizontalalignment='center',
                       fontsize=12, fontweight='bold')

        # Right column: Token F1
        ax_tf1 = fig.add_subplot(gs[row, 1])

        model_tf1_data = token_f1_data[token_f1_data['model'] == model].copy()
        if not model_tf1_data.empty:
            # Calculate mean deltas for title annotation
            model_gender_deltas = model_tf1_data.groupby('gender')['delta'].mean()
            male_delta = model_gender_deltas.get('Male', 0)
            female_delta = model_gender_deltas.get('Female', 0)

            plot_token_f1_subplot(ax_tf1, model_tf1_data, group_baseline_order, colors,
                          model_display, male_delta, female_delta, is_bottom_row=(row == 2))

            # Add column header for first row
            if row == 0:
                ax_tf1.text(0.5, 1.15, 'Token-F1', transform=ax_tf1.transAxes,
                          horizontalalignment='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'sst_binary_f1_and_token_f1_dumbbell_transposed.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created transposed binary F1 and token F1 dumbbell plot: sst_binary_f1_and_token_f1_dumbbell_transposed.png")


def plot_binary_f1_subplot(ax, model_data, group_baseline_order, colors, model_display, male_delta, female_delta, show_legend=False, is_bottom_row=False):
    """Plot binary F1 subplot for one model"""
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
        ax.set_xlabel('Binary Macro F1 (%)', fontsize=12)
    ax.set_title(f'♂ {male_delta:+.1f}%, ♀ {female_delta:+.1f}%', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(80, 100)  # Adjusted range for F1 percentages

    # Add legend only to first subplot
    if show_legend:
        ax.legend(loc='upper left', fontsize=10)


def plot_token_f1_subplot(ax, model_data, group_baseline_order, colors, model_display, male_delta, female_delta, is_bottom_row=False):
    """Plot Token F1 subplot for one model"""
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
    ax.set_xlim(0.45, 0.68)

    # Format x-axis to show only decimal places
    def format_decimal(x, pos):
        if x >= 1:
            return f'{x:.0f}'
        elif x <= 0:
            return '0'
        else:
            return f'{x:.2f}'[1:]

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))


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
    ax.set_xlim(0.33, 0.60)

    # Format x-axis to show only decimal places
    def format_decimal(x, pos):
        if x >= 1:
            return f'{x:.0f}'
        elif x <= 0:
            return '0'
        else:
            return f'{x:.2f}'[1:]

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_decimal))


def plot_iou_f1_transposed():
    """Create transposed plot: models in rows, IOU-F1 only"""
    binary_f1_data = load_binary_f1_data()
    iou_f1_data = load_iou_f1_data()

    # Create figure with 3 rows (models), 1 column (IOU-F1)
    fig = plt.figure(figsize=(6, 9))
    gs = fig.add_gridspec(3, 1, hspace=0.3)

    # Order groups by baseline binary F1 of reference model
    group_baseline_order = get_group_order_by_baseline(binary_f1_data)

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
    plt.savefig(PLOTS_DIR / 'sst_iou_f1_dumbbell_transposed.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created transposed IOU-F1 dumbbell plot: sst_iou_f1_dumbbell_transposed.png")


def main():
    """Generate SST binary F1/Token F1/IOU-F1 visualizations."""
    print("Creating SST Binary F1, Token-F1, and IOU-F1 visualizations...")

    PLOTS_DIR.mkdir(exist_ok=True)

    print("- Building transposed dumbbell Binary F1/Token F1 plot...")
    plot_binary_f1_and_token_f1_transposed()

    print("- Building transposed dumbbell IOU-F1 plot...")
    plot_iou_f1_transposed()

    print(f"\nPlots saved to: {PLOTS_DIR}")
    print("✅ SST Binary F1, Token-F1, and IOU-F1 visualizations completed!")


if __name__ == "__main__":
    main()