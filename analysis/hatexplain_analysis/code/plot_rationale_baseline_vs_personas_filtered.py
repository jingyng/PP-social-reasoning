import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
CSV_DIR = THIS_DIR / "csv"
PLOTS_DIR = THIS_DIR / "plots"

# Consistent mapping for model identifiers to display names
MODEL_DISPLAY_NAMES = {
    'gpt_oss_120b': 'GPT-OSS-120B',
    'mistral_medium': 'Mistral-Medium',
    'qwen3_32b': 'Qwen3-32B'
}


def get_persona_groups():
    """Define persona groups by attribute category with colors matching accuracy plots"""
    persona_groups = {
        'Age': {
            'personas': ['15', '35', '65'],
            'color': '#1f77b4'  # blue
        },
        'Education': {
            'personas': ['nfe', 'hs', 'he'],
            'color': '#ff7f0e'  # orange
        },
        'Gender': {
            'personas': ['m', 'f'],
            'color': '#2ca02c'  # green
        },
        'Loneliness': {
            'personas': ['nl', 'sl'],
            'color': '#d62728'  # red
        },
        'Political': {
            'personas': ['l', 'r', 'c'],
            'color': '#9467bd'  # purple
        },
        'Race': {
            'personas': ['w', 'b', 'a'],
            'color': '#8c564b'  # brown
        },
        'Religion': {
            'personas': ['chr', 'mus', 'jew', 'ath', 'hin'],
            'color': '#e377c2'  # pink
        }
    }
    return persona_groups


def get_persona_label(persona):
    """Convert persona codes to readable labels"""
    label_mapping = {
        '15': '15 years', '35': '35 years', '65': '65 years',
        'nfe': 'No Formal Education', 'hs': 'High School Education', 'he': 'Higher Education',
        'm': 'Male', 'f': 'Female',
        'nl': 'Not Lonely', 'sl': 'Somewhat Lonely',
        'l': 'Left-wing', 'r': 'Right-wing', 'c': 'Centrist',
        'w': 'White', 'b': 'Black', 'a': 'Asian',
        'chr': 'Christian', 'mus': 'Muslim', 'jew': 'Jewish', 'ath': 'Atheist', 'hin': 'Hindu'
    }
    return label_mapping.get(persona, persona)


def load_filtered_rationale_data():
    """Load filtered rationale performance data"""
    persona_file = CSV_DIR / "persona_rationale_performance_averaged_filtered.csv"
    baseline_file = CSV_DIR / "baseline_rationale_performance_averaged_filtered.csv"

    persona_df = pd.read_csv(persona_file)
    baseline_df = pd.read_csv(baseline_file)

    return persona_df, baseline_df


def stripping_format(value: float) -> str:
    """Format decimals to two places with leading zero stripped (-> .23)."""
    return f"{value:.2f}".lstrip('0')


def plot_baseline_vs_personas_token_f1_filtered():
    """Baseline vs persona token-f1 rationale performance (filtered), using swapped axes."""
    persona_df, baseline_df = load_filtered_rationale_data()

    persona_groups = get_persona_groups()
    models = list(persona_df['model'].unique())

    # Build ordered personas grouped by attribute
    ordered_personas, persona_colors, persona_labels = [], [], []
    group_positions = {}
    current_pos = 0
    for group_name, group_info in persona_groups.items():
        group_start = current_pos
        for persona in group_info['personas']:
            ordered_personas.append(persona)
            persona_colors.append(group_info['color'])
            persona_labels.append(get_persona_label(persona))
            current_pos += 1
        group_positions[group_name] = (group_start, current_pos - 1, group_info['color'])

    # Reverse order for top-to-bottom reading
    ordered_personas = ordered_personas[::-1]
    persona_colors = persona_colors[::-1]
    persona_labels = persona_labels[::-1]

    total_personas = len(ordered_personas)
    reversed_group_positions = {}
    for group_name, (start_idx, end_idx, color) in group_positions.items():
        new_start = total_personas - 1 - end_idx
        new_end = total_personas - 1 - start_idx
        reversed_group_positions[group_name] = (new_start, new_end, color)
    group_positions = reversed_group_positions

    y_spacing = 1.2
    y_ticks = [j * y_spacing for j in range(total_personas)]
    y_min = -0.6
    y_max = (total_personas - 1) * y_spacing + 0.6

    fig, axes = plt.subplots(1, len(models), figsize=(14, 6), sharey=True)
    if len(models) == 1:
        axes = [axes]

    percent_formatter = FuncFormatter(lambda x, _: stripping_format(x))

    global_min = float('inf')
    global_max = float('-inf')

    legend_positions = ['upper left', 'upper right', 'upper right']

    for i, model in enumerate(models):
        ax = axes[i]
        model_persona = persona_df[persona_df['model'] == model]
        model_baseline = baseline_df[baseline_df['model'] == model]
        model_display = MODEL_DISPLAY_NAMES.get(model, model)

        baseline_token_f1 = model_baseline['token_f1_mean'].iloc[0]
        baseline_std = model_baseline['token_f1_std'].iloc[0]

        # Baseline vertical line with uncertainty band
        ax.axvline(baseline_token_f1, color='black', linestyle='--', linewidth=2,
                   label=f'Baseline ({stripping_format(baseline_token_f1)}±{stripping_format(baseline_std)})')
        ax.fill_betweenx([y_min, y_max], baseline_token_f1 - baseline_std, baseline_token_f1 + baseline_std,
                         color='black', alpha=0.1)

        global_min = min(global_min, baseline_token_f1 - baseline_std)
        global_max = max(global_max, baseline_token_f1 + baseline_std)

        ordered_values = []
        ordered_stds = []
        for persona in ordered_personas:
            row = model_persona[model_persona['persona'] == persona]
            if row.empty:
                ordered_values.append(np.nan)
                ordered_stds.append(0.0)
            else:
                ordered_values.append(row['token_f1_mean'].iloc[0])
                ordered_stds.append(row['token_f1_std'].iloc[0])

        for j, (value, std, color) in enumerate(zip(ordered_values, ordered_stds, persona_colors)):
            if np.isnan(value):
                continue
            y_pos = j * y_spacing
            ax.errorbar(value, y_pos, xerr=std, fmt='o', color=color, alpha=0.85,
                        capsize=3, capthick=1, markersize=6, markeredgecolor='black', markeredgewidth=0.4)
            global_min = min(global_min, value - std)
            global_max = max(global_max, value + std)

        ax.set_yticks(y_ticks)
        if i == 0:
            ax.set_yticklabels(persona_labels, fontsize=11)
        else:
            ax.tick_params(axis='y', labelleft=False)

        for _, (start_idx, end_idx, group_color) in group_positions.items():
            y_start = start_idx * y_spacing - 0.4
            y_end = (end_idx + 1) * y_spacing - 0.8
            ax.axhspan(y_start, y_end, color=group_color, alpha=0.08, zorder=0)

        for start_idx, _, _ in group_positions.values():
            if start_idx > 0:
                ax.axhline((start_idx - 0.5) * y_spacing, color='white', linestyle='-', alpha=0.8, linewidth=1)

        ax.set_title(model_display, fontsize=15)
        ax.set_xlabel('Token-F1', fontsize=13)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_formatter(percent_formatter)

        legend_loc = legend_positions[i] if i < len(legend_positions) else 'upper right'
        ax.legend(loc=legend_loc, fontsize=11)

    margin = 0.01
    global_min = global_min if global_min != float('inf') else 0.0
    global_max = global_max if global_max != float('-inf') else 1.0
    for ax in axes:
        ax.set_xlim(global_min - margin, global_max + margin)

    group_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=info['color'], markersize=6,
                   label=name, markeredgecolor='black')
        for name, info in persona_groups.items()
    ]

    first_ax = axes[0]
    baseline_legend = first_ax.get_legend()
    if baseline_legend:
        first_ax.add_artist(baseline_legend)
    first_ax.legend(handles=group_handles, loc='lower left', fontsize=10, frameon=True)

    # fig.text(0.04, 0.5, 'Personas by Attribute Groups', rotation='vertical', va='center', fontsize=13)

    # fig.suptitle('Token F1 Rationale Performance: Personas vs Baseline\n(Offensive Language + Hate Speech Only)',
                #  fontsize=16, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.9)
    plt.savefig(PLOTS_DIR / 'rationale_baseline_vs_personas_token_f1_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_baseline_vs_personas_iou_f1_filtered():
    """Baseline vs persona IOU-F1 rationale performance (filtered), using swapped axes."""
    persona_df, baseline_df = load_filtered_rationale_data()

    persona_groups = get_persona_groups()
    models = list(persona_df['model'].unique())

    # Build ordered personas grouped by attribute
    ordered_personas, persona_colors, persona_labels = [], [], []
    group_positions = {}
    current_pos = 0
    for group_name, group_info in persona_groups.items():
        group_start = current_pos
        for persona in group_info['personas']:
            ordered_personas.append(persona)
            persona_colors.append(group_info['color'])
            persona_labels.append(get_persona_label(persona))
            current_pos += 1
        group_positions[group_name] = (group_start, current_pos - 1, group_info['color'])

    # Reverse order for top-to-bottom reading
    ordered_personas = ordered_personas[::-1]
    persona_colors = persona_colors[::-1]
    persona_labels = persona_labels[::-1]

    total_personas = len(ordered_personas)
    reversed_group_positions = {}
    for group_name, (start_idx, end_idx, color) in group_positions.items():
        new_start = total_personas - 1 - end_idx
        new_end = total_personas - 1 - start_idx
        reversed_group_positions[group_name] = (new_start, new_end, color)
    group_positions = reversed_group_positions

    y_spacing = 1.2
    y_ticks = [j * y_spacing for j in range(total_personas)]
    y_min = -0.6
    y_max = (total_personas - 1) * y_spacing + 0.6

    fig, axes = plt.subplots(1, len(models), figsize=(14, 6), sharey=True)
    if len(models) == 1:
        axes = [axes]

    percent_formatter = FuncFormatter(lambda x, _: stripping_format(x))

    global_min = float('inf')
    global_max = float('-inf')

    legend_positions = ['upper left', 'upper right', 'upper right']

    for i, model in enumerate(models):
        ax = axes[i]
        model_persona = persona_df[persona_df['model'] == model]
        model_baseline = baseline_df[baseline_df['model'] == model]
        model_display = MODEL_DISPLAY_NAMES.get(model, model)

        baseline_iou_f1 = model_baseline['iou_f1_mean'].iloc[0]
        baseline_std = model_baseline['iou_f1_std'].iloc[0]

        # Baseline vertical line with uncertainty band
        ax.axvline(baseline_iou_f1, color='black', linestyle='--', linewidth=2,
                   label=f'Baseline ({stripping_format(baseline_iou_f1)}±{stripping_format(baseline_std)})')
        ax.fill_betweenx([y_min, y_max], baseline_iou_f1 - baseline_std, baseline_iou_f1 + baseline_std,
                         color='black', alpha=0.1)

        global_min = min(global_min, baseline_iou_f1 - baseline_std)
        global_max = max(global_max, baseline_iou_f1 + baseline_std)

        ordered_values = []
        ordered_stds = []
        for persona in ordered_personas:
            row = model_persona[model_persona['persona'] == persona]
            if row.empty:
                ordered_values.append(np.nan)
                ordered_stds.append(0.0)
            else:
                ordered_values.append(row['iou_f1_mean'].iloc[0])
                ordered_stds.append(row['iou_f1_std'].iloc[0])

        for j, (value, std, color) in enumerate(zip(ordered_values, ordered_stds, persona_colors)):
            if np.isnan(value):
                continue
            y_pos = j * y_spacing
            ax.errorbar(value, y_pos, xerr=std, fmt='o', color=color, alpha=0.85,
                        capsize=3, capthick=1, markersize=6, markeredgecolor='black', markeredgewidth=0.4)
            global_min = min(global_min, value - std)
            global_max = max(global_max, value + std)

        ax.set_yticks(y_ticks)
        if i == 0:
            ax.set_yticklabels(persona_labels, fontsize=11)
        else:
            ax.tick_params(axis='y', labelleft=False)

        for _, (start_idx, end_idx, group_color) in group_positions.items():
            y_start = start_idx * y_spacing - 0.4
            y_end = (end_idx + 1) * y_spacing - 0.8
            ax.axhspan(y_start, y_end, color=group_color, alpha=0.08, zorder=0)

        for start_idx, _, _ in group_positions.values():
            if start_idx > 0:
                ax.axhline((start_idx - 0.5) * y_spacing, color='white', linestyle='-', alpha=0.8, linewidth=1)

        ax.set_title(model_display, fontsize=15)
        ax.set_xlabel('IOU-F1', fontsize=13)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_formatter(percent_formatter)

        legend_loc = legend_positions[i] if i < len(legend_positions) else 'upper right'
        ax.legend(loc=legend_loc, fontsize=11)

    margin = 0.01
    global_min = global_min if global_min != float('inf') else 0.0
    global_max = global_max if global_max != float('-inf') else 1.0
    for ax in axes:
        ax.set_xlim(global_min - margin, global_max + margin)

    group_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=info['color'], markersize=6,
                   label=name, markeredgecolor='black')
        for name, info in persona_groups.items()
    ]

    first_ax = axes[0]
    baseline_legend = first_ax.get_legend()
    if baseline_legend:
        first_ax.add_artist(baseline_legend)
    first_ax.legend(handles=group_handles, loc='lower left', fontsize=10, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.9)
    plt.savefig(PLOTS_DIR / 'rationale_baseline_vs_personas_iou_f1_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_filtered_vs_all():
    """Plot comparison between filtered and all-data results"""
    # Load both datasets
    persona_filtered, baseline_filtered = load_filtered_rationale_data()

    # Load original (all data) results
    persona_all = pd.read_csv(CSV_DIR / "persona_rationale_performance_averaged.csv")
    baseline_all = pd.read_csv(CSV_DIR / "baseline_rationale_performance_averaged.csv")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    models = persona_filtered['model'].unique()

    for i, model in enumerate(models):
        ax = axes[i]

        # Get baseline data
        baseline_filtered_model = baseline_filtered[baseline_filtered['model'] == model]['token_f1_mean'].iloc[0]
        baseline_all_model = baseline_all[baseline_all['model'] == model]['token_f1_mean'].iloc[0]

        # Get persona averages
        persona_filtered_model = persona_filtered[persona_filtered['model'] == model]['token_f1_mean'].mean()
        persona_all_model = persona_all[persona_all['model'] == model]['token_f1_mean'].mean()

        # Create bar plot
        categories = ['All Data\n(500 samples)', 'Filtered Data\n(Offensive+Hate only)\n(199 samples)']
        baseline_values = [baseline_all_model, baseline_filtered_model]
        persona_values = [persona_all_model, persona_filtered_model]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8, color='lightcoral')
        bars2 = ax.bar(x + width/2, persona_values, width, label='Personas (avg)', alpha=0.8, color='skyblue')

        # Add value labels
        for bar, val in zip(bars1, baseline_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        for bar, val in zip(bars2, persona_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        ax.set_title(MODEL_DISPLAY_NAMES.get(model, model), fontsize=14)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Token-F1', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 0.8)

    plt.suptitle('Rationale Performance Comparison: All Data vs Filtered Data (Offensive+Hate only)',
                fontsize=16, y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(PLOTS_DIR / 'rationale_performance_comparison_filtered_vs_all.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("Creating filtered rationale performance plots...")

    # Create plots directory
    PLOTS_DIR.mkdir(exist_ok=True)

    # Create the baseline vs personas plot for filtered data (Token-F1)
    print("Creating baseline vs personas Token-F1 plot for filtered data...")
    plot_baseline_vs_personas_token_f1_filtered()

    # Create the baseline vs personas plot for filtered data (IOU-F1)
    print("Creating baseline vs personas IOU-F1 plot for filtered data...")
    plot_baseline_vs_personas_iou_f1_filtered()

    # Create comparison plot between filtered and all data
    print("Creating comparison plot: filtered vs all data...")
    plot_comparison_filtered_vs_all()

    print(f"Plots saved to: {PLOTS_DIR}")
    print("- rationale_baseline_vs_personas_token_f1_filtered.png")
    print("- rationale_baseline_vs_personas_iou_f1_filtered.png")
    print("- rationale_performance_comparison_filtered_vs_all.png")
    print("\nFiltered rationale performance plots completed!")


if __name__ == "__main__":
    main()
