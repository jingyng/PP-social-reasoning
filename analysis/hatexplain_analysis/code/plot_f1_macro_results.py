import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# Get the absolute path to the script's directory
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
CSV_DIR = THIS_DIR / 'csv'
PLOTS_DIR = THIS_DIR / 'plots'

def load_results():
    """Load the averaged accuracy and F1 results"""
    return pd.read_csv(CSV_DIR / 'accuracy_f1_me_mae_flagging_rates_results_averaged.csv')

def get_persona_groups():
    """Define persona groups by attribute category with colors"""
    persona_groups = {
        'Age': {
            'personas': ['15', '35', '65'],
            'color': '#1f77b4'  # blue
        },
        'Education': {
            'personas': ['no_formal_education', 'high_school_education', 'higher_education'],
            'color': '#ff7f0e'  # orange
        },
        'Gender': {
            'personas': ['male', 'female'],
            'color': '#2ca02c'  # green
        },
        'Loneliness': {
            'personas': ['not_lonely', 'somewhat_lonely'],
            'color': '#d62728'  # red
        },
        'Political': {
            'personas': ['left-wing', 'right-wing', 'centrist'],
            'color': '#9467bd'  # purple
        },
        'Race': {
            'personas': ['white', 'black', 'asian'],
            'color': '#8c564b'  # brown
        },
        'Religion': {
            'personas': ['christian', 'muslim', 'jewish', 'atheist', 'hindu'],
            'color': '#e377c2'  # pink
        }
    }
    return persona_groups

def plot_baseline_vs_personas_f1_macro(df, out_basename="baseline_vs_personas_f1_macro"):
    """
    Plot baseline vs persona F1 macro scores:
      - Shared y-axis across subplots
      - Persona names on the LEFT y-axis (only on left panel)
      - Vertical baseline line with per-panel value annotation
      - Compact layout for LaTeX
    """

    persona_groups = get_persona_groups()
    models = list(df['model'].unique())

    # ----- Build ordered personas and groups (reversed for top-to-bottom reading) -----
    ordered_personas, persona_colors, persona_labels = [], [], []
    group_positions = {}  # name -> (start_idx, end_idx, color)
    current_pos = 0
    for group_name, group_info in persona_groups.items():
        group_start = current_pos
        for persona in group_info['personas']:
            ordered_personas.append(persona)
            persona_colors.append(group_info['color'])
            label = persona.replace('_', ' ').replace('-', '-').title()
            if persona in ['15', '35', '65']:
                label = f"{persona} years"
            persona_labels.append(label)
            current_pos += 1
        group_positions[group_name] = (group_start, current_pos - 1, group_info['color'])

    # Reverse the order so first group appears at top of y-axis
    ordered_personas = ordered_personas[::-1]
    persona_colors = persona_colors[::-1]
    persona_labels = persona_labels[::-1]

    # Update group positions for reversed order
    total_personas = len(ordered_personas)
    reversed_group_positions = {}
    for group_name, (start_idx, end_idx, color) in group_positions.items():
        new_start = total_personas - 1 - end_idx
        new_end = total_personas - 1 - start_idx
        reversed_group_positions[group_name] = (new_start, new_end, color)
    group_positions = reversed_group_positions

    # ----- Layout / ticks -----
    y_spacing = 1.2
    y_ticks = [j * y_spacing for j in range(len(ordered_personas))]
    y_min = -0.6
    y_max = (len(ordered_personas) - 1) * y_spacing + 0.6

    # Model name mapping for better readability
    model_display_names = {
        'gpt_oss_120b': 'GPT-OSS-120B',
        'mistral_medium': 'Mistral-Medium',
        'qwen3_32b': 'Qwen3-32B'
    }

    # Shared y-axis across three panels - more compact
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    percent_formatter = FuncFormatter(lambda x, _: f"{x * 100:.1f}")

    for i, model in enumerate(models):
        model_display = model_display_names.get(model, model)
        ax = axes[i]
        model_data = df[df['model'] == model]
        baseline_data = model_data[model_data['persona_type'] == 'baseline']
        baseline_f1 = baseline_data['f1_macro_mean'].iloc[0]
        baseline_std = baseline_data['f1_macro_std'].iloc[0]

        persona_data = model_data[model_data['persona_type'] == 'persona']

        # --- Baseline: vertical line with error band (swapped coordinates) ---
        ax.axvline(x=baseline_f1, color='black', linestyle='--', linewidth=2,
                  label=f'Baseline ({baseline_f1 * 100:.1f}±{baseline_std * 100:.1f})')
        ax.fill_betweenx([y_min, y_max],  # y range matching ylim
                        baseline_f1 - baseline_std, baseline_f1 + baseline_std,
                        color='black', alpha=0.1)

        # Persona points with error bars
        for j, persona in enumerate(ordered_personas):
            row = persona_data[persona_data['persona_id'] == persona]
            if row.empty:
                continue
            f1 = row['f1_macro_mean'].iloc[0]
            std = row['f1_macro_std'].iloc[0]
            y_pos = j * y_spacing
            ax.errorbar(f1, y_pos, xerr=std, fmt='o',
                        color=persona_colors[j], alpha=0.9,
                        capsize=3, capthick=1, markersize=7,
                        markeredgecolor='black', markeredgewidth=0.5)

        # Y ticks shared; only left panel shows labels
        ax.set_yticks(y_ticks)
        if i == 0:
            ax.set_yticklabels(persona_labels, fontsize=12)
        else:
            ax.tick_params(axis='y', labelleft=False)

        # Colored background bands for attribute groups (all subplots for consistency)
        for group_name, (start_idx, end_idx, group_color) in group_positions.items():
            y_start = start_idx * y_spacing - 0.4
            y_end = (end_idx + 1) * y_spacing - 0.8
            ax.axhspan(y_start, y_end, color=group_color, alpha=0.08, zorder=0)

        # Group separators (thin lines between groups)
        for start_idx, _, _ in group_positions.values():
            if start_idx > 0:
                ax.axhline((start_idx - 0.5) * y_spacing, color='white', linestyle='-', alpha=0.8, linewidth=1)

        # Formatting
        ax.set_title(f'{model_display}', fontsize=16)
        ax.set_xlabel('Macro-F1 Score', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0.30, 0.55)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_formatter(percent_formatter)

        # Add baseline legend with different position for each panel
        legend_positions = ['upper left', 'upper right', 'upper right']
        ax.legend(loc=legend_positions[i], fontsize=12)

    # Add attribute group legend only to the first panel (lower right)
    group_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=info['color'], markersize=7,
                   label=name, markeredgecolor='black')
        for name, info in persona_groups.items()
    ]
    # Preserve the first panel's baseline legend and add attribute legend in different position
    first_ax = axes[0]
    baseline_legend = first_ax.get_legend()
    if baseline_legend:
        first_ax.add_artist(baseline_legend)
    first_ax.legend(handles=group_handles, loc='lower left', fontsize=11, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.92)

    # Save (PNG + PDF for LaTeX)
    png_path = PLOTS_DIR / f'{out_basename}.png'
    pdf_path = PLOTS_DIR / f'{out_basename}.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.01)
    plt.close()

    print(f"F1 Macro plot saved to {png_path}")

def plot_baseline_vs_personas_me(df, out_basename="baseline_vs_personas_mean_error"):
    """
    Plot baseline vs persona Mean Error scores:
      - Shared y-axis across subplots
      - Persona names on the LEFT y-axis (only on left panel)
      - Vertical baseline line with per-panel value annotation
      - Compact layout for LaTeX
    """

    persona_groups = get_persona_groups()
    models = list(df['model'].unique())

    # ----- Build ordered personas and groups (reversed for top-to-bottom reading) -----
    ordered_personas, persona_colors, persona_labels = [], [], []
    group_positions = {}  # name -> (start_idx, end_idx, color)
    current_pos = 0
    for group_name, group_info in persona_groups.items():
        group_start = current_pos
        for persona in group_info['personas']:
            ordered_personas.append(persona)
            persona_colors.append(group_info['color'])
            label = persona.replace('_', ' ').replace('-', '-').title()
            if persona in ['15', '35', '65']:
                label = f"{persona} years"
            persona_labels.append(label)
            current_pos += 1
        group_positions[group_name] = (group_start, current_pos - 1, group_info['color'])

    # Reverse the order so first group appears at top of y-axis
    ordered_personas = ordered_personas[::-1]
    persona_colors = persona_colors[::-1]
    persona_labels = persona_labels[::-1]

    # Update group positions for reversed order
    total_personas = len(ordered_personas)
    reversed_group_positions = {}
    for group_name, (start_idx, end_idx, color) in group_positions.items():
        new_start = total_personas - 1 - end_idx
        new_end = total_personas - 1 - start_idx
        reversed_group_positions[group_name] = (new_start, new_end, color)
    group_positions = reversed_group_positions

    # ----- Layout / ticks -----
    y_spacing = 1.2
    y_ticks = [j * y_spacing for j in range(len(ordered_personas))]
    y_min = -0.6
    y_max = (len(ordered_personas) - 1) * y_spacing + 0.6

    # Model name mapping for better readability
    model_display_names = {
        'gpt_oss_120b': 'GPT-OSS-120B',
        'mistral_medium': 'Mistral-Medium',
        'qwen3_32b': 'Qwen3-32B'
    }

    # Shared y-axis across three panels - more compact
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    percent_formatter = FuncFormatter(lambda x, _: f"{x * 100:.1f}")

    for i, model in enumerate(models):
        model_display = model_display_names.get(model, model)
        ax = axes[i]
        model_data = df[df['model'] == model]
        baseline_data = model_data[model_data['persona_type'] == 'baseline']
        baseline_f1 = baseline_data['mean_error_mean'].iloc[0]
        baseline_std = baseline_data['mean_error_std'].iloc[0]

        persona_data = model_data[model_data['persona_type'] == 'persona']

        # --- Baseline: vertical line with error band (swapped coordinates) ---
        ax.axvline(x=baseline_f1, color='black', linestyle='--', linewidth=2,
                  label=f'Baseline ({baseline_f1 * 100:.1f}±{baseline_std * 100:.1f})')
        ax.fill_betweenx([y_min, y_max],  # y range matching ylim
                        baseline_f1 - baseline_std, baseline_f1 + baseline_std,
                        color='black', alpha=0.1)

        # Persona points with error bars
        for j, persona in enumerate(ordered_personas):
            row = persona_data[persona_data['persona_id'] == persona]
            if row.empty:
                continue
            f1 = row['mean_error_mean'].iloc[0]
            std = row['mean_error_std'].iloc[0]
            y_pos = j * y_spacing
            ax.errorbar(f1, y_pos, xerr=std, fmt='o',
                        color=persona_colors[j], alpha=0.9,
                        capsize=3, capthick=1, markersize=7,
                        markeredgecolor='black', markeredgewidth=0.5)

        # Y ticks shared; only left panel shows labels
        ax.set_yticks(y_ticks)
        if i == 0:
            ax.set_yticklabels(persona_labels, fontsize=12)
        else:
            ax.tick_params(axis='y', labelleft=False)

        # Colored background bands for attribute groups (all subplots for consistency)
        for group_name, (start_idx, end_idx, group_color) in group_positions.items():
            y_start = start_idx * y_spacing - 0.4
            y_end = (end_idx + 1) * y_spacing - 0.8
            ax.axhspan(y_start, y_end, color=group_color, alpha=0.08, zorder=0)

        # Group separators (thin lines between groups)
        for start_idx, _, _ in group_positions.values():
            if start_idx > 0:
                ax.axhline((start_idx - 0.5) * y_spacing, color='white', linestyle='-', alpha=0.8, linewidth=1)

        # Formatting
        ax.set_title(f'{model_display}', fontsize=16)
        ax.set_xlabel('Mean Error Score', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        # ax.set_xlim(0.24, 0.60)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_formatter(percent_formatter)

        # Add baseline legend with different position for each panel
        legend_positions = ['upper left', 'upper right', 'upper right']
        ax.legend(loc=legend_positions[i], fontsize=12)

    # Add attribute group legend only to the first panel (lower right)
    group_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=info['color'], markersize=7,
                   label=name, markeredgecolor='black')
        for name, info in persona_groups.items()
    ]
    # Preserve the first panel's baseline legend and add attribute legend in different position
    first_ax = axes[2]
    baseline_legend = first_ax.get_legend()
    if baseline_legend:
        first_ax.add_artist(baseline_legend)
    first_ax.legend(handles=group_handles, loc='lower right', fontsize=11, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.92)

    # Save (PNG + PDF for LaTeX)
    png_path = PLOTS_DIR / f'{out_basename}.png'
    pdf_path = PLOTS_DIR / f'{out_basename}.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.01)
    plt.close()

    print(f"ME plot saved to {png_path}")

def plot_baseline_vs_personas_mae(df, out_basename="baseline_vs_personas_mean_absolute_error"):
    """
    Plot baseline vs persona Mean Absolute Error scores:
      - Shared y-axis across subplots
      - Persona names on the LEFT y-axis (only on left panel)
      - Vertical baseline line with per-panel value annotation
      - Compact layout for LaTeX
    """

    persona_groups = get_persona_groups()
    models = list(df['model'].unique())

    # ----- Build ordered personas and groups (reversed for top-to-bottom reading) -----
    ordered_personas, persona_colors, persona_labels = [], [], []
    group_positions = {}  # name -> (start_idx, end_idx, color)
    current_pos = 0
    for group_name, group_info in persona_groups.items():
        group_start = current_pos
        for persona in group_info['personas']:
            ordered_personas.append(persona)
            persona_colors.append(group_info['color'])
            label = persona.replace('_', ' ').replace('-', '-').title()
            if persona in ['15', '35', '65']:
                label = f"{persona} years"
            persona_labels.append(label)
            current_pos += 1
        group_positions[group_name] = (group_start, current_pos - 1, group_info['color'])

    # Reverse the order so first group appears at top of y-axis
    ordered_personas = ordered_personas[::-1]
    persona_colors = persona_colors[::-1]
    persona_labels = persona_labels[::-1]

    # Update group positions for reversed order
    total_personas = len(ordered_personas)
    reversed_group_positions = {}
    for group_name, (start_idx, end_idx, color) in group_positions.items():
        new_start = total_personas - 1 - end_idx
        new_end = total_personas - 1 - start_idx
        reversed_group_positions[group_name] = (new_start, new_end, color)
    group_positions = reversed_group_positions

    # ----- Layout / ticks -----
    y_spacing = 1.2
    y_ticks = [j * y_spacing for j in range(len(ordered_personas))]
    y_min = -0.6
    y_max = (len(ordered_personas) - 1) * y_spacing + 0.6

    # Model name mapping for better readability
    model_display_names = {
        'gpt_oss_120b': 'GPT-OSS-120B',
        'mistral_medium': 'Mistral-Medium',
        'qwen3_32b': 'Qwen3-32B'
    }

    # Shared y-axis across three panels - more compact
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    percent_formatter = FuncFormatter(lambda x, _: f"{x * 100:.1f}")

    for i, model in enumerate(models):
        model_display = model_display_names.get(model, model)
        ax = axes[i]
        model_data = df[df['model'] == model]
        baseline_data = model_data[model_data['persona_type'] == 'baseline']
        baseline_f1 = baseline_data['mean_absolute_error_mean'].iloc[0]
        baseline_std = baseline_data['mean_absolute_error_std'].iloc[0]

        persona_data = model_data[model_data['persona_type'] == 'persona']

        # --- Baseline: vertical line with error band (swapped coordinates) ---
        ax.axvline(x=baseline_f1, color='black', linestyle='--', linewidth=2,
                  label=f'Baseline ({baseline_f1 * 100:.1f}±{baseline_std * 100:.1f})')
        ax.fill_betweenx([y_min, y_max],  # y range matching ylim
                        baseline_f1 - baseline_std, baseline_f1 + baseline_std,
                        color='black', alpha=0.1)

        # Persona points with error bars
        for j, persona in enumerate(ordered_personas):
            row = persona_data[persona_data['persona_id'] == persona]
            if row.empty:
                continue
            f1 = row['mean_absolute_error_mean'].iloc[0]
            std = row['mean_absolute_error_std'].iloc[0]
            y_pos = j * y_spacing
            ax.errorbar(f1, y_pos, xerr=std, fmt='o',
                        color=persona_colors[j], alpha=0.9,
                        capsize=3, capthick=1, markersize=7,
                        markeredgecolor='black', markeredgewidth=0.5)

        # Y ticks shared; only left panel shows labels
        ax.set_yticks(y_ticks)
        if i == 0:
            ax.set_yticklabels(persona_labels, fontsize=12)
        else:
            ax.tick_params(axis='y', labelleft=False)

        # Colored background bands for attribute groups (all subplots for consistency)
        for group_name, (start_idx, end_idx, group_color) in group_positions.items():
            y_start = start_idx * y_spacing - 0.4
            y_end = (end_idx + 1) * y_spacing - 0.8
            ax.axhspan(y_start, y_end, color=group_color, alpha=0.08, zorder=0)

        # Group separators (thin lines between groups)
        for start_idx, _, _ in group_positions.values():
            if start_idx > 0:
                ax.axhline((start_idx - 0.5) * y_spacing, color='white', linestyle='-', alpha=0.8, linewidth=1)

        # Formatting
        ax.set_title(f'{model_display}', fontsize=16)
        ax.set_xlabel('Mean Absolute Error Score', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        # ax.set_xlim(0.24, 0.60)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_formatter(percent_formatter)

        # Add baseline legend with different position for each panel
        legend_positions = ['upper left', 'upper right', 'upper right']
        ax.legend(loc=legend_positions[i], fontsize=12)

    # Add attribute group legend only to the first panel (lower right)
    group_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=info['color'], markersize=7,
                   label=name, markeredgecolor='black')
        for name, info in persona_groups.items()
    ]
    # Preserve the first panel's baseline legend and add attribute legend in different position
    first_ax = axes[2]
    baseline_legend = first_ax.get_legend()
    if baseline_legend:
        first_ax.add_artist(baseline_legend)
    first_ax.legend(handles=group_handles, loc='lower right', fontsize=11, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.92)

    # Save (PNG + PDF for LaTeX)
    png_path = PLOTS_DIR / f'{out_basename}.png'
    pdf_path = PLOTS_DIR / f'{out_basename}.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.01)
    plt.close()

    print(f"MAE plot saved to {png_path}")

def plot_f1_heatmap(df, out_basename="f1_macro_heatmap"):
    """Plot F1 macro scores as heatmap with averaged data, grouped by attribute"""
    persona_groups = get_persona_groups()
    persona_data = df[df['persona_type'] == 'persona']

    # Create ordered list of personas by group
    ordered_personas = []
    persona_labels = []

    for group_name, group_info in persona_groups.items():
        for persona in group_info['personas']:
            ordered_personas.append(persona)
            # Create readable labels
            label = persona.replace('_', ' ').replace('-', '-').title()
            if persona in ['15', '35', '65']:
                label = f"{persona} years"
            persona_labels.append(label)

    # Reorder data according to our grouping
    models = persona_data['model'].unique()
    heatmap_data = []

    # Add baseline first
    baseline_data = df[df['persona_type'] == 'baseline']
    baseline_row = []
    for model in models:
        baseline_f1 = baseline_data[baseline_data['model'] == model]['f1_macro_mean'].iloc[0]
        baseline_row.append(baseline_f1)
    heatmap_data.append(baseline_row)

    # Add persona data in group order
    for persona in ordered_personas:
        persona_row = []
        for model in models:
            persona_model_data = persona_data[(persona_data['persona_id'] == persona) &
                                            (persona_data['model'] == model)]
            if not persona_model_data.empty:
                persona_row.append(persona_model_data['f1_macro_mean'].iloc[0])
            else:
                persona_row.append(0)
        heatmap_data.append(persona_row)

    # Model name mapping for better readability
    model_display_names = {
        'gpt_oss_120b': 'GPT-OSS-120B',
        'mistral_medium': 'Mistral-Medium',
        'qwen3_32b': 'Qwen3-32B'
    }
    model_labels = [model_display_names.get(model, model) for model in models]

    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(heatmap_data,
                             index=['BASELINE'] + persona_labels,
                             columns=model_labels)

    fig, ax = plt.subplots(figsize=(8, 14))

    # Create heatmap
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0.45, ax=ax, cbar_kws={'label': 'Macro F1 Score (averaged)'})

    # Add horizontal lines to separate groups
    group_boundaries = [0.5]  # After baseline
    current_pos = 1  # Start after baseline
    for group_name, group_info in persona_groups.items():
        current_pos += len(group_info['personas'])
        if current_pos <= len(ordered_personas):
            group_boundaries.append(current_pos + 0.5)

    for boundary in group_boundaries[:-1]:  # Don't draw line after last group
        ax.axhline(y=boundary, color='white', linewidth=2)

    # Add group labels on the right
    current_pos = 1  # Start after baseline
    for group_name, group_info in persona_groups.items():
        middle_pos = current_pos + len(group_info['personas']) / 2 - 0.5
        ax.text(len(models) + 0.1, middle_pos, group_name,
               rotation=0, va='center', fontweight='bold', fontsize=10)
        current_pos += len(group_info['personas'])

    ax.set_title('Macro F1 Score Heatmap by Attribute Groups (Averaged across 3 runs)', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Persona/Baseline', fontsize=12)

    plt.tight_layout()
    png_path = PLOTS_DIR / f'{out_basename}.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"F1 Macro heatmap saved to {png_path}")

def plot_f1_summary_statistics(df):
    """Print summary statistics for F1 macro scores"""
    print("\nMacro F1 Score Summary Statistics:")
    for model_name in ['gpt_oss_120b', 'mistral_medium', 'qwen3_32b']:
        model_data = df[df['model'] == model_name]

        # Baseline results
        baseline_data = model_data[model_data['persona_type'] == 'baseline']
        if not baseline_data.empty:
            baseline = baseline_data.iloc[0]
            print(f"{model_name} baseline: f1_macro={baseline['f1_macro_mean']:.3f}±{baseline['f1_macro_std']:.3f}")

        # Persona results (average across all personas)
        persona_data = model_data[model_data['persona_type'] == 'persona']
        if not persona_data.empty:
            f1_macro_means = persona_data['f1_macro_mean'].values
            print(f"{model_name} personas: f1_macro={np.mean(f1_macro_means):.3f}±{np.std(f1_macro_means):.3f} "
                  f"(avg across {len(f1_macro_means)} personas)")

def main():
    """Main function to generate F1 macro plots"""
    # Load results
    df = load_results()

    print("Creating F1 Macro visualizations...")

    # Create swapped baseline vs personas plot (main plot)
    plot_baseline_vs_personas_f1_macro(df)
    print("  ✓ Baseline vs personas F1 macro plot (swapped axes)")

    plot_baseline_vs_personas_me(df)
    print("  ✓ Baseline vs personas Mean Error plot (swapped axes)")

    plot_baseline_vs_personas_mae(df)
    print("  ✓ Baseline vs personas Mean Absolute Error plot (swapped axes)")
    # Create heatmap
    plot_f1_heatmap(df)
    print("  ✓ F1 macro heatmap")

    # Print summary statistics
    plot_f1_summary_statistics(df)

    print(f"\nAll plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()