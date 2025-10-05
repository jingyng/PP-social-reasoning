import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

def load_data():
    """Load the averaged flagging rates data"""
    csv_file = THIS_DIR / "csv" / "accuracy_f1_me_mae_flagging_rates_results_averaged.csv"
    df = pd.read_csv(csv_file)
    return df

def get_persona_display_names():
    """Get mapping from persona IDs to display names"""
    mapping = {
        'baseline': 'Baseline',
        '15': '15y',
        '35': '35y',
        '65': '65y',
        'no_formal_education': 'No Edu',
        'high_school_education': 'HS Edu',
        'higher_education': 'Higher Edu',
        'male': 'Male',
        'female': 'Female',
        'not_lonely': 'Not Lonely',
        'somewhat_lonely': 'Lonely',
        'left-wing': 'Left',
        'right-wing': 'Right',
        'centrist': 'Center',
        'white': 'White',
        'black': 'Black',
        'asian': 'Asian',
        'christian': 'Christian',
        'muslim': 'Muslim',
        'jewish': 'Jewish',
        'atheist': 'Atheist',
        'hindu': 'Hindu'
    }
    return mapping

def create_heatmap_data(df, metric_column, models=['gpt_oss_120b', 'mistral_medium', 'qwen3_32b']):
    """Create a pivot table for heatmap visualization"""
    # Get persona display names
    persona_mapping = get_persona_display_names()

    # Filter data and add display names
    filtered_df = df[df['model'].isin(models)].copy()
    filtered_df['persona_display'] = filtered_df['persona_id'].map(persona_mapping)

    # Create pivot table
    pivot_data = filtered_df.pivot(index='persona_display', columns='model', values=metric_column)

    # Define the order of personas (baseline first, then by category)
    persona_order = [
        'Baseline',
        '15y', '35y', '65y',
        'No Edu', 'HS Edu', 'Higher Edu',
        'Male', 'Female',
        'Not Lonely', 'Lonely',
        'Left', 'Center', 'Right',
        'White', 'Black', 'Asian',
        'Christian', 'Muslim', 'Jewish', 'Atheist', 'Hindu'
    ]

    # Reorder rows according to persona_order
    available_personas = [p for p in persona_order if p in pivot_data.index]
    pivot_data = pivot_data.reindex(available_personas)

    return pivot_data

def create_model_heatmap_data(df, model_name):
    """Create data for a single model across all three flagging rate metrics"""
    # Get persona display names
    persona_mapping = get_persona_display_names()

    # Filter data for the specific model
    model_df = df[df['model'] == model_name].copy()
    model_df['persona_display'] = model_df['persona_id'].map(persona_mapping)

    # Create a dataframe with personas as rows and metrics as columns
    metrics_data = []
    metric_names = ['N→H', 'N→O', 'O→H']
    metric_columns = ['hate_over_flagging_rate_mean', 'offensive_over_flagging_rate_mean', 'offensive_to_hate_rate_mean']

    for persona_display in model_df['persona_display'].unique():
        persona_data = model_df[model_df['persona_display'] == persona_display].iloc[0]
        row_data = [persona_data[col] for col in metric_columns]
        metrics_data.append(row_data)

    # Create DataFrame
    import pandas as pd
    result_df = pd.DataFrame(metrics_data,
                           index=model_df['persona_display'].unique(),
                           columns=metric_names)

    # Define the order of personas (baseline first, then by category)
    persona_order = [
        'Baseline',
        '15y', '35y', '65y',
        'No Edu', 'HS Edu', 'Higher Edu',
        'Male', 'Female',
        'Not Lonely', 'Lonely',
        'Left', 'Center', 'Right',
        'White', 'Black', 'Asian',
        'Christian', 'Muslim', 'Jewish', 'Atheist', 'Hindu'
    ]

    # Reorder rows according to persona_order
    available_personas = [p for p in persona_order if p in result_df.index]
    result_df = result_df.reindex(available_personas)

    return result_df

def create_merged_heatmap_data(df):
    """Create data for all models and metrics in a single heatmap"""
    # Get persona display names
    persona_mapping = get_persona_display_names()

    models = ['gpt_oss_120b', 'mistral_medium', 'qwen3_32b']
    model_names = ['GPT-OSS', 'Mistral', 'Qwen3']
    metric_names = ['N→H', 'N→O', 'O→H']
    metric_columns = ['hate_over_flagging_rate_mean', 'offensive_over_flagging_rate_mean', 'offensive_to_hate_rate_mean']

    # Create column names for the merged heatmap
    merged_columns = []
    for model_name in model_names:
        for metric_name in metric_names:
            merged_columns.append(f'{model_name}\n{metric_name}')

    # Get personas in order
    persona_order = [
        'Baseline',
        '15y', '35y', '65y',
        'No Edu', 'HS Edu', 'Higher Edu',
        'Male', 'Female',
        'Not Lonely', 'Lonely',
        'Left', 'Center', 'Right',
        'White', 'Black', 'Asian',
        'Christian', 'Muslim', 'Jewish', 'Atheist', 'Hindu'
    ]

    # Create the merged data
    merged_data = []
    for persona in persona_order:
        row_data = []
        for model in models:
            model_df = df[df['model'] == model].copy()
            model_df['persona_display'] = model_df['persona_id'].map(persona_mapping)

            # Find the data for this persona and model
            persona_data = model_df[model_df['persona_display'] == persona]
            if not persona_data.empty:
                persona_data = persona_data.iloc[0]
                for col in metric_columns:
                    row_data.append(persona_data[col])
            else:
                # Fill with NaN if data not found
                row_data.extend([np.nan] * len(metric_columns))

        merged_data.append(row_data)

    # Create DataFrame
    import pandas as pd
    result_df = pd.DataFrame(merged_data, index=persona_order, columns=merged_columns)

    # Remove any personas that have all NaN values
    result_df = result_df.dropna(how='all')

    return result_df

def plot_heatmap(data, title, filename, cmap='Reds', figsize=(10, 12)):
    """Create and save a heatmap using matplotlib"""
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap using matplotlib imshow with tighter spacing
    im = ax.imshow(data.values, cmap=cmap, aspect='equal', interpolation='nearest')

    # Set ticks and labels with tighter spacing
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_xticklabels(data.columns, fontsize=12)
    ax.set_yticklabels(data.index, fontsize=12)

    # Reduce tick spacing and remove margins
    ax.tick_params(axis='both', which='major', pad=2)
    ax.set_xlim(-0.5, len(data.columns)-0.5)
    ax.set_ylim(len(data.index)-0.5, -0.5)

    # Add text annotations with better contrast
    data_min, data_max = data.values.min(), data.values.max()
    threshold = data_min + 0.7 * (data_max - data_min)  # Use white only for top 30% of values
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            value = data.iloc[i, j]
            text_color = "white" if value > threshold else "black"
            text = ax.text(j, i, f'{value:.2f}',
                         ha="center", va="center", color=text_color, fontweight='bold', fontsize=9)

    # Add colorbar matching plot height
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Rate', rotation=270, labelpad=20, fontsize=13)

    # Set title and labels
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Persona', fontsize=14, fontweight='bold')

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Heatmap saved to: {filename}")

def main():
    # Load data
    df = load_data()

    # Create output directory
    plots_dir = THIS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Define metrics to plot
    metrics = [
        {
            'column': 'hate_over_flagging_rate_mean',
            'title': 'Hate Over-flagging Rate\n(P(predict Hate | GT=Normal))',
            'filename': plots_dir / 'hate_over_flagging_rate_heatmap.png',
            'cmap': 'Reds'
        },
        {
            'column': 'offensive_over_flagging_rate_mean',
            'title': 'Offensive Over-flagging Rate\n(P(predict Offensive | GT=Normal))',
            'filename': plots_dir / 'offensive_over_flagging_rate_heatmap.png',
            'cmap': 'Oranges'
        },
        {
            'column': 'offensive_to_hate_rate_mean',
            'title': 'Offensive-to-Hate Escalation Rate\n(P(predict Hate | GT=Offensive))',
            'filename': plots_dir / 'offensive_to_hate_rate_heatmap.png',
            'cmap': 'Purples'
        }
    ]

    # Create heatmaps for each metric
    for metric in metrics:
        print(f"\nCreating heatmap for {metric['title']}...")

        # Create pivot data
        heatmap_data = create_heatmap_data(df, metric['column'])

        # Plot heatmap
        plot_heatmap(
            data=heatmap_data,
            title=metric['title'],
            filename=metric['filename'],
            cmap=metric['cmap']
        )

        # Print summary statistics
        print(f"Range: {heatmap_data.min().min():.3f} - {heatmap_data.max().max():.3f}")
        print(f"Mean: {heatmap_data.mean().mean():.3f}")

    # Create a combined plot with each model as a subfigure
    print(f"\nCreating combined heatmap with one model per subfigure...")
    models = ['gpt_oss_120b', 'mistral_medium', 'qwen3_32b']
    model_names = ['GPT-OSS-120B', 'Mistral-Medium', 'Qwen3-32B']

    fig, axes = plt.subplots(1, 3, figsize=(14, 12))

    # Get all data to determine global min/max for consistent color scaling
    all_data = []
    model_datasets = []
    for model in models:
        model_data = create_model_heatmap_data(df, model)
        model_datasets.append(model_data)
        all_data.extend(model_data.values.flatten())

    # Calculate global min and max for consistent color scaling
    vmin = min(all_data)
    vmax = max(all_data)

    # Use a single colormap for all three plots
    shared_cmap = 'RdYlBu_r'  # Red-Yellow-Blue reversed for intuitive interpretation

    for i, (model, model_name, model_data) in enumerate(zip(models, model_names, model_datasets)):
        # Create heatmap using matplotlib imshow with consistent color scale and tighter spacing
        im = axes[i].imshow(model_data.values, cmap=shared_cmap, aspect='equal', vmin=vmin, vmax=vmax, interpolation='nearest')

        # Set ticks and labels with tighter spacing
        axes[i].set_xticks(np.arange(len(model_data.columns)))
        axes[i].set_yticks(np.arange(len(model_data.index)))
        axes[i].set_xticklabels(model_data.columns, fontsize=12)

        # Reduce tick spacing
        axes[i].tick_params(axis='both', which='major', pad=2)

        # Only show y-labels on the first subplot
        if i == 0:
            axes[i].set_yticklabels(model_data.index, fontsize=12)
            axes[i].set_ylabel('Persona', fontsize=14, fontweight='bold')
        else:
            axes[i].set_yticklabels([])

        # Add text annotations with better contrast logic
        for j in range(len(model_data.index)):
            for k in range(len(model_data.columns)):
                value = model_data.iloc[j, k]
                # Use a more conservative threshold for text color
                # Use white text only for the darkest colors, black for everything else
                threshold = vmin + 0.7 * (vmax - vmin)  # Use white only for top 30% of values
                text_color = "white" if value > threshold else "black"
                text = axes[i].text(k, j, f'{value:.2f}',
                                  ha="center", va="center", color=text_color,
                                  fontsize=9, fontweight='bold')

        # Add colorbar only to the last subplot to avoid clutter
        if i == len(models) - 1:
            cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.set_label('Flagging Rate', rotation=270, labelpad=20, fontsize=13)

        # Set title and labels
        axes[i].set_title(model_name, fontsize=14, fontweight='bold', pad=15)
        axes[i].set_xlabel('Flagging Type', fontsize=12, fontweight='bold')

        # Remove extra margins around the heatmap
        axes[i].set_xlim(-0.5, len(model_data.columns)-0.5)
        axes[i].set_ylim(len(model_data.index)-0.5, -0.5)

        # Don't rotate x-axis labels since they're now short
        # plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add main title
    # plt.suptitle('Flagging Rates Across Models and Personas', fontsize=16, fontweight='bold', y=0.96)

    # Adjust spacing between subplots - much tighter spacing
    plt.subplots_adjust(wspace=0.05, hspace=0.02)
    plt.tight_layout(pad=0.5)

    combined_filename = plots_dir / 'flagging_rates_by_model_heatmap.png'
    plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Combined heatmap by model saved to: {combined_filename}")
    print(f"Global range: {vmin:.3f} - {vmax:.3f}")

    # Create a single merged heatmap with all models and metrics (transposed)
    print(f"\nCreating single merged heatmap (transposed)...")
    merged_data = create_merged_heatmap_data(df)

    # Transpose the data
    transposed_data = merged_data.T

    # Create the merged heatmap with reduced height for tighter row spacing
    fig, ax = plt.subplots(figsize=(16, 6))

    # Use the same color scale as before (aspect='auto' allows tighter row spacing)
    im = ax.imshow(transposed_data.values, cmap=shared_cmap, aspect='auto', vmin=vmin, vmax=vmax, interpolation='nearest')

    # Set ticks and labels with larger fonts
    ax.set_xticks(np.arange(len(transposed_data.columns)))
    ax.set_yticks(np.arange(len(transposed_data.index)))
    ax.set_xticklabels(transposed_data.columns, fontsize=12, rotation=45, ha='right')

    # Create custom y-axis labels: show model name on the left of metric name for all rows
    y_labels = []
    for i, label in enumerate(transposed_data.index):
        # Extract model and metric names
        if '\n' in label:
            model_name = label.split('\n')[0]
            metric_name = label.split('\n')[1]
            # Show model name on all rows
            y_labels.append(f'{model_name} {metric_name}')
        else:
            y_labels.append(label)

    ax.set_yticklabels(y_labels, fontsize=12)

    # Reduce spacing and remove margins
    ax.tick_params(axis='both', which='major', pad=2, length=0)
    ax.set_xlim(-0.5, len(transposed_data.columns)-0.5)
    ax.set_ylim(len(transposed_data.index)-0.5, -0.5)

    # Reduce space between rows by adjusting aspect ratio
    ax.set_aspect('auto')

    # Add text annotations with percentages
    for i in range(len(transposed_data.index)):
        for j in range(len(transposed_data.columns)):
            value = transposed_data.iloc[i, j]
            if not np.isnan(value):
                threshold = vmin + 0.7 * (vmax - vmin)
                text_color = "white" if value > threshold else "black"
                # Convert to percentage
                value_pct = value * 100
                ax.text(j, i, f'{value_pct:.1f}',
                       ha="center", va="center", color=text_color,
                       fontsize=10, fontweight='bold')

    # Add smaller vertical colorbar closer to the figure
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.04, shrink=0.8, aspect=30)
    cbar.set_label('Over-flagging Rate', fontsize=16, fontweight='bold')

    # Set labels with larger fonts
    # ax.set_xlabel('Persona', fontsize=16, fontweight='bold')
    # ax.set_ylabel('Model & Flagging Type', fontsize=16, fontweight='bold')

    # Add horizontal lines to separate models (now they're rows) - thicker for better separation
    for i in [2.5, 5.5]:
        ax.axhline(y=i, color='white', linewidth=4)

    plt.tight_layout()

    merged_filename = plots_dir / 'merged_flagging_rates_heatmap_transposed.png'
    plt.savefig(merged_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Transposed merged heatmap saved to: {merged_filename}")

    print("\nAll heatmaps created successfully!")

if __name__ == "__main__":
    main()