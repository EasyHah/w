import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

def get_directories():
    return [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.')]

def group_directories(dirs):
    groups = {}
    for d in dirs:
        # Match pattern <prefix>_<suffix>
        # We want to capture the prefix which is everything up to the last underscore
        match = re.match(r'^(.*)_([^_]+)$', d)
        if match:
            prefix = match.group(1)
            suffix = match.group(2)
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(d)
    return groups

def load_metrics(group_dirs):
    summary_data = []
    per_image_data = pd.DataFrame()

    for d in group_dirs:
        # Load metrics_summary.json
        summary_path = os.path.join(d, 'metrics_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                try:
                    data = json.load(f)
                    # Add folder name to data
                    data['folder'] = d
                    summary_data.append(data)
                except json.JSONDecodeError:
                    print(f"Error decoding {summary_path}")

        # Load per_image_metrics.csv
        csv_path = os.path.join(d, 'per_image_metrics.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df['folder'] = d
                per_image_data = pd.concat([per_image_data, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

    return summary_data, per_image_data

def visualize_group(prefix, summary_data, per_image_data):
    output_dir = f'comparison_results_{prefix}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating visualizations for {prefix} in {output_dir}...")

    # 1. Bar chart for aggregate metrics
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        metrics_to_plot = ['mean_precision', 'mean_recall', 'mean_dice', 'mean_iou', 'roc_auc']
        # Filter columns that actually exist
        metrics_to_plot = [m for m in metrics_to_plot if m in df_summary.columns]

        if metrics_to_plot:
            # Melt for plotting
            df_melted = df_summary.melt(id_vars='folder', value_vars=metrics_to_plot, var_name='Metric', value_name='Value')

            plt.figure(figsize=(12, 6))
            sns.barplot(data=df_melted, x='Metric', y='Value', hue='folder')
            plt.title(f'Aggregate Metrics Comparison for {prefix}')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'aggregate_metrics_comparison.png'))
            plt.close()

        # 4. Heatmap of summary metrics
        if metrics_to_plot:
            df_heatmap = df_summary.set_index('folder')[metrics_to_plot]
            # Normalize or just show values? Just show values for now.
            plt.figure(figsize=(10, len(df_summary) * 0.8 + 2))
            sns.heatmap(df_heatmap, annot=True, cmap='viridis', fmt='.3f')
            plt.title(f'Metrics Heatmap for {prefix}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'))
            plt.close()

    # 2. Box plots and Violin plots for per-image metrics distributions
    if not per_image_data.empty:
        metrics_to_dist = ['dice', 'iou', 'precision', 'recall']
        metrics_to_dist = [m for m in metrics_to_dist if m in per_image_data.columns]

        for metric in metrics_to_dist:
            # Box plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=per_image_data, x='folder', y=metric)
            plt.title(f'{metric} Distribution Comparison for {prefix}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_distribution_boxplot.png'))
            plt.close()

            # Violin plot
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=per_image_data, x='folder', y=metric)
            plt.title(f'{metric} Distribution Comparison for {prefix}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_distribution_violinplot.png'))
            plt.close()

        # 3. Scatter plot Precision vs Recall
        if 'precision' in per_image_data.columns and 'recall' in per_image_data.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=per_image_data, x='recall', y='precision', hue='folder', style='folder', alpha=0.7)
            plt.title(f'Precision vs Recall per Image for {prefix}')
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'precision_vs_recall_scatter.png'))
            plt.close()

def main():
    dirs = get_directories()
    groups = group_directories(dirs)

    for prefix, group_dirs in groups.items():
        # Only compare if there's more than 1 folder, or if user explicitly wants visualization for single groups too.
        # User said "compare contents in folders", implying plural.
        # But even with 1 folder, visualization is useful. I'll process all groups.

        print(f"Processing group: {prefix} with folders: {group_dirs}")
        # Sort group_dirs naturally
        try:
            # Natural sort
            group_dirs.sort(key=lambda text: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)])
        except Exception as e:
            print(f"Sorting failed, falling back to default sort: {e}")
            group_dirs.sort()

        summary_data, per_image_data = load_metrics(group_dirs)

        if summary_data or not per_image_data.empty:
            visualize_group(prefix, summary_data, per_image_data)
        else:
            print(f"No data found for group {prefix}")

if __name__ == '__main__':
    main()
