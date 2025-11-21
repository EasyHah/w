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

    # Explicitly add test_mc_0 to advanced_test group if both exist
    if 'advanced_test' in groups and 'test_mc_0' in dirs:
        # Check if it's already there (unlikely given the regex, but good for safety)
        if 'test_mc_0' not in groups['advanced_test']:
            groups['advanced_test'].append('test_mc_0')

    return groups

def load_metrics(group_dirs):
    summary_data = []
    per_image_data = pd.DataFrame()
    per_class_data_list = []

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

        # Load per_image_confusion
        confusion_dir = os.path.join(d, 'per_image_confusion')
        if os.path.exists(confusion_dir):
            txt_files = glob.glob(os.path.join(confusion_dir, '*_confusion.txt'))
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r') as f:
                        content = f.read()
                        # Extract Image name
                        img_match = re.search(r'Image: (.+)', content)
                        img_name = img_match.group(1) if img_match else os.path.basename(txt_file).replace('_confusion.txt', '')

                        # Extract Classes
                        classes_match = re.search(r'Classes \(True rows vs Pred cols\):\n(.+)', content)
                        if classes_match:
                            # Split by comma and strip whitespace
                            classes = [c.strip() for c in classes_match.group(1).split(',')]
                        else:
                            # Default fallback for older files
                            classes = ['Background', 'Foreground']

                        # Extract Counts Matrix
                        # Expecting:
                        # Counts Matrix:
                        # 379730,3281
                        # 14286,12303
                        # It captures multiple lines of numbers
                        matrix_match = re.search(r'Counts Matrix:\n((?:[\d,]+\n?)+)', content)
                        if matrix_match:
                            matrix_str = matrix_match.group(1).strip()
                            matrix_rows = matrix_str.split('\n')
                            matrix = []
                            for row in matrix_rows:
                                # Remove empty strings if any from split
                                if row.strip():
                                    matrix.append([int(x) for x in row.split(',')])

                            matrix = np.array(matrix)
                            total_sum = np.sum(matrix)

                            # Calculate per-class metrics
                            for i, class_name in enumerate(classes):
                                if i >= matrix.shape[0]:
                                    break

                                tp = matrix[i, i]
                                fp = np.sum(matrix[:, i]) - tp
                                fn = np.sum(matrix[i, :]) - tp
                                tn = total_sum - (tp + fp + fn)

                                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

                                per_class_data_list.append({
                                    'folder': d,
                                    'image': img_name,
                                    'class': class_name,
                                    'precision': precision,
                                    'recall': recall,
                                    'f1': f1,
                                    'iou': iou,
                                    'TP': tp,
                                    'FP': fp,
                                    'FN': fn,
                                    'TN': tn
                                })

                except Exception as e:
                    print(f"Error reading confusion file {txt_file}: {e}")

    return summary_data, per_image_data, pd.DataFrame(per_class_data_list)

def visualize_group(prefix, summary_data, per_image_data, per_class_data):
    output_dir = f'comparison_results_{prefix}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating visualizations for {prefix} in {output_dir}...")

    # 1. Bar chart for aggregate metrics (Overall)
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        metrics_to_plot = ['mean_precision', 'mean_recall', 'mean_dice', 'mean_iou', 'roc_auc']
        metrics_to_plot = [m for m in metrics_to_plot if m in df_summary.columns]

        if metrics_to_plot:
            df_melted = df_summary.melt(id_vars='folder', value_vars=metrics_to_plot, var_name='Metric', value_name='Value')

            plt.figure(figsize=(12, 6))
            sns.barplot(data=df_melted, x='Metric', y='Value', hue='folder')
            plt.title(f'Overall Aggregate Metrics Comparison for {prefix}')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'aggregate_metrics_comparison.png'))
            plt.close()

            # Heatmap of summary metrics
            df_heatmap = df_summary.set_index('folder')[metrics_to_plot]
            plt.figure(figsize=(10, len(df_summary) * 0.8 + 2))
            sns.heatmap(df_heatmap, annot=True, cmap='viridis', fmt='.3f')
            plt.title(f'Overall Metrics Heatmap for {prefix}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'))
            plt.close()

    # 2. Per-Class Metrics Comparison
    if not per_class_data.empty:
        per_class_dir = os.path.join(output_dir, 'per_class_comparison')
        os.makedirs(per_class_dir, exist_ok=True)

        classes = per_class_data['class'].unique()

        for cls in classes:
            # Sanitize class name for filename
            safe_cls_name = re.sub(r'[^\w\-_\.]', '_', cls)

            cls_data = per_class_data[per_class_data['class'] == cls]

            metrics = ['iou', 'f1', 'precision', 'recall']

            # Boxplot of distributions
            cls_melted = cls_data.melt(id_vars=['folder', 'image'], value_vars=metrics, var_name='Metric', value_name='Value')

            plt.figure(figsize=(12, 6))
            sns.boxplot(data=cls_melted, x='Metric', y='Value', hue='folder')
            plt.title(f'Class: {cls} - Metrics Distribution Comparison')
            plt.ylim(0, 1.1)
            plt.tight_layout()
            plt.savefig(os.path.join(per_class_dir, f'class_{safe_cls_name}_metrics_boxplot.png'))
            plt.close()

            # Barplot of Mean values
            cls_means = cls_data.groupby('folder')[metrics].mean().reset_index()
            cls_means_melted = cls_means.melt(id_vars='folder', value_vars=metrics, var_name='Metric', value_name='Mean Value')

            plt.figure(figsize=(12, 6))
            sns.barplot(data=cls_means_melted, x='Metric', y='Mean Value', hue='folder')
            plt.title(f'Class: {cls} - Mean Metrics Comparison')
            plt.ylim(0, 1.1)
            plt.tight_layout()
            plt.savefig(os.path.join(per_class_dir, f'class_{safe_cls_name}_mean_metrics.png'))
            plt.close()

    # 3. Per-Image Scatter plots (Global)
    if not per_image_data.empty:
        if 'precision' in per_image_data.columns and 'recall' in per_image_data.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=per_image_data, x='recall', y='precision', hue='folder', style='folder', alpha=0.7)
            plt.title(f'Global Precision vs Recall per Image for {prefix}')
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'global_precision_vs_recall_scatter.png'))
            plt.close()

def main():
    dirs = get_directories()
    groups = group_directories(dirs)

    for prefix, group_dirs in groups.items():
        print(f"Processing group: {prefix} with folders: {group_dirs}")
        # Sort group_dirs naturally
        try:
            group_dirs.sort(key=lambda text: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)])
        except Exception as e:
            print(f"Sorting failed, falling back to default sort: {e}")
            group_dirs.sort()

        summary_data, per_image_data, per_class_data = load_metrics(group_dirs)

        if summary_data or not per_image_data.empty or not per_class_data.empty:
            visualize_group(prefix, summary_data, per_image_data, per_class_data)
        else:
            print(f"No data found for group {prefix}")

if __name__ == '__main__':
    main()
