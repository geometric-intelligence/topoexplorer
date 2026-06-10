import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.stats.multitest as mt
import matplotlib.pyplot as plt
import seaborn as sns
import os

def normalize_score(row, baseline_dict):
    ds = row['dataset_name']
    if ds not in baseline_dict:
        return np.nan
    
    base_score = baseline_dict[ds]['score']
    metric = baseline_dict[ds]['metric']
    model_score = row['score']
    
    if 'mae' in metric:
        if base_score == 0: 
            return np.nan
        return (base_score - model_score) / base_score
    else:
        if base_score >= 1.0: 
            return 0.0
        return (model_score - base_score) / (1.0 - base_score)

def run_correlations(data_subset, name, metric_cols):
    results = []
    dropped_few_samples = []
    dropped_zero_variance = []
    
    for metric in metric_cols:
        clean_data = data_subset.dropna(subset=[metric, 'normalized_score'])
        
        if len(clean_data) < 5:
            dropped_few_samples.append(metric)
            continue
            
        if clean_data[metric].nunique() <= 1 or clean_data['normalized_score'].nunique() <= 1:
            dropped_zero_variance.append(metric)
            continue
            
        rho, p_val = spearmanr(clean_data[metric], clean_data['normalized_score'])
        results.append({
            'Metric': metric,
            'Spearman_Rho': rho,
            'P_Value': p_val,
            'N_samples': len(clean_data)
        })
        
    print(f"Exclusion summary for {name}:")
    print(f"Dropped {len(dropped_few_samples)} metrics due to < 5 valid samples:")
    # print(dropped_few_samples) # Too verbose
    print(f"Dropped {len(dropped_zero_variance)} metrics due to zero variance:")
    # print(dropped_zero_variance) # Too verbose
    print("")
    
    res_df = pd.DataFrame(results).dropna()
    
    if len(res_df) > 0:
        res_df['Adj_P_Value'] = mt.multipletests(res_df['P_Value'], method='fdr_bh')[1]
        res_df['Significant'] = res_df['Adj_P_Value'] < 0.05
    
    if not res_df.empty:
        res_df['Abs_Rho'] = res_df['Spearman_Rho'].abs()
        res_df = res_df.sort_values(by='Abs_Rho', ascending=False).drop(columns=['Abs_Rho'])
        res_df.to_csv(f"{name}.csv", index=False)
    
    return res_df

def plot_top_metric(res_df, data_subset, name):
    if res_df.empty or not res_df['Significant'].any():
        return

    top_metric = res_df[res_df['Significant']].iloc[0]['Metric']
    clean_data = data_subset.dropna(subset=[top_metric, 'normalized_score']).copy()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=clean_data, x=top_metric, y='normalized_score', s=100)
    
    for i, row in clean_data.iterrows():
        plt.annotate(
            row['dataset_name'], 
            (row[top_metric], row['normalized_score']),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
        
    plt.title(f"Top Metric: {top_metric}")
    plt.xlabel(top_metric)
    plt.ylabel("Normalized Score Improvement")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{name}_plot.png", dpi=300)
    plt.close()

# Load data
input_file = 'correlation/runs_summaries/best_runs_with_hasse_metrics.csv'
if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
    exit(1)

df = pd.read_csv(input_file)

# Calculate baselines
baseline_models = ['gin', 'gat', 'gcn']
baselines = df[df['model_type'].isin(baseline_models)]

baseline_dict = {}
for dataset, group in baselines.groupby('dataset_name'):
    score_name = group['score_name'].iloc[0].lower()
    if 'mae' in score_name:
        best_score = group['score'].min()
    else:
        best_score = group['score'].max()
    baseline_dict[dataset] = {'score': best_score, 'metric': score_name}

# Identification of metric columns
exclude_cols = [
    'dataset_name', 'model_type', 'model_category', 'config', 'score', 
    'score_std', 'val_score', 'run_count', 'normalized_score', 'domain', 
    'score_name'
]
# Filter columns that are likely Hasse metrics
metric_cols = [c for c in df.columns if c not in exclude_cols and any(x in c for x in ['up_', 'down_', 'incidence_', 'adjacency'])]
mean_metric_cols = [c for c in metric_cols if c.endswith('_mean')]

# Models to study
models_to_study = ['topotune', 'hopse_m']

for model in models_to_study:
    print(f"\n{'='*40}")
    print(f"Analyzing model: {model}")
    print(f"{'='*40}\n")
    
    model_df = df[df['model_type'] == model].copy()
    if model_df.empty:
        print(f"No data found for model: {model}")
        continue
        
    model_df['normalized_score'] = model_df.apply(lambda row: normalize_score(row, baseline_dict), axis=1)
    model_df = model_df.dropna(subset=['normalized_score'])
    
    if model_df.empty:
        print(f"No valid normalized scores for model: {model}")
        continue

    output_dir = f"correlation/spearman_{model}_results"
    os.makedirs(output_dir, exist_ok=True)
    
    subsets = [
        (model_df[model_df['config.model.model_domain'].isin(['cell', 'simplicial'])], f"{output_dir}/combined_cell_simplicial", metric_cols),
        (model_df[model_df['config.model.model_domain'] == 'cell'], f"{output_dir}/cell_only", metric_cols),
        (model_df[model_df['config.model.model_domain'] == 'simplicial'], f"{output_dir}/simplicial_only", metric_cols),
        (model_df[model_df['config.model.model_domain'].isin(['cell', 'simplicial'])], f"{output_dir}/combined_cell_simplicial_mean_only", mean_metric_cols),
        (model_df[model_df['config.model.model_domain'] == 'cell'], f"{output_dir}/cell_only_mean_only", mean_metric_cols),
        (model_df[model_df['config.model.model_domain'] == 'simplicial'], f"{output_dir}/simplicial_only_mean_only", mean_metric_cols)
    ]

    for data_subset, path_name, columns in subsets:
        if data_subset.empty:
            print(f"Subset empty for {path_name}, skipping.")
            continue
        res = run_correlations(data_subset, path_name, columns)
        plot_top_metric(res, data_subset, path_name)

print("\nSpearman correlation study completed.")
