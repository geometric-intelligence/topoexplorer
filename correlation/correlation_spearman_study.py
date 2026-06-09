import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.stats.multitest as mt
import matplotlib.pyplot as plt
import seaborn as sns

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
    print(dropped_few_samples)
    print(f"Dropped {len(dropped_zero_variance)} metrics due to zero variance:")
    print(dropped_zero_variance)
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

df = pd.read_csv('correlation/wandb_runs/best_runs_with_hasse_metrics.csv')

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

topotune = df[df['model_type'] == 'topotune'].copy()
topotune['normalized_score'] = topotune.apply(lambda row: normalize_score(row, baseline_dict), axis=1)
topotune = topotune.dropna(subset=['normalized_score'])

exclude_cols = ['dataset_name', 'model_type', 'model_category', 'config', 'score', 'score_std', 'val_score', 'run_count', 'normalized_score']
metric_cols = [c for c in df.columns if c not in exclude_cols and ('up_' in c or 'down_' in c or 'incidence_' in c or 'adjacency' in c)]
mean_metric_cols = [c for c in metric_cols if c.endswith('_mean')]

subsets = [
    (topotune[topotune['config.model.model_domain'].isin(['cell', 'simplicial'])], "correlation/correlation_results/combined_cell_simplicial", metric_cols),
    (topotune[topotune['config.model.model_domain'] == 'cell'], "correlation/correlation_results/cell_only", metric_cols),
    (topotune[topotune['config.model.model_domain'] == 'simplicial'], "correlation/correlation_results/simplicial_only", metric_cols),
    (topotune[topotune['config.model.model_domain'].isin(['cell', 'simplicial'])], "correlation/correlation_results/combined_cell_simplicial_mean_only", mean_metric_cols),
    (topotune[topotune['config.model.model_domain'] == 'cell'], "correlation/correlation_results/cell_only_mean_only", mean_metric_cols),
    (topotune[topotune['config.model.model_domain'] == 'simplicial'], "correlation/correlation_results/simplicial_only_mean_only", mean_metric_cols)
]

for data_subset, path_name, columns in subsets:
    res = run_correlations(data_subset, path_name, columns)
    plot_top_metric(res, data_subset, path_name)