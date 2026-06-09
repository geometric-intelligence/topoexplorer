import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

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

def run_random_forest(data_subset, name, metric_cols, output_dir):
    clean_data = data_subset.dropna(subset=['normalized_score'])
    
    if len(clean_data) < 5:
        print(f"Insufficient data for {name}.")
        return None

    X_raw = clean_data[metric_cols]
    y = clean_data['normalized_score'].values

    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X_raw)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        oob_score=True,
        random_state=42
    )
    
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    
    res_df = pd.DataFrame({
        'Metric': metric_cols,
        'Importance': importances
    })
    
    res_df = res_df.sort_values(by='Importance', ascending=False)
    res_df = res_df[res_df['Importance'] > 0]
    
    output_path = os.path.join(output_dir, f"{name}_rf_importance.csv")
    res_df.to_csv(output_path, index=False)
    
    print(f"[{name}] OOB Score (R^2): {rf.oob_score_:.4f}")
    
    return res_df

output_dir = 'correlation/random_forest_results'
os.makedirs(output_dir, exist_ok=True)

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

run_random_forest(
    topotune[topotune['config.model.model_domain'].isin(['cell', 'simplicial'])], 
    "combined_cell_simplicial", 
    metric_cols,
    output_dir
)

run_random_forest(
    topotune[topotune['config.model.model_domain'] == 'cell'], 
    "cell_only", 
    metric_cols,
    output_dir
)

run_random_forest(
    topotune[topotune['config.model.model_domain'] == 'simplicial'], 
    "simplicial_only", 
    metric_cols,
    output_dir
)

run_random_forest(
    topotune[topotune['config.model.model_domain'].isin(['cell', 'simplicial'])], 
    "combined_cell_simplicial_mean_only", 
    mean_metric_cols,
    output_dir
)

run_random_forest(
    topotune[topotune['config.model.model_domain'] == 'cell'], 
    "cell_only_mean_only", 
    mean_metric_cols,
    output_dir
)

run_random_forest(
    topotune[topotune['config.model.model_domain'] == 'simplicial'], 
    "simplicial_only_mean_only", 
    mean_metric_cols,
    output_dir
)