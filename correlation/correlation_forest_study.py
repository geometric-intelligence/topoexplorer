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
        print(f"Insufficient data for {name} ({len(clean_data)} samples).")
        return None

    X_raw = clean_data[metric_cols]
    y = clean_data['normalized_score'].values

    # Check if we have enough variance in y
    if np.var(y) == 0:
        print(f"Zero variance in normalized_score for {name}. Skipping.")
        return None

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

# Load data
df = pd.read_csv('correlation/runs_summaries/best_runs_with_hasse_metrics.csv')

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

exclude_cols = ['dataset_name', 'model_type', 'model_category', 'config', 'score', 'score_std', 'val_score', 'run_count', 'normalized_score', 'domain']
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

    model_output_dir = f'correlation/random_forest_{model}_results'
    os.makedirs(model_output_dir, exist_ok=True)

    subsets = [
        (model_df[model_df['config.model.model_domain'].isin(['cell', 'simplicial'])], "combined_cell_simplicial", metric_cols),
        (model_df[model_df['config.model.model_domain'] == 'cell'], "cell_only", metric_cols),
        (model_df[model_df['config.model.model_domain'] == 'simplicial'], "simplicial_only", metric_cols),
        (model_df[model_df['config.model.model_domain'].isin(['cell', 'simplicial'])], "combined_cell_simplicial_mean_only", mean_metric_cols),
        (model_df[model_df['config.model.model_domain'] == 'cell'], "cell_only_mean_only", mean_metric_cols),
        (model_df[model_df['config.model.model_domain'] == 'simplicial'], "simplicial_only_mean_only", mean_metric_cols)
    ]

    for data_subset, name, columns in subsets:
        run_random_forest(data_subset, name, columns, model_output_dir)

print("\nRandom Forest study completed.")
