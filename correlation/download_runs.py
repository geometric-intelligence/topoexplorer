#!/usr/bin/env python3
"""
Download all W&B runs for a specific entity and projects starting with a prefix.
Saves each project's runs as a CSV file in correlation/wandb_runs/.
Saves best runs and summaries in correlation/runs_summaries/.
Aggregates results by hyperparameters and identifies best runs.
"""

import argparse
import gc
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import wandb
from tqdm import tqdm

# --- Configuration & Paths ---
WANDB_ENTITY = "gbg141-hopse"
OUTPUT_DIR = Path("correlation/wandb_runs/")
SUMMARIES_DIR = Path("correlation/runs_summaries/")

# Project Prefixes
PREFIX_TOPOTUNE = "topotune"
PREFIXES_GNN = ["gcn", "gat", "gin"]
PREFIXES_HOPSE_M = ["hopse_m"]
ALL_PREFIXES = [PREFIX_TOPOTUNE] + PREFIXES_GNN + PREFIXES_HOPSE_M

# Output Filenames
NAME_AGGREGATED = "aggregated_results.csv"
NAME_AGGREGATED_GNN = "aggregated_results_gnn.csv"
NAME_AGGREGATED_HOPSE_M = "aggregated_results_hopse_m.csv"

NAME_BEST_RUNS = "best_runs_summary.csv"
NAME_BEST_RUNS_GNN = "best_runs_summary_gnn.csv"
NAME_BEST_RUNS_HOPSE_M = "best_runs_summary_hopse_m.csv"

# Hyperparameters to group by
GROUP_HPARAMS = [
    "config.model.model_domain",
    "config.model.tune_gnn",
    "config.model.backbone.neighborhoods",
    "config.model.backbone.GNN.num_layers",
    "config.model.feature_encoder.out_channels",
    "config.model.feature_encoder.proj_dropout",
    "config.optimizer.parameters.lr",
    "config.optimizer.parameters.weight_decay",
    "config.dataset.dataloader_params.batch_size",
]

def flatten_config(obj: Any, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flattens a nested dictionary."""
    out: dict[str, Any] = {}
    if not isinstance(obj, Mapping):
        return {parent_key: obj} if parent_key else {}

    for k, v in obj.items():
        k = str(k)
        if not parent_key and k.startswith("_"):
            continue
        if isinstance(v, dict) and "value" in v:
            v = v["value"]
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, Mapping):
            out.update(flatten_config(v, new_key, sep=sep))
        else:
            out[new_key] = v
    return out

def download_project_runs(api: wandb.Api, entity: str, project: str, output_path: Path, force: bool = False):
    """Downloads all runs for a given project and saves them to a CSV."""
    if output_path.exists() and not force:
        return

    print(f"Downloading runs for project: {entity}/{project}")
    runs = api.runs(f"{entity}/{project}")
    
    rows = []
    for run in tqdm(runs, desc=f"Runs in {project}", leave=False):
        run_data = {
            "run_id": run.id, "run_name": run.name, "run_state": run.state,
            "run_url": run.url, "run_tags": ",".join(run.tags) if run.tags else "",
            "run_created_at": run.created_at,
        }
        config = flatten_config(run.config)
        for k, v in config.items(): run_data[f"config.{k}"] = v
        summary = flatten_config(run.summary._json_dict)
        for k, v in summary.items(): run_data[f"summary.{k}"] = v
        rows.append(run_data)
    
    if rows:
        df = pd.DataFrame(rows)
        del rows
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df)} runs to {output_path}")
        del df
    gc.collect()

def aggregate_results(input_dir: Path, output_file: Path, target_prefixes: list[str]):
    """Aggregates CSVs matching target_prefixes into a single CSV."""
    csv_files = []
    for prefix in target_prefixes:
        csv_files.extend(list(input_dir.glob(f"{prefix}_*.csv")))
    
    if not csv_files:
        print(f"No CSV files found for prefixes: {target_prefixes}")
        return None

    all_dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        fname = f.stem
        model_type, dataset_name = "unknown", fname
        for pref in ALL_PREFIXES:
            if fname.startswith(f"{pref}_"):
                model_type = pref
                dataset_name = fname.replace(f"{pref}_", "", 1)
                # Remove ablation prefix from dataset name if present
                if dataset_name.startswith("ablation_"):
                    dataset_name = dataset_name.replace("ablation_", "", 1)
                break
        df["dataset_name"], df["model_type"] = dataset_name, model_type
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    # Deduplicate
    if "config.dataset.split_params.data_seed" in combined_df.columns:
        combined_df = combined_df.sort_values("run_created_at", ascending=False)
        all_config_cols = [c for c in combined_df.columns if c.startswith("config.")]
        dedup_cols = ["dataset_name", "model_type"] + all_config_cols
        combined_df = combined_df.drop_duplicates(subset=dedup_cols, keep="first")

    # Metrics
    metric_cols = [c for c in combined_df.columns if c.startswith(("summary.test_best_rerun/", "summary.val_best_rerun/"))]
    if not metric_cols:
        print(f"No metrics found for aggregation in {output_file}")
        del combined_df
        return None

    for col in metric_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    group_cols = ["dataset_name", "model_type"] + [c for c in GROUP_HPARAMS if c in combined_df.columns]
    agg_dict = {col: ['mean', 'std'] for col in metric_cols}
    summary_df = combined_df.groupby(group_cols, dropna=False).agg(agg_dict)
    summary_df.columns = [f"{col}_{stat}" for col, stat in summary_df.columns]
    summary_df["run_count"] = combined_df.groupby(group_cols, dropna=False).size()
    summary_df = summary_df.reset_index()
    
    del combined_df
    gc.collect()
    
    summary_df.to_csv(output_file, index=False)
    print(f"Aggregated results saved to {output_file}")
    return summary_df

def save_best_runs(df: pd.DataFrame, output_file: Path, model_category_prefix: str):
    """Saves the best run (by validation performance) for each dataset and domain."""
    if df is None or df.empty: return
    
    # Filter out undesired datasets
    df = df[~df["dataset_name"].str.contains("cocitation|ZINC", case=False, regex=True)]
    
    processed_rows = []
    domain_col = "config.model.model_domain" if "config.model.model_domain" in df.columns else None
    
    for ds_name in df["dataset_name"].unique():
        ds_full_group = df[df["dataset_name"] == ds_name].copy()
        
        # Handle Betti numbers separately (multi-target)
        if "betti_numbers" in ds_name:
            for suffix in ["1", "2"]:
                metric_base = f"f1-{suffix}"
                val_col = f"summary.val_best_rerun/{metric_base}_mean"
                test_col = f"summary.test_best_rerun/{metric_base}_mean"
                test_std = f"summary.test_best_rerun/{metric_base}_std"
                
                if val_col not in ds_full_group.columns: continue
                
                # Further group by domain if available
                groups = ds_full_group.groupby(domain_col) if domain_col else [(None, ds_full_group)]
                for domain, group in groups:
                    sub_ds = group.dropna(subset=[val_col]).copy()
                    if sub_ds.empty: continue
                    best_row = sub_ds.loc[sub_ds[val_col].idxmax()].copy()
                    best_row["dataset_name"] = f"mantra_betti_number_{suffix}"
                    best_row["score_name"], best_row["score"], best_row["score_std"] = "f1", best_row[test_col], best_row[test_std]
                    best_row["val_score"] = best_row[val_col]
                    best_row["model_category"] = f"{model_category_prefix}_{domain}" if domain else model_category_prefix
                    processed_rows.append(best_row)
        else:
            # Determine metric and goal
            if any(x in ds_name for x in ["Clearance_Hepatocyte_AZ", "Caco2_Wang"]):
                metric, goal = "mae", "min"
            elif "mantra" in ds_name:
                metric, goal = "f1", "max"
            else:
                metric, goal = "accuracy", "max"
                
            val_col, test_col, test_std = f"summary.val_best_rerun/{metric}_mean", f"summary.test_best_rerun/{metric}_mean", f"summary.test_best_rerun/{metric}_std"
            if val_col not in ds_full_group.columns: continue
            
            groups = ds_full_group.groupby(domain_col) if domain_col else [(None, ds_full_group)]
            for domain, group in groups:
                sub_ds = group.dropna(subset=[val_col]).copy()
                if sub_ds.empty: continue
                best_idx = sub_ds[val_col].idxmin() if goal == "min" else sub_ds[val_col].idxmax()
                best_row = sub_ds.loc[best_idx].copy()
                best_row["score_name"], best_row["score"], best_row["score_std"] = metric, best_row[test_col], best_row[test_std]
                best_row["val_score"] = best_row[val_col]
                best_row["model_category"] = f"{model_category_prefix}_{domain}" if domain else model_category_prefix
                processed_rows.append(best_row)

    if processed_rows:
        best_runs = pd.DataFrame(processed_rows)
        cols_to_keep = ["dataset_name", "model_type", "model_category", "score_name", "score", "score_std", "val_score", "run_count"] + \
                       [c for c in GROUP_HPARAMS if c in best_runs.columns]
        best_runs[cols_to_keep].sort_values(["dataset_name", "model_category"]).to_csv(output_file, index=False)
        print(f"Best runs summary saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Download and aggregate W&B runs.")
    parser.add_argument("--entity", default=WANDB_ENTITY, help="W&B entity name")
    parser.add_argument("--projects", nargs="+", help="Specific project names to download (bypasses prefix search)")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force download even if files exist")
    parser.add_argument("--skip-aggregation", action="store_true", help="Skip the aggregation step")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=120)
    
    if args.projects:
        target_projects = args.projects
    else:
        print(f"Fetching projects for entity: {args.entity} matching prefixes: {ALL_PREFIXES}")
        try:
            # Force evaluation of the paginator to catch auth errors early
            all_projects = list(api.projects(entity=args.entity))
            target_projects = [p.name for p in all_projects if any(p.name.startswith(pref) for pref in ALL_PREFIXES)]
        except Exception as e:
            print(f"\nError: Failed to fetch projects from W&B: {e}")
            if "relogin required" in str(e).lower():
                print(">>> Your W&B session may have expired. Please run `wandb login` in your terminal.")
            return

    if not target_projects:
        print("No matching projects found.")
        return
        
    print(f"Found {len(target_projects)} projects to process.")
    
    for project_name in tqdm(target_projects, desc="Downloading"):
        # Re-instantiate API to clear potential internal caches between projects
        current_api = wandb.Api(timeout=120)
        download_project_runs(current_api, args.entity, project_name, args.output_dir / f"{project_name}.csv", force=args.force)
        del current_api
        gc.collect()

    if not args.skip_aggregation:
        print("\nAggregating results...")
        df_all = aggregate_results(args.output_dir, args.output_dir / NAME_AGGREGATED, ALL_PREFIXES)
        
        # Optimize: reuse df_all for sub-aggregations if possible
        if df_all is not None:
            print("Extracting GNN and HOPSE_M results from aggregated data...")
            df_gnn = df_all[df_all["model_type"].isin(PREFIXES_GNN)].copy()
            df_gnn.to_csv(args.output_dir / NAME_AGGREGATED_GNN, index=False)
            
            df_hopse_m = df_all[df_all["model_type"].isin(PREFIXES_HOPSE_M)].copy()
            df_hopse_m.to_csv(args.output_dir / NAME_AGGREGATED_HOPSE_M, index=False)

            print("\nFinding best runs...")
            save_best_runs(df_all, SUMMARIES_DIR / NAME_BEST_RUNS, "combined")
            save_best_runs(df_gnn, SUMMARIES_DIR / NAME_BEST_RUNS_GNN, "gnn_baseline")
            save_best_runs(df_hopse_m, SUMMARIES_DIR / NAME_BEST_RUNS_HOPSE_M, "hopse_m")
            
            del df_all, df_gnn, df_hopse_m
            gc.collect()

if __name__ == "__main__":
    main()
