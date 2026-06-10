import argparse
import os
import signal
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch_geometric.utils import to_networkx
from tqdm import tqdm

try:
    from topobench.data.loaders.graph import TUDatasetLoader
    from topobench.data.preprocessor import PreProcessor
    from topobench.data.utils.utils import get_routes_from_neighborhoods
except (ImportError, RuntimeError):
    TUDatasetLoader = None
    PreProcessor = None
    get_routes_from_neighborhoods = None
from torch_geometric.data import Data

# try:
#     from GraphRicciCurvature.OllivierRicci import OllivierRicci
# except ImportError:
#     OllivierRicci = None


class TimeoutException(Exception):
    """Exception raised when a computation exceeds the timeout."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException


def calculate_structural_metrics(G, debug=False, enabled_metrics=None):
    """Calculate structural metrics for a given NetworkX graph."""
    metrics = {}
    timings = {}
    n_nodes = G.number_of_nodes()
    
    all_struct = [
        "spectral_gap", "spectral_radius", "mean_ricci", "ricci_variance",
        "degree_assortativity", "effective_diameter", "algebraic_connectivity",
        "kirchhoff_index", "clustering_coefficient"
    ]
    if enabled_metrics is None:
        enabled_metrics = set(all_struct)

    if n_nodes < 2:
        return {m: None for m in all_struct if m in enabled_metrics}, {}

    # Identify Largest Connected Component (LCC) for distance-based metrics
    t0_comp = time.perf_counter()
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    LCC = G.subgraph(components[0])
    n_lcc = LCC.number_of_nodes()
    timings["component_split"] = time.perf_counter() - t0_comp

    # Adjacency eigenvalues (Full Graph)
    if "spectral_gap" in enabled_metrics or "spectral_radius" in enabled_metrics:
        t0 = time.perf_counter()
        adj_matrix = nx.to_numpy_array(G)
        adj_eigenvalues = np.linalg.eigvalsh(adj_matrix)
        if "spectral_radius" in enabled_metrics:
            metrics["spectral_radius"] = float(np.max(np.abs(adj_eigenvalues)))
        
        if "spectral_gap" in enabled_metrics:
            unique_evals = np.unique(np.round(adj_eigenvalues, decimals=8))
            if len(unique_evals) >= 2:
                metrics["spectral_gap"] = float(unique_evals[-1] - unique_evals[-2])
            else:
                metrics["spectral_gap"] = None
        timings["spectral_adj"] = time.perf_counter() - t0

    # Ollivier-Ricci (on LCC)
    # if "mean_ricci" in enabled_metrics or "ricci_variance" in enabled_metrics:
    #     t0 = time.perf_counter()
    #     if OllivierRicci is not None and n_lcc > 1:
    #         try:
    #             orc = OllivierRicci(LCC, alpha=0.5, verbose="ERROR")
    #             orc.compute_ricci_curvature()
    #             curvatures = [data["ricciCurvature"] for _, _, data in orc.G.edges(data=True)]
    #             if curvatures:
    #                 if "mean_ricci" in enabled_metrics: metrics["mean_ricci"] = float(np.mean(curvatures))
    #                 if "ricci_variance" in enabled_metrics: metrics["ricci_variance"] = float(np.var(curvatures))
    #             else:
    #                 if "mean_ricci" in enabled_metrics: metrics["mean_ricci"] = None
    #                 if "ricci_variance" in enabled_metrics: metrics["ricci_variance"] = None
    #         except Exception:
    #             if "mean_ricci" in enabled_metrics: metrics["mean_ricci"] = None
    #             if "ricci_variance" in enabled_metrics: metrics["ricci_variance"] = None
    #     else:
    #         if "mean_ricci" in enabled_metrics: metrics["mean_ricci"] = None
    #         if "ricci_variance" in enabled_metrics: metrics["ricci_variance"] = None
    #     timings["ricci_curvature"] = time.perf_counter() - t0

    # Degree Assortativity
    if "degree_assortativity" in enabled_metrics:
        t0 = time.perf_counter()
        try:
            val = nx.degree_assortativity_coefficient(G)
            metrics["degree_assortativity"] = float(val) if not np.isnan(val) else None
        except Exception:
            metrics["degree_assortativity"] = None
        timings["degree_assortativity"] = time.perf_counter() - t0

    # Effective Diameter (On LCC)
    if "effective_diameter" in enabled_metrics:
        t0 = time.perf_counter()
        if n_lcc > 1:
            lengths = dict(nx.all_pairs_shortest_path_length(LCC))
            all_lengths = [l for src in lengths for l in lengths[src].values() if l > 0]
            metrics["effective_diameter"] = float(np.percentile(all_lengths, 90)) if all_lengths else None
        else:
            metrics["effective_diameter"] = None
        timings["effective_diameter"] = time.perf_counter() - t0

    # Algebraic Connectivity
    if "algebraic_connectivity" in enabled_metrics:
        t0 = time.perf_counter()
        try:
            L = nx.normalized_laplacian_matrix(G).toarray()
            evals = np.linalg.eigvalsh(L)
            nz_evals = evals[evals > 1e-10]
            metrics["algebraic_connectivity"] = float(nz_evals[0]) if len(nz_evals) > 0 else None
        except Exception:
            metrics["algebraic_connectivity"] = None
        timings["algebraic_connectivity"] = time.perf_counter() - t0

    # Kirchhoff Index (On LCC)
    if "kirchhoff_index" in enabled_metrics:
        t0 = time.perf_counter()
        if n_lcc > 1:
            L_lcc = nx.laplacian_matrix(LCC).toarray()
            evals = np.linalg.eigvalsh(L_lcc)
            nz_evals = evals[evals > 1e-10]
            metrics["kirchhoff_index"] = float(n_lcc * np.sum(1.0 / nz_evals)) if len(nz_evals) > 0 else None
        else:
            metrics["kirchhoff_index"] = None
        timings["kirchhoff_index"] = time.perf_counter() - t0

    # Clustering Coefficient
    if "clustering_coefficient" in enabled_metrics:
        t0 = time.perf_counter()
        val = nx.average_clustering(G)
        metrics["clustering_coefficient"] = float(val) if not np.isnan(val) else None
        timings["clustering_coefficient"] = time.perf_counter() - t0

    return metrics, timings


def calculate_feature_metrics(G, x, y, debug=False, enabled_metrics=None):
    """Calculate feature-based metrics."""
    metrics = {}
    timings = {}
    n_nodes = G.number_of_nodes()

    if enabled_metrics is None:
        enabled_metrics = {"dirichlet_energy", "adjusted_homophily"}

    if n_nodes < 2 or x is None:
        return {
            m: 0.0
            for m in ["dirichlet_energy", "adjusted_homophily"]
            if m in enabled_metrics
        }, {}

    if "dirichlet_energy" in enabled_metrics:
        t0 = time.perf_counter()
        L = nx.laplacian_matrix(G).toarray()
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)
        dirichlet = np.trace(x_np.T @ L @ x_np)
        metrics["dirichlet_energy"] = float(dirichlet)
        timings["dirichlet_energy"] = time.perf_counter() - t0

    if "adjusted_homophily" in enabled_metrics:
        t0 = time.perf_counter()
        if y is not None:
            y_np = y.detach().cpu().numpy()
            if y_np.size == n_nodes:
                edge_index = np.array(list(G.edges())).T
                if edge_index.size > 0:
                    src_labels = y_np[edge_index[0]]
                    dst_labels = y_np[edge_index[1]]
                    h = np.mean(src_labels == dst_labels)
                    _, counts = np.unique(y_np, return_counts=True)
                    proportions = counts / n_nodes
                    sum_p_squared = np.sum(proportions**2)
                    if sum_p_squared < 1.0:
                        h_adj = (h - sum_p_squared) / (1.0 - sum_p_squared)
                        metrics["adjusted_homophily"] = float(h_adj)
                    else:
                        metrics["adjusted_homophily"] = 0.0
                else:
                    metrics["adjusted_homophily"] = 0.0
            else:
                metrics["adjusted_homophily"] = None
        else:
            metrics["adjusted_homophily"] = None
        timings["adjusted_homophily"] = time.perf_counter() - t0

    return metrics, timings


def interrank_boundary_index(x_src, boundary_index, n_dst_nodes):
    node_ids = (
        boundary_index[0]
        if torch.is_tensor(boundary_index[0])
        else torch.tensor(boundary_index[0], dtype=torch.int32)
    )
    edge_ids = (
        boundary_index[1]
        if torch.is_tensor(boundary_index[1])
        else torch.tensor(boundary_index[1], dtype=torch.int32)
    )
    max_node_id = n_dst_nodes
    adjusted_edge_ids = edge_ids + max_node_id
    edge_index = torch.zeros((2, node_ids.numel()), dtype=node_ids.dtype)
    edge_index[0, :] = node_ids
    edge_index[1, :] = adjusted_edge_ids
    edge_attr = x_src[edge_ids].squeeze()
    return edge_index, edge_attr


def intrarank_expand(params, src_rank, nbhd):
    neighborhood = getattr(params, nbhd).coalesce()
    batch_route = Data(
        x=getattr(params, f"x_{src_rank}"),
        edge_index=neighborhood.indices(),
        edge_weight=neighborhood.values().squeeze(),
        edge_attr=neighborhood.values().squeeze(),
        requires_grad=True,
    )
    return batch_route


def interrank_expand(params, src_rank, dst_rank, nbhd_cache, membership):
    src_batch = membership[src_rank]
    dst_batch = membership[dst_rank]
    edge_index, edge_attr = nbhd_cache
    device = getattr(params, f"x_{src_rank}").device
    feat_on_dst = torch.zeros_like(getattr(params, f"x_{dst_rank}"))
    x_in = torch.vstack([feat_on_dst, getattr(params, f"x_{src_rank}")])
    batch_expanded = torch.cat([dst_batch, src_batch], dim=0)
    batch_route = Data(
        x=x_in,
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device),
        edge_weight=edge_attr.to(device),
        batch=batch_expanded.to(device),
    )
    return batch_route


def scorer(data_list, neighborhoods, routes, debug=False, enabled_metrics=None):
    """Aggregate metrics across all graphs in the dataset using running stats."""
    # running_stats[nbhd][metric] = {min, max, sum, count}
    stats = defaultdict(lambda: defaultdict(lambda: {"min": float('inf'), "max": float('-inf'), "sum": 0.0, "count": 0}))
    cumulative_timings = defaultdict(float)

    # Set up timeout signal (only works on Linux/Unix)
    signal.signal(signal.SIGALRM, timeout_handler)

    pbar = tqdm(data_list, desc="Scoring graphs", leave=False)
    for idx, data in enumerate(pbar):
        max_r_val = max([max(r) for r in routes])
        membership = {
            j: torch.zeros(
                getattr(data, f"x_{j}").shape[0], dtype=torch.long
            )
            for j in range(max_r_val + 1)
        }
        
        # Pre-cache interrank neighborhoods
        nbhd_cache = {}
        for neighborhood, route in zip(neighborhoods, routes):
            src_rank, dst_rank = route
            if src_rank != dst_rank and (src_rank, dst_rank) not in nbhd_cache and hasattr(data, neighborhood):
                n_dst_nodes = getattr(data, f"x_{dst_rank}").shape[0]
                if n_dst_nodes > 0:
                    if src_rank > dst_rank:
                        boundary = getattr(data, neighborhood).coalesce()
                        nbhd_cache[(src_rank, dst_rank)] = interrank_boundary_index(getattr(data, f"x_{src_rank}"), boundary.indices(), n_dst_nodes)
                    elif src_rank < dst_rank:
                        coboundary = getattr(data, neighborhood).coalesce()
                        nbhd_cache[(src_rank, dst_rank)] = interrank_boundary_index(getattr(data, f"x_{src_rank}"), coboundary.indices(), n_dst_nodes)

        for nbhd, route in zip(neighborhoods, routes):
            if not hasattr(data, nbhd):
                continue
            
            src_rank, dst_rank = route
            if src_rank == dst_rank:
                hg = intrarank_expand(data, src_rank, nbhd)
            else:
                cache = nbhd_cache.get((src_rank, dst_rank))
                hg = interrank_expand(data, src_rank, dst_rank, cache, membership) if cache else None
            
            if hg is not None:
                # Apply timeout for each graph's computation
                signal.alarm(60) # 1 minute timeout
                try:
                    t0_nx = time.perf_counter()
                    G = to_networkx(hg, to_undirected=True)
                    t_nx = time.perf_counter() - t0_nx
                    cumulative_timings["to_networkx"] += t_nx

                    s_metrics, s_timings = calculate_structural_metrics(G, debug, enabled_metrics)
                    f_metrics, f_timings = calculate_feature_metrics(G, hg.x, data.y, debug, enabled_metrics)
                    
                    signal.alarm(0) # Disable alarm after success

                    # Update running stats
                    combined = {**s_metrics, **f_metrics}
                    for m_name, val in combined.items():
                        if val is not None:
                            curr = stats[nbhd][m_name]
                            curr["min"] = min(curr["min"], val)
                            curr["max"] = max(curr["max"], val)
                            curr["sum"] += val
                            curr["count"] += 1

                    for k, v in s_timings.items(): cumulative_timings[k] += v
                    for k, v in f_timings.items(): cumulative_timings[k] += v

                    if debug:
                        total_g = t_nx + sum(s_timings.values()) + sum(f_timings.values())
                        if total_g > 0.1:
                            slowest = sorted({**s_timings, **f_timings, "to_networkx": t_nx}.items(), key=lambda x: x[1], reverse=True)[:3]
                            pbar.write(f"[DEBUG] Graph {idx} | Nbhd {nbhd} | Time {total_g:.4f}s | Top: {', '.join([f'{k}: {v:.4f}s' for k, v in slowest])}")

                except TimeoutException:
                    pbar.write(f"[WARNING] Timeout processing graph {idx} for neighborhood {nbhd}. Skipping.")
                    continue
                except Exception as e:
                    pbar.write(f"[ERROR] Error processing graph {idx} for neighborhood {nbhd}: {e}")
                    continue
                finally:
                    signal.alarm(0)

    if debug:
        print("\n--- Timing report (Cumulative per metric) ---")
        sorted_timings = sorted(cumulative_timings.items(), key=lambda x: x[1], reverse=True)
        total_time = sum(cumulative_timings.values())
        if total_time > 0:
            for metric, t in sorted_timings:
                print(f"{metric:25} | {t:10.4f}s | {100*t/total_time:6.2f}%")
        print("--------------------------------------------")

    # Finalize summary
    summary = {}
    for nbhd, m_stats in stats.items():
        summary[nbhd] = {}
        for m_name, s in m_stats.items():
            if s["count"] > 0:
                summary[nbhd][m_name] = {
                    "min": float(s["min"]),
                    "max": float(s["max"]),
                    "mean": float(s["sum"] / s["count"]),
                }
    return summary


def main():
    print("Starting main execution...")
    parser = argparse.ArgumentParser(description="Expand complexes into Hasse graphs with timeout and checkpointing.")
    parser.add_argument("--output_csv", type=str, default="correlation/runs_summaries/best_runs_with_hasse_metrics.csv", help="Output final merged CSV file.")
    parser.add_argument("--checkpoint_csv", type=str, default="correlation/runs_summaries/hasse_metrics_checkpoints.csv", help="Sidecar CSV for progress.")
    parser.add_argument("--neighborhoods", type=str, default="[up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,down_adjacency-2,2-down_adjacency-2,up_incidence-0,up_incidence-1,2-up_incidence-0,down_incidence-1,down_incidence-2,2-down_incidence-2]", help="Neighborhoods list.")
    parser.add_argument("--metrics", type=str, default="all_except_ricci", help="Comma-separated list of metrics or 'all', 'all_except_ricci'.")
    parser.add_argument("--debug", action="store_true", help="Print timing information.")
    parser.add_argument("--force", action="store_true", help="Force recalculation of metrics even if in checkpoint.")
    args = parser.parse_args()

    # Define source paths
    gnn_summary_path = "correlation/runs_summaries/best_runs_summary_gnn.csv"
    hopse_m_summary_path = "correlation/runs_summaries/best_runs_summary_hopse_m.csv"
    # Note: topotune is often in the main summary or can be extracted from aggregated
    aggregated_path = "correlation/wandb_runs/aggregated_results.csv"

    if not all(os.path.exists(p) for p in [gnn_summary_path, hopse_m_summary_path, aggregated_path]):
        print("Required input CSV files not found.")
        return

    # 1. Construct the target summary_df
    print("Restructuring summary data (5 lines per dataset, 3 for Mantra)...")
    gnn_df = pd.read_csv(gnn_summary_path)
    hopse_m_df = pd.read_csv(hopse_m_summary_path)
    agg_df = pd.read_csv(aggregated_path)
    
    # Extract Topotune from aggregated if not in a separate summary
    topotune_agg = agg_df[agg_df["model_type"] == "topotune"].copy()
    
    # Function to get best run from a group
    def get_best_from_agg(group, suffix=None):
        # Determine metric based on dataset name from first row
        ds_name = group["dataset_name"].iloc[0]
        if suffix:
            metric_base, goal = f"f1-{suffix}", "max"
        elif any(x in ds_name for x in ["Clearance_Hepatocyte_AZ", "Caco2_Wang"]):
            metric_base, goal = "mae", "min"
        elif "mantra" in ds_name:
            metric_base, goal = "f1", "max"
        else:
            metric_base, goal = "accuracy", "max"
            
        val_col = f"summary.val_best_rerun/{metric_base}_mean"
        test_col = f"summary.test_best_rerun/{metric_base}_mean"
        test_std = f"summary.test_best_rerun/{metric_base}_std"
        
        if val_col not in group.columns: return None
        
        sub = group.dropna(subset=[val_col]).copy()
        if sub.empty: return None
        
        best_idx = sub[val_col].idxmin() if goal == "min" else sub[val_col].idxmax()
        row = sub.loc[best_idx].copy()
        # Clean up score names for consistency
        row["score_name"] = "f1" if "f1" in metric_base else metric_base
        row["score"], row["score_std"] = row[test_col], row[test_std]
        row["val_score"] = row[val_col]
        return row

    topotune_rows = []
    for (ds, dom), group in topotune_agg.groupby(["dataset_name", "config.model.model_domain"], dropna=False):
        # Mantra is simplicial only
        if "mantra" in ds.lower() and dom != "simplicial": continue
        if dom not in ["cell", "simplicial"]: continue
        
        if "betti_numbers" in ds:
            for suffix in ["1", "2"]:
                best_row = get_best_from_agg(group, suffix=suffix)
                if best_row is not None:
                    best_row["dataset_name"] = f"mantra_betti_number_{suffix}"
                    best_row["model_category"] = f"topotune_{dom}"
                    topotune_rows.append(best_row)
        else:
            best_row = get_best_from_agg(group)
            if best_row is not None:
                best_row["model_category"] = f"topotune_{dom}"
                topotune_rows.append(best_row)
    
    topotune_df = pd.DataFrame(topotune_rows)
    
    # Combine everything
    # We want: Best GNN (1), Topotune (2), Hopse_m (2)
    summary_df = pd.concat([gnn_df, topotune_df, hopse_m_df], ignore_index=True)
    
    # Drop cocitation/ZINC as in download_runs.py
    summary_df = summary_df[~summary_df["dataset_name"].str.contains("cocitation|ZINC", case=False, regex=True)]
    
    # Ensure 'domain' is correctly mapped for metric lookup
    # 1. Start from config.model.model_domain if available
    if "config.model.model_domain" in summary_df.columns:
        summary_df["domain"] = summary_df["config.model.model_domain"]
    else:
        summary_df["domain"] = "cell"

    # 2. Refine mapping
    summary_df["is_mantra_flag"] = summary_df["dataset_name"].str.contains("mantra", case=False)
    # Mantra always simplicial
    summary_df.loc[summary_df["is_mantra_flag"], "domain"] = "simplicial"
    # Others: anything not simplicial is cell
    summary_df.loc[(~summary_df["is_mantra_flag"]) & (summary_df["domain"] != "simplicial"), "domain"] = "cell"
    summary_df["domain"] = summary_df["domain"].fillna("cell")
    summary_df = summary_df.drop(columns=["is_mantra_flag"])

    dataset_names = summary_df["dataset_name"].unique().tolist()

    nbhd_str = args.neighborhoods.strip().strip("[]")
    neighborhoods = [n.strip() for n in nbhd_str.split(",") if n.strip()]
    routes = get_routes_from_neighborhoods(neighborhoods)

    all_possible_metrics = {
        "spectral_gap", "spectral_radius", "mean_ricci", "ricci_variance",
        "degree_assortativity", "effective_diameter", "algebraic_connectivity",
        "kirchhoff_index", "clustering_coefficient", "dirichlet_energy", "adjusted_homophily"
    }
    if args.metrics == "all": enabled_metrics = all_possible_metrics
    elif args.metrics == "all_except_ricci": enabled_metrics = all_possible_metrics - {"mean_ricci", "ricci_variance"}
    else:
        enabled_metrics = {m.strip() for m in args.metrics.split(",")} & all_possible_metrics

    # Checkpoint loading
    if os.path.exists(args.checkpoint_csv):
        checkpoint_df = pd.read_csv(args.checkpoint_csv)
        if "domain" not in checkpoint_df.columns:
            checkpoint_df["domain"] = "cell"
        checkpoint_df["domain"] = checkpoint_df["domain"].fillna("cell")

        # Reorder columns to make domain the second one
        cols = checkpoint_df.columns.tolist()
        if "domain" in cols:
            cols.remove("domain")
            cols.insert(1, "domain")
            checkpoint_df = checkpoint_df[cols]

        processed_datasets = set(checkpoint_df["dataset_name"].unique())
        print(f"Resuming from checkpoint. Already processed: {processed_datasets}")
    else:
        checkpoint_df = pd.DataFrame(columns=["dataset_name", "domain"])
        processed_datasets = set()

    # Identify unique graph names for Mantra (to avoid redundant scoring)
    processed_mantra = {"cell": False, "simplicial": False}
    if not checkpoint_df.empty:
        for d in ["cell", "simplicial"]:
            if ((checkpoint_df["domain"] == d) & (checkpoint_df["dataset_name"].str.contains("mantra", case=False))).any():
                processed_mantra[d] = True
                print(f"Mantra already processed for domain: {d}")

    for data_name in tqdm(dataset_names, desc="Datasets"):
        # Get domains present for this dataset in summary_df
        relevant_rows = summary_df[summary_df["dataset_name"] == data_name]
        is_mantra = "mantra" in data_name.lower()
        
        # Determine target domains: map anything not 'simplicial' to 'cell'
        if "config.model.model_domain" in relevant_rows.columns:
            raw_domains = relevant_rows["config.model.model_domain"].unique().tolist()
        else:
            raw_domains = ["cell"]
        
        target_domains = set()
        for d in raw_domains:
            if d == "simplicial":
                target_domains.add("simplicial")
            else:
                # For Mantra, we skip 'cell' domain since it's naturally simplicial
                if not is_mantra:
                    target_domains.add("cell")
        
        # If Mantra somehow has no simplicial rows but is present, ensure we only target simplicial
        if is_mantra:
            target_domains = {"simplicial"}

        for domain in target_domains:
            # Check if (data_name, domain) already in checkpoint_df
            if not args.force and not checkpoint_df.empty and \
               ((checkpoint_df["dataset_name"] == data_name) & (checkpoint_df["domain"] == domain)).any():
                continue

            # Check for Mantra redundancy per domain
            if not args.force and is_mantra and processed_mantra.get(domain):
                continue

            print(f"\nProcessing {data_name} (domain: {domain})...")
            
            # Setup lifting: For Mantra, we skip the lifting transform because it's already a simplicial complex
            if is_mantra:
                transforms_config = None 
            elif domain == "cell":
                transforms_config = OmegaConf.create({
                    "cycle_lifting": {"transform_type": "lifting", "transform_name": "CellCycleLifting", "complex_dim": 2, "neighborhoods": neighborhoods}
                })
            else:
                transforms_config = OmegaConf.create({
                    "clique_lifting": {"transform_type": "lifting", "transform_name": "SimplicialCliqueLifting", "complex_dim": 2, "neighborhoods": neighborhoods}
                })

            # Dispatch loader
            loader_kwargs = {"data_name": data_name, "data_dir": "./datasets/graph/TUDataset"}
            if data_name in ["BBB_Martins", "CYP3A4_Veith", "Caco2_Wang", "Clearance_Hepatocyte_AZ", "PAMPA_NCATS", "HIA_Hou", "Pgp_Broccatelli", "Bioavailability_Ma", "CYP1A2_Veith", "CYP2C19_Veith", "CYP2D6_Veith", "CYP2C9_Veith", "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels", "CYP3A4_Substrate_CarbonMangels", "Lipophilicity_AstraZeneca", "Solubility_AqSolDB", "HydrationFreeEnergy_FreeSolv", "PPBR_AZ", "VDss_Lombardo", "Half_Life_Obach", "Clearance_Microsome_AZ"]:
                from topobench.data.loaders.graph.adme_datasets import ADMEDatasetLoader
                loader_class, data_dir = ADMEDatasetLoader, "./datasets/graph/ADME"
                loader_kwargs["data_dir"] = data_dir
            elif is_mantra:
                from topobench.data.loaders.graph.mantra_dataset import MantraSimplicialDatasetLoader
                loader_class, data_dir = MantraSimplicialDatasetLoader, "./datasets/graph/MANTRA"
                loader_kwargs["data_dir"] = data_dir
                loader_kwargs["manifold_dim"] = 2 
                loader_kwargs["version"] = "v0.0.5"
                # Extract task variable from name (e.g., mantra_betti_number_1 -> betti_numbers)
                if "betti" in data_name:
                    loader_kwargs["task_variable"] = "betti_numbers"
                elif "name" in data_name:
                    loader_kwargs["task_variable"] = "name"
                elif "orientation" in data_name:
                    loader_kwargs["task_variable"] = "orientation"
                else:
                    loader_kwargs["task_variable"] = "betti_numbers"
            else:
                from topobench.data.loaders.graph.tu_datasets import TUDatasetLoader
                loader_class, data_dir = TUDatasetLoader, "./datasets/graph/TUDataset"

            try:
                dataset, dataset_dir = loader_class(OmegaConf.create(loader_kwargs)).load()
                preprocessor = PreProcessor(dataset=dataset, data_dir=dataset_dir, transforms_config=transforms_config)
                
                dataset_metrics = scorer(preprocessor, neighborhoods, routes, args.debug, enabled_metrics)
                
                # Save results (for Mantra, copy to all variants in the same domain)
                names_to_save = [data_name]
                if is_mantra:
                    names_to_save = [n for n in dataset_names if "mantra" in n.lower()]
                    processed_mantra[domain] = True

                for name in names_to_save:
                    row = {"dataset_name": name, "domain": domain}
                    for nbhd, m_data in dataset_metrics.items():
                        for m_name, stats in m_data.items():
                            for stype, val in stats.items():
                                row[f"{nbhd}_{m_name}_{stype}"] = val
                    
                    new_row_df = pd.DataFrame([row])
                    # Filter out existing entries for this name and domain to avoid duplicates
                    if not checkpoint_df.empty:
                        checkpoint_df = checkpoint_df[~((checkpoint_df["dataset_name"] == name) & (checkpoint_df["domain"] == domain))]
                    checkpoint_df = pd.concat([checkpoint_df, new_row_df], ignore_index=True)
                
                checkpoint_df.to_csv(args.checkpoint_csv, index=False)
                print(f"Dataset result saved for: {names_to_save} ({domain})")
                
            except Exception as e:
                print(f"Error processing dataset {data_name} ({domain}): {e}")
                import traceback
                traceback.print_exc()
                continue

    # Final Merge
    merged_df = pd.merge(summary_df, checkpoint_df, on=["dataset_name", "domain"], how="left")
    merged_df.to_csv(args.output_csv, index=False)
    print(f"\nFinal merged results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
