"""
Interactive Hypergraph Neighborhood Explorer

Graph views open in the default browser as a standalone D3 page.

Run with: streamlit run analysis/neighborhood_explorer_app.py
"""

import sys
import os
import copy
import re
import math
import platform
import shutil
import subprocess
import tempfile
import uuid
import webbrowser
from pathlib import Path
import yaml

# Add parent directory to path for imports
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import torch

from d3_graph_html import build_standalone_d3_html
from omegaconf import OmegaConf
from torch_geometric.utils import to_undirected

# ============================================================================
# Dataset Discovery Functions
# ============================================================================

@st.cache_resource
def discover_available_datasets():
    """Scan configs/dataset folder and discover all available datasets."""
    datasets_by_domain = {}
    config_dir = Path(__file__).parent.parent / "configs" / "dataset"
    
    if not config_dir.exists():
        return datasets_by_domain
    
    # Scan each domain folder
    for domain_folder in config_dir.iterdir():
        if not domain_folder.is_dir():
            continue
        
        domain_name = domain_folder.name
        datasets_by_domain[domain_name] = []
        
        # Scan all yaml files in the domain folder
        for yaml_file in domain_folder.glob("*.yaml"):
            dataset_name = yaml_file.stem
            if dataset_name != "manual_dataset":  # Skip special files
                datasets_by_domain[domain_name].append(dataset_name)
        
        # Sort dataset names
        datasets_by_domain[domain_name].sort()
    
    return datasets_by_domain

@st.cache_resource
def discover_available_liftings():
    """Scan configs/transforms/liftings folder and discover all available liftings, grouped by source domain."""
    liftings_by_source = {}
    liftings_dir = Path(__file__).parent.parent / "configs" / "transforms" / "liftings"

    if not liftings_dir.exists():
        return liftings_by_source

    for subfolder in sorted(liftings_dir.iterdir()):
        if not subfolder.is_dir():
            continue

        folder_name = subfolder.name  # e.g. "graph2hypergraph"
        parts = folder_name.split("2")
        if len(parts) != 2:
            continue

        source_domain, target_domain = parts[0], parts[1]

        if source_domain not in liftings_by_source:
            liftings_by_source[source_domain] = []

        for yaml_file in sorted(subfolder.glob("*.yaml")):
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
            transform_name = config.get('transform_name', yaml_file.stem)
            liftings_by_source[source_domain].append({
                'name': transform_name,
                'file': yaml_file.stem,
                'source': source_domain,
                'target': target_domain,
                'config_path': str(yaml_file),
                'config': config,
            })

    return liftings_by_source


def load_dataset_config(domain, dataset_name):
    """Load the yaml config for a specific dataset and resolve interpolations."""
    config_path = Path(__file__).parent.parent / "configs" / "dataset" / domain / f"{dataset_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict

# ============================================================================
# Configuration and Constants
# ============================================================================

st.set_page_config(
    page_title="Hypergraph Neighborhood Explorer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ~2× Streamlit’s default sidebar width (~256px in recent releases).
_SIDEBAR_TARGET_WIDTH_PX = 512
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] {{
        min-width: {_SIDEBAR_TARGET_WIDTH_PX}px;
        max-width: {_SIDEBAR_TARGET_WIDTH_PX}px;
        width: {_SIDEBAR_TARGET_WIDTH_PX}px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_RANK_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]

def _data_keys(data):
    """Best-effort key extraction across PyG Data-like objects."""
    try:
        keys = list(data.keys())
    except Exception:
        keys = []
    if not keys:
        keys = [
            k for k in dir(data)
            if not k.startswith("_") and not callable(getattr(data, k, None))
        ]
    return keys


def get_rank_labels(domain, dataset_name, data):
    """
    Rank names from dataset config when present, else generic labels.

    Supported config shapes:
    - rank_labels: {0: "...", 1: "..."}
    - metadata.rank_labels: {...}
    - parameters.rank_labels: {...}
    """
    labels = {}
    cfg = {}
    try:
        cfg = load_dataset_config(domain, dataset_name) or {}
    except Exception:
        cfg = {}

    rank_labels_cfg = (
        cfg.get("rank_labels")
        or cfg.get("metadata", {}).get("rank_labels")
        or cfg.get("parameters", {}).get("rank_labels")
        or {}
    )
    if isinstance(rank_labels_cfg, dict):
        for k, v in rank_labels_cfg.items():
            try:
                labels[int(k)] = str(v)
            except Exception:
                continue

    seen_ranks = {0}
    for key in _data_keys(data):
        m = re.match(r"^(?:incidence|adjacency|coadjacency|up_laplacian|down_laplacian|hodge_laplacian)_(\d+)$", key)
        if not m:
            continue
        rk = int(m.group(1))
        seen_ranks.add(rk)
        if key.startswith("incidence_"):
            seen_ranks.add(max(rk - 1, 0))
    for rank in sorted(seen_ranks):
        labels.setdefault(rank, f"Rank {rank}")
    return labels


def rank_color(rank):
    """Stable color assignment per rank."""
    if rank is None:
        return "#888888"
    return DEFAULT_RANK_PALETTE[int(rank) % len(DEFAULT_RANK_PALETTE)]

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_dataset(domain, dataset_name):
    """Load a dataset by properly resolving config interpolations."""
    # Load the yaml config
    config_path = Path(__file__).parent.parent / "configs" / "dataset" / domain / f"{dataset_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    
    # Load raw YAML
    with open(config_path, 'r') as f:
        dataset_yaml = yaml.safe_load(f)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create base config with paths that can be interpolated
    base_config = {
        'paths': {
            'root_dir': str(project_root),
            'data_dir': str(project_root / 'datasets'),
            'log_dir': str(project_root / 'logs'),
        }
    }
    
    # Create OmegaConf and merge - this allows interpolation resolution
    cfg = OmegaConf.create(base_config)
    dataset_cfg = OmegaConf.create({'dataset': dataset_yaml})
    cfg = OmegaConf.merge(cfg, dataset_cfg)
    
    # Resolve all interpolations
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved_dataset = cfg_resolved['dataset']
    
    # Get the loader class name from _target_
    loader_target = resolved_dataset['loader'].get('_target_')
    if not loader_target:
        raise ValueError("No loader target found in config")
    
    class_name = loader_target.split('.')[-1]
    
    # Import the loader class
    from topobench.data import loaders as loaders_module
    loader_class = getattr(loaders_module, class_name)
    
    # Create OmegaConf config from resolved loader parameters
    loader_params = OmegaConf.create(resolved_dataset['loader']['parameters'])
    
    # Instantiate loader and load data
    loader = loader_class(loader_params)
    data, dataset_dir = loader.load()
    
    return data, domain


def edge_index_to_sparse_adj(data):
    """Undirected sparse adjacency from PyG ``edge_index`` (graph view)."""
    if not hasattr(data, "edge_index") or data.edge_index is None:
        return None
    ei = data.edge_index
    if ei.numel() == 0:
        return None
    num_nodes = data.num_nodes
    if num_nodes is None:
        num_nodes = int(ei.max().item()) + 1
    ei = to_undirected(ei.clone(), num_nodes=num_nodes)
    row, col = ei[0], ei[1]
    vals = torch.ones(row.size(0), dtype=torch.float32, device=row.device)
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        vals,
        (num_nodes, num_nodes),
        device=row.device,
    ).coalesce()
    return adj


def incidence_to_sparse_incidence(data):
    """Sparse node × hyperedge incidence if ``incidence_hyperedges`` is set."""
    inc = None
    if hasattr(data, "incidence_hyperedges") and data.incidence_hyperedges is not None:
        inc = data.incidence_hyperedges
    elif "incidence_hyperedges" in data:
        inc = data["incidence_hyperedges"]
    if inc is None:
        return None
    if inc.layout == torch.sparse_coo:
        return inc.coalesce()
    if torch.is_tensor(inc) and inc.dim() == 2 and inc.size(0) == 2:
        row, col = inc[0].long(), inc[1].long()
        num_nodes = data.num_nodes
        if num_nodes is None:
            num_nodes = int(row.max().item()) + 1
        num_he = getattr(data, "num_hyperedges", None)
        if num_he is None:
            num_he = int(col.max().item()) + 1
        vals = torch.ones(row.size(0), dtype=torch.float32, device=row.device)
        return torch.sparse_coo_tensor(
            torch.stack([row, col]),
            vals,
            (num_nodes, num_he),
            device=row.device,
        ).coalesce()
    return None


def _sparse_coo_nnz(sp) -> int:
    if sp is None:
        return 0
    sp = sp.coalesce()
    return int(sp.values().numel())


def _to_sparse_coo(matrix):
    """Coerce torch sparse / dense edge-index / SciPy sparse to coalesced COO."""
    if matrix is None:
        return None
    if hasattr(matrix, "layout") and matrix.layout == torch.sparse_coo:
        return matrix.coalesce()
    if torch.is_tensor(matrix) and matrix.dim() == 2 and matrix.size(0) == 2:
        row, col = matrix[0].long(), matrix[1].long()
        n0 = int(row.max().item()) + 1 if row.numel() else 0
        n1 = int(col.max().item()) + 1 if col.numel() else 0
        vals = torch.ones(row.size(0), dtype=torch.float32, device=row.device)
        return torch.sparse_coo_tensor(
            torch.stack([row, col]),
            vals,
            (n0, n1),
            device=row.device,
        ).coalesce()
    try:
        import scipy.sparse as sps  # type: ignore

        if sps.issparse(matrix):
            coo = matrix.tocoo()
            row = torch.as_tensor(coo.row, dtype=torch.long)
            col = torch.as_tensor(coo.col, dtype=torch.long)
            vals = torch.as_tensor(coo.data, dtype=torch.float32)
            shape = (int(coo.shape[0]), int(coo.shape[1]))
            return torch.sparse_coo_tensor(
                torch.stack([row, col]), vals, size=shape, device=vals.device
            ).coalesce()
    except Exception:
        pass
    return None


def _get_data_attr(data, key):
    """Best-effort attribute access on PyG ``Data``-like objects."""
    val = None
    if hasattr(data, key):
        val = getattr(data, key)
    if val is None:
        try:
            if key in data:
                val = data[key]
        except Exception:
            val = None
    return val


def incidence_rank_k_to_sparse(data, rank: int):
    """
    Sparse (k-1)-cell to k-cell incidence from ``get_complex_connectivity`` output.

    Accepts torch sparse COO, ``[2, nnz]`` index tensors, or SciPy sparse matrices.
    """
    return _to_sparse_coo(_get_data_attr(data, f"incidence_{rank}"))


def adjacency_rank_k_to_sparse(data, rank: int):
    """Sparse rank-k adjacency from ``get_complex_connectivity`` output."""
    return _to_sparse_coo(_get_data_attr(data, f"adjacency_{rank}"))


def incidence_rank_one_to_sparse(data):
    """Sparse 0-cell to 1-cell incidence (alias for :func:`incidence_rank_k_to_sparse`)."""
    return incidence_rank_k_to_sparse(data, 1)


def _incidence_ranks_present(data):
    ranks = []
    for key in _data_keys(data):
        m = re.match(r"^incidence_(\d+)$", key)
        if m:
            ranks.append(int(m.group(1)))
    return sorted(set(ranks), reverse=True)


def pick_primary_complex_incidence(data):
    """
    Choose a non-empty ``incidence_k`` for visualization.

    Graph liftings share the same 1-skeleton, so ``incidence_1`` is often identical
    across methods; higher ``k`` (faces, volumes, …) reflects the lift.
    Prefers the largest ``k >= 2`` with nonzeros, then falls back to ``k == 1``.
    """
    ranks = _incidence_ranks_present(data)
    for prefer_high in (True, False):
        for k in ranks:
            if prefer_high and k < 2:
                continue
            if not prefer_high and k != 1:
                continue
            sp = incidence_rank_k_to_sparse(data, k)
            if sp is None or _sparse_coo_nnz(sp) == 0:
                continue
            return (
                sp,
                f"Rank-{k - 1} to rank-{k} incidence",
                {"type": "bipartite", "source_rank": k - 1, "target_rank": k},
            )
    return None


def _connectivity_keys_present(data, kind: str):
    """Return ranks of ``{kind}_k`` keys present on ``data`` (sorted ascending)."""
    ranks = []
    pattern = re.compile(rf"^{kind}_(\d+)$")
    for key in _data_keys(data):
        m = pattern.match(key)
        if m:
            ranks.append(int(m.group(1)))
    return sorted(set(ranks))


def enumerate_neighborhoods(data):
    """
    List visualizable neighborhoods on a (possibly lifted) ``Data`` object.

    Empty matrices (zero nnz) are filtered out. Order: raw ``graph``,
    ``hyperedges``, ``incidence_k`` ascending, ``adjacency_k`` ascending.

    Returns
    -------
    list[dict]
        Items shaped like
        ``{"id": str, "label": str, "kind": str, "rank": int | None}``.
    """
    out = []

    adj = edge_index_to_sparse_adj(data)
    if adj is not None and _sparse_coo_nnz(adj) > 0:
        nnz = _sparse_coo_nnz(adj)
        out.append({
            "id": "graph",
            "label": f"graph — edge_index adjacency ({nnz:,} nnz)",
            "kind": "graph",
            "rank": None,
        })

    hyper = incidence_to_sparse_incidence(data)
    if hyper is not None and _sparse_coo_nnz(hyper) > 0:
        nnz = _sparse_coo_nnz(hyper)
        out.append({
            "id": "hyperedges",
            "label": f"incidence_hyperedges — Rank 0 → hyperedges ({nnz:,} nnz)",
            "kind": "hyperedges",
            "rank": None,
        })

    for k in _connectivity_keys_present(data, "incidence"):
        sp = incidence_rank_k_to_sparse(data, k)
        nnz = _sparse_coo_nnz(sp)
        if nnz == 0:
            continue
        out.append({
            "id": f"incidence_{k}",
            "label": f"incidence_{k} — Rank {k - 1} → Rank {k} ({nnz:,} nnz)",
            "kind": "incidence",
            "rank": k,
        })

    for k in _connectivity_keys_present(data, "adjacency"):
        sp = adjacency_rank_k_to_sparse(data, k)
        nnz = _sparse_coo_nnz(sp)
        if nnz == 0:
            continue
        out.append({
            "id": f"adjacency_{k}",
            "label": f"adjacency_{k} — Rank {k} ↔ Rank {k} ({nnz:,} nnz)",
            "kind": "adjacency",
            "rank": k,
        })

    return out


def get_named_visualization_matrix(data, neigh_id):
    """
    Resolve an explicit neighborhood id to ``(sparse, description, relation_ctx)``.

    Returns ``(None, None, None)`` if the id is not present on ``data``.
    """
    if not neigh_id:
        return None, None, None

    if neigh_id == "graph":
        adj = edge_index_to_sparse_adj(data)
        if adj is None:
            return None, None, None
        return (
            adj,
            "Graph (adjacency from edge_index)",
            {"type": "adjacency", "source_rank": 0},
        )

    if neigh_id == "hyperedges":
        inc = incidence_to_sparse_incidence(data)
        if inc is None:
            return None, None, None
        return (
            inc,
            "incidence_hyperedges (Rank 0 → hyperedges)",
            {"type": "bipartite", "source_rank": 0, "target_kind": "hyperedge"},
        )

    m = re.match(r"^incidence_(\d+)$", neigh_id)
    if m:
        k = int(m.group(1))
        sp = incidence_rank_k_to_sparse(data, k)
        if sp is None:
            return None, None, None
        return (
            sp,
            f"incidence_{k} (Rank {k - 1} → Rank {k})",
            {"type": "bipartite", "source_rank": max(k - 1, 0), "target_rank": k},
        )

    m = re.match(r"^adjacency_(\d+)$", neigh_id)
    if m:
        k = int(m.group(1))
        sp = adjacency_rank_k_to_sparse(data, k)
        if sp is None:
            return None, None, None
        return (
            sp,
            f"adjacency_{k} (Rank {k} ↔ Rank {k})",
            {"type": "adjacency", "source_rank": k},
        )

    return None, None, None


def pick_default_neighborhood_id(available):
    """
    Pick a sensible default from ``enumerate_neighborhoods`` output.

    Preference order: highest-rank ``incidence_k`` with k>=2, then
    ``incidence_1``, then ``graph``, then ``hyperedges``, then first item.
    """
    if not available:
        return None
    by_id = {n["id"]: n for n in available}

    incidence_ranks = sorted(
        (n["rank"] for n in available if n["kind"] == "incidence"),
        reverse=True,
    )
    for k in incidence_ranks:
        if k >= 2:
            return f"incidence_{k}"
    if any(n["id"] == "incidence_1" for n in available):
        return "incidence_1"
    if "graph" in by_id:
        return "graph"
    if "hyperedges" in by_id:
        return "hyperedges"
    return available[0]["id"]


def get_primary_visualization_matrix(data, mode):
    """
    Parameters
    ----------
    mode : str
        ``'auto'`` | ``'graph'`` | ``'incidence'``

    Returns
    -------
    tuple
        ``(sparse_tensor, description, relation_context)`` or ``(None, None, None)``.
    """
    adj = edge_index_to_sparse_adj(data)
    inc = incidence_to_sparse_incidence(data)
    complex_inc = pick_primary_complex_incidence(data)

    if mode == "graph":
        if adj is None:
            return None, None, None
        return (
            adj,
            "Graph (adjacency from pairwise edges)",
            {"type": "adjacency", "source_rank": 0},
        )
    if mode == "incidence":
        if inc is not None:
            ctx = (
                {"type": "bipartite", "source_rank": 0, "target_kind": "hyperedge"}
                if getattr(data, "num_hyperedges", None) is not None
                else {"type": "bipartite", "source_rank": 0, "target_rank": 1}
            )
            return inc, "Node–hyperedge incidence", ctx
        if complex_inc is not None:
            return complex_inc
        return None, None, None

    # auto: AbstractLifting merges lifted tensors into ``Data`` but keeps the
    # original ``edge_index``. Prefer lifted connectivity over raw graph edges.
    if mode == "auto":
        if inc is not None and getattr(data, "num_hyperedges", None) is not None:
            return (
                inc,
                "Node–hyperedge incidence (lifted)",
                {"type": "bipartite", "source_rank": 0, "target_kind": "hyperedge"},
            )
        if complex_inc is not None:
            return complex_inc
        if adj is not None:
            return (
                adj,
                "Graph (adjacency from pairwise edges)",
                {"type": "adjacency", "source_rank": 0},
            )
        if inc is not None:
            return (
                inc,
                "Node–hyperedge incidence",
                {"type": "bipartite", "source_rank": 0, "target_rank": 1},
            )
        return None, None, None

    return None, None, None


def get_matrix_stats(sparse_tensor):
    """Get statistics for a sparse matrix."""
    if sparse_tensor is None:
        return None
    
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    
    num_edges = len(values)
    shape = tuple(sparse_tensor.shape)
    max_possible = shape[0] * shape[1]
    sparsity = 1 - (num_edges / max_possible) if max_possible > 0 else 0
    
    # Compute degree statistics
    row_degrees = torch.zeros(shape[0])
    col_degrees = torch.zeros(shape[1])
    for i in range(len(indices[0])):
        row_degrees[indices[0][i]] += 1
        col_degrees[indices[1][i]] += 1
    
    return {
        'shape': shape,
        'num_edges': num_edges,
        'sparsity': sparsity,
        'density': 1 - sparsity,
        'avg_out_degree': row_degrees[row_degrees > 0].mean().item() if (row_degrees > 0).any() else 0,
        'max_out_degree': row_degrees.max().item(),
        'avg_in_degree': col_degrees[col_degrees > 0].mean().item() if (col_degrees > 0).any() else 0,
        'max_in_degree': col_degrees.max().item()
    }


# ============================================================================
# Graph Conversion Functions
# ============================================================================

def sparse_to_networkx(sparse_tensor, max_nodes=200, min_degree=0):
    """Convert sparse tensor to NetworkX graph with filtering."""
    if sparse_tensor is None:
        return None, {}
    
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices().numpy()
    n_edges = indices.shape[1]
    
    if n_edges == 0:
        return None, {}
    
    # Determine if adjacency (square) or incidence (rectangular)
    is_adjacency = sparse_tensor.shape[0] == sparse_tensor.shape[1]

    # Compute node degrees
    node_degrees = {}
    if is_adjacency:
        # Adjacency may contain both (u,v) and (v,u). Count each undirected pair once.
        seen_pairs = set()
        for i in range(n_edges):
            src, tgt = int(indices[0, i]), int(indices[1, i])
            if src == tgt:
                continue
            pair = (src, tgt) if src < tgt else (tgt, src)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            u, v = pair
            node_degrees[u] = node_degrees.get(u, 0) + 1
            node_degrees[v] = node_degrees.get(v, 0) + 1
    else:
        for i in range(n_edges):
            src, tgt = int(indices[0, i]), int(indices[1, i])
            node_degrees[src] = node_degrees.get(src, 0) + 1
            node_degrees[tgt] = node_degrees.get(tgt, 0) + 1
    
    # Filter by minimum degree
    if min_degree > 0:
        node_degrees = {k: v for k, v in node_degrees.items() if v >= min_degree}

    if not node_degrees:
        return None, {}

    # Sample top nodes if too many
    if len(node_degrees) > max_nodes:
        top_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)[:max_nodes]
        node_degrees = {k: node_degrees[k] for k in top_nodes}
    
    valid_nodes = set(node_degrees.keys())
    
    if is_adjacency:
        G = nx.Graph()
        for node in valid_nodes:
            G.add_node(node, degree=node_degrees[node])
        
        for i in range(n_edges):
            src, tgt = indices[0, i], indices[1, i]
            if src in valid_nodes and tgt in valid_nodes and src != tgt:
                G.add_edge(src, tgt)
    else:
        # Bipartite graph
        G = nx.DiGraph()
        src_nodes = set(indices[0])
        tgt_nodes = set(indices[1])
        
        for node in src_nodes:
            if node in valid_nodes:
                G.add_node(f"src_{node}", bipartite=0, original_id=node)
        for node in tgt_nodes:
            if node in valid_nodes:
                G.add_node(f"tgt_{node}", bipartite=1, original_id=node)
        
        for i in range(n_edges):
            src, tgt = indices[0, i], indices[1, i]
            src_key = f"src_{src}"
            tgt_key = f"tgt_{tgt}"
            if src_key in G.nodes() and tgt_key in G.nodes():
                G.add_edge(src_key, tgt_key)
    
    return G, node_degrees


# ============================================================================
# D3 graph (standalone HTML, opens in browser)
# ============================================================================


def _json_safe_float(x) -> float:
    """Finite Python float for JSON (avoids NaN/Inf with allow_nan=False)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return v


def networkx_to_d3_payload(
    G,
    node_degrees,
    *,
    neighborhood_str=None,
    rank_labels=None,
    relation_context=None,
    plot_title=None,
    plot_subtitle=None,
    selected_node=None,
):
    """Build a JSON-serializable payload for :func:`build_standalone_d3_html`."""
    if G is None or len(G.nodes()) == 0:
        return None

    if rank_labels is None:
        rank_labels = {}

    if plot_title is not None:
        is_adjacency = not any(str(n).startswith("src_") for n in G.nodes())
        src_rank = (relation_context or {}).get("source_rank", 0)
        target_rank = (relation_context or {}).get(
            "target_rank", 1 if not is_adjacency else 0
        )
        target_kind = (relation_context or {}).get(
            "target_kind", "rank" if not is_adjacency else None
        )
        title = plot_title
        subtitle = plot_subtitle or ""
    else:
        is_adjacency = not any(str(n).startswith("src_") for n in G.nodes())
        src_rank = (relation_context or {}).get("source_rank", 0)
        target_rank = (relation_context or {}).get(
            "target_rank", 1 if not is_adjacency else 0
        )
        target_kind = (relation_context or {}).get(
            "target_kind", "rank" if not is_adjacency else None
        )
        title = neighborhood_str or "Graph"
        subtitle = plot_subtitle or ""

    graph_type = "adjacency" if is_adjacency else "bipartite"
    neighbor_set = set()
    if selected_node is not None and selected_node in G.nodes():
        neighbor_set = set(G.neighbors(selected_node))

    nodes_out = []
    for node in G.nodes():
        layer = 0
        if is_adjacency:
            color = rank_color(src_rank)
            deg_key = node
            label = f"rank_{src_rank}_id={node}"
        else:
            if str(node).startswith("src_"):
                color = rank_color(src_rank)
                deg_key = G.nodes[node].get("original_id", node)
                label = f"rank_{src_rank}_id={deg_key}"
                layer = 0
            else:
                color = rank_color(target_rank)
                deg_key = G.nodes[node].get("original_id", node)
                if target_kind == "hyperedge":
                    label = f"hyperedge_id={deg_key}"
                else:
                    label = f"rank_{target_rank}_id={deg_key}"
                layer = 1

        degree = _json_safe_float(node_degrees.get(deg_key, 1))
        stroke = "#fff"
        color_use = color
        if selected_node is not None:
            if node == selected_node:
                stroke = "#e41a1c"
                color_use = "#ff6666"
            elif node in neighbor_set:
                stroke = "#f0b000"
                color_use = "#fff566"

        nd = {
            "id": str(node),
            "label": label,
            "degree": degree,
            "color": color_use,
            "stroke": stroke,
        }
        if not is_adjacency:
            nd["layer"] = int(layer)
        nodes_out.append(nd)

    links_out = [{"source": str(u), "target": str(v)} for u, v in G.edges()]

    return {
        "graphType": graph_type,
        "title": title,
        "subtitle": subtitle,
        "nodes": nodes_out,
        "links": links_out,
    }


# ============================================================================
# Layered (multi-incidence) builders
# ============================================================================

def _layered_node_id(rank: int, original_id) -> str:
    return f"r{int(rank)}_n{int(original_id)}"


def build_layered_networkx(
    incidence_specs,
    *,
    max_nodes: int = 200,
    min_degree: int = 0,
):
    """
    Stack consecutive ``incidence_k`` matrices into a single multi-layer graph.

    Parameters
    ----------
    incidence_specs : list[tuple]
        Each item ``(sparse_matrix, source_rank, target_rank)`` where the
        sparse matrix is a coalesced COO with shape
        ``(num_source_cells, num_target_cells)``.
    max_nodes : int
        Total node cap; sampling is **stratified per layer** so no rank is
        wiped out (``ceil(max_nodes / num_layers)`` per layer).
    min_degree : int
        Per-node total-degree threshold across all stacked incidences.

    Returns
    -------
    tuple
        ``(G, node_degrees, layers)`` or ``(None, {}, [])`` if empty.
        - ``G``: ``nx.Graph`` with node attrs ``layer`` (rank) and
          ``original_id``.
        - ``node_degrees``: dict keyed by ``G`` node id.
        - ``layers``: sorted unique ranks.
    """
    if not incidence_specs:
        return None, {}, []

    specs = sorted(
        [(sp.coalesce() if sp is not None else None, int(sr), int(tr))
         for sp, sr, tr in incidence_specs if sp is not None],
        key=lambda t: t[1],
    )
    if not specs:
        return None, {}, []

    raw_edges = []
    layer_nodes = {}
    for sp, sr, tr in specs:
        idx = sp.indices().numpy()
        for i in range(idx.shape[1]):
            src_orig = int(idx[0, i])
            tgt_orig = int(idx[1, i])
            u = _layered_node_id(sr, src_orig)
            v = _layered_node_id(tr, tgt_orig)
            raw_edges.append((u, v))
            layer_nodes.setdefault(sr, {})[u] = src_orig
            layer_nodes.setdefault(tr, {})[v] = tgt_orig

    if not raw_edges:
        return None, {}, []

    degree = {}
    for u, v in raw_edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1

    if min_degree > 0:
        degree = {n: d for n, d in degree.items() if d >= min_degree}
    if not degree:
        return None, {}, []

    layers_sorted = sorted(layer_nodes.keys())
    num_layers = max(len(layers_sorted), 1)
    per_layer_cap = max(1, math.ceil(max_nodes / num_layers))

    selected = set()
    for rank in layers_sorted:
        candidates = [n for n in layer_nodes[rank] if n in degree]
        candidates.sort(key=lambda n: degree[n], reverse=True)
        selected.update(candidates[:per_layer_cap])

    if not selected:
        return None, {}, []

    G = nx.Graph()
    for n in selected:
        rank, orig = None, None
        for r, mp in layer_nodes.items():
            if n in mp:
                rank, orig = r, mp[n]
                break
        G.add_node(n, layer=int(rank), original_id=int(orig))

    seen_pairs = set()
    for u, v in raw_edges:
        if u not in selected or v not in selected:
            continue
        pair = (u, v) if u < v else (v, u)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        G.add_edge(u, v)

    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)
    if len(G.nodes()) == 0:
        return None, {}, []

    node_degrees = {n: G.degree(n) for n in G.nodes()}
    final_layers = sorted({G.nodes[n]["layer"] for n in G.nodes()})
    return G, node_degrees, final_layers


def networkx_to_layered_d3_payload(
    G,
    node_degrees,
    layers,
    *,
    rank_labels=None,
    plot_title=None,
    plot_subtitle=None,
):
    """Build a layered D3 payload (``graphType = 'layered'``)."""
    if G is None or len(G.nodes()) == 0:
        return None

    rank_labels = rank_labels or {}
    layers_sorted = sorted(layers) if layers else sorted(
        {G.nodes[n].get("layer", 0) for n in G.nodes()}
    )

    nodes_out = []
    for n in G.nodes():
        rank = int(G.nodes[n].get("layer", 0))
        orig = G.nodes[n].get("original_id", n)
        nodes_out.append({
            "id": str(n),
            "label": f"rank_{rank}_id={orig}",
            "degree": _json_safe_float(node_degrees.get(n, G.degree(n))),
            "color": rank_color(rank),
            "stroke": "#fff",
            "layer": rank,
        })

    links_out = [{"source": str(u), "target": str(v)} for u, v in G.edges()]

    return {
        "graphType": "layered",
        "title": plot_title or "Layered incidences",
        "subtitle": plot_subtitle or "",
        "nodes": nodes_out,
        "links": links_out,
        "layers": [int(r) for r in layers_sorted],
        "layerLabels": {
            str(r): rank_labels.get(r, f"Rank {r}") for r in layers_sorted
        },
    }


def launch_html_in_browser(path: Path) -> bool:
    """
    Open a local HTML file in a real browser.

    Prefer the OS default browser first (so it matches the user's current choice),
    then fall back to common browser executables on Windows.
    """
    path = path.resolve()
    try:
        if webbrowser.open(path.as_uri()):
            return True
    except Exception:
        pass

    if platform.system() == "Windows":
        candidates = [
            shutil.which("chrome"),
            os.environ.get("PROGRAMFILES", "")
            + r"\Google\Chrome\Application\chrome.exe",
            os.environ.get("PROGRAMFILES(X86)", "")
            + r"\Google\Chrome\Application\chrome.exe",
            shutil.which("msedge"),
            os.environ.get("PROGRAMFILES(X86)", "")
            + r"\Microsoft\Edge\Application\msedge.exe",
            os.environ.get("PROGRAMFILES", "") + r"\Microsoft\Edge\Application\msedge.exe",
            shutil.which("firefox"),
            os.environ.get("PROGRAMFILES", "") + r"\Mozilla Firefox\firefox.exe",
        ]
        for exe in candidates:
            if not exe:
                continue
            exe_path = Path(exe)
            if not exe_path.exists():
                continue
            try:
                subprocess.Popen([str(exe_path), str(path)], close_fds=True)
                return True
            except Exception:
                continue
        try:
            os.startfile(str(path))  # noqa: S606
            return True
        except Exception:
            pass
    return False


def open_d3_graph_window(payload):
    """Write standalone HTML to a temp file and open in browser."""
    if payload is None:
        st.warning("No graph to display.")
        return
    sel_ids = st.session_state.get("selected_neighborhood_ids") or []
    marker = "+".join(sel_ids) if sel_ids else None
    html_doc = build_standalone_d3_html(payload, cache_marker=marker)
    st.session_state["_d3_last_html"] = html_doc
    path = Path(tempfile.gettempdir()) / f"topobench_graph_{uuid.uuid4().hex}.html"
    path.write_text(html_doc, encoding="utf-8")
    if launch_html_in_browser(path):
        st.success("Opened graph in your browser.")
    else:
        st.warning(
            "Could not launch a browser automatically. Use **Download last D3 graph** "
            "below and open the file in Edge or Chrome (not VS Code preview)."
        )


# ============================================================================
# Lifting Application
# ============================================================================

def _ensure_float_node_features(data):
    """
    Make ``data.x`` lifting-friendly.

    TopoBench liftings touch ``data.x.shape`` while building the NetworkX view
    (``topobench/transforms/liftings/liftings.py``), so a missing ``x`` raises
    ``'NoneType' object has no attribute 'shape'``. Some feature liftings
    (e.g. ``ProjectionSum``) also use ``torch.matmul`` / ``sparse_mm`` with
    float sparse incidence; integer ``x`` (bag-of-words, raw counts) triggers
    ``expected scalar type Float but found Long``.

    Behavior:
    - If ``x`` is ``None`` (or missing): synthesize a ``[num_nodes, 1]`` ones
      tensor as a placeholder so liftings can iterate over nodes.
    - If ``x`` exists but is integral: cast to float.
    """
    if data is None or not hasattr(data, "clone"):
        return data
    try:
        d = data.clone()
    except Exception:
        d = data

    x = getattr(d, "x", None)
    if x is None:
        num_nodes = getattr(d, "num_nodes", None)
        if num_nodes is None:
            ei = getattr(d, "edge_index", None)
            if ei is not None and ei.numel() > 0:
                num_nodes = int(ei.max().item()) + 1
        if num_nodes is None or num_nodes <= 0:
            return d
        d.x = torch.ones((int(num_nodes), 1), dtype=torch.float32)
        return d

    if hasattr(x, "dtype") and not torch.is_floating_point(x):
        d.x = x.float()
    return d


def apply_lifting(data, lifting_info):
    """
    Instantiate and apply a TopoBench lifting transform to a data object.

    Uses DataTransform — the same wrapper TopoBench uses internally.
    DataTransform looks up the class by transform_name in the TRANSFORMS
    registry dict, passes all kwargs to the constructor, then calls
    transform(data) -> transformed_data.

    feature_lifting is passed as a string key ('ProjectionSum', etc.),
    exactly as AbstractLifting.__init__ expects.
    """
    from topobench.transforms.data_transform import DataTransform

    transform_name = lifting_info['name']
    config = lifting_info['config']

    # Keys that are not constructor parameters or need special handling
    SKIP_KEYS = {'transform_type', 'transform_name'}

    params = {}
    for k, v in config.items():
        if k in SKIP_KEYS:
            continue
        # Drop unresolved Hydra interpolation strings (e.g. "${oc.select:...}")
        if isinstance(v, str) and v.startswith('${'):
            continue
        params[k] = v

    # DataTransform takes transform_name as first positional arg + **kwargs
    transform = DataTransform(transform_name=transform_name, **params)

    # Extract a single graph object if the dataset is an iterable collection
    if hasattr(data, '__getitem__'):
        single = data[0]
    else:
        single = data

    single = _ensure_float_node_features(single)

    # PyG BaseTransform.__call__ -> forward(data) -> transformed Data
    transformed = transform(single)
    return transformed


BASIC_EDITABLE_KEYS = {
    "feature_lifting",
    "complex_dim",
    "signed",
    "k_value",
    "loop",
    "k_neighbors",
    "num_communities",
    "max_k_simplices",
    "distance_threshold",
}

GRAPH_EDITOR_TARGETS = {"hypergraph", "simplicial", "cell", "combinatorial"}


def _selected_lifting_editor_id(lifting_info):
    """Stable ID used to scope editable config state."""
    if lifting_info is None:
        return None
    return "::".join(
        [
            str(lifting_info.get("source", "")),
            str(lifting_info.get("target", "")),
            str(lifting_info.get("name", "")),
            str(lifting_info.get("config_path", "")),
        ]
    )


def _is_graph_family_lifting(lifting_info):
    """True for graph->(hypergraph/simplicial/cell/combinatorial) liftings."""
    if lifting_info is None:
        return False
    return (
        lifting_info.get("source") == "graph"
        and lifting_info.get("target") in GRAPH_EDITOR_TARGETS
    )


def validate_basic_lifting_config(config):
    """Validate exposed editable config fields."""
    errors = []

    def _is_pos_int(v):
        return isinstance(v, int) and not isinstance(v, bool) and v >= 1

    if "complex_dim" in config and not _is_pos_int(config["complex_dim"]):
        errors.append("`complex_dim` must be an integer >= 1.")
    if "k_value" in config and not _is_pos_int(config["k_value"]):
        errors.append("`k_value` must be an integer >= 1.")
    if "k_neighbors" in config and not _is_pos_int(config["k_neighbors"]):
        errors.append("`k_neighbors` must be an integer >= 1.")
    if "num_communities" in config and not _is_pos_int(config["num_communities"]):
        errors.append("`num_communities` must be an integer >= 1.")
    if "max_k_simplices" in config and not _is_pos_int(config["max_k_simplices"]):
        errors.append("`max_k_simplices` must be an integer >= 1.")
    if "distance_threshold" in config:
        try:
            v = float(config["distance_threshold"])
            if v <= 0:
                errors.append("`distance_threshold` must be > 0.")
        except Exception:
            errors.append("`distance_threshold` must be a number > 0.")

    if "signed" in config and not isinstance(config["signed"], bool):
        errors.append("`signed` must be true/false.")
    if "loop" in config and not isinstance(config["loop"], bool):
        errors.append("`loop` must be true/false.")
    if "feature_lifting" in config and config["feature_lifting"] in ("", None):
        errors.append("`feature_lifting` cannot be empty.")
    return errors


def _hydra_default_from_interpolation(value):
    """Extract fallback default from `${oc.select:...,<default>}` style strings."""
    if not isinstance(value, str):
        return None
    m = re.match(r"^\$\{oc\.select:[^,]+,(.+)\}$", value.strip())
    if not m:
        return None
    return m.group(1).strip()


def _safe_start_int(value, default=1):
    """Integer start value tolerant to Hydra interpolation strings."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        interp = _hydra_default_from_interpolation(value)
        if interp is not None:
            value = interp
        try:
            return int(value)
        except Exception:
            return default
    try:
        return int(value)
    except Exception:
        return default


def _safe_start_float(value, default=1.0):
    """Float start value tolerant to Hydra interpolation strings."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        interp = _hydra_default_from_interpolation(value)
        if interp is not None:
            value = interp
        try:
            return float(value)
        except Exception:
            return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_start_bool(value, default=False):
    """Boolean start value tolerant to Hydra interpolation strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        interp = _hydra_default_from_interpolation(value)
        v = (interp if interp is not None else value).strip().lower()
        if v in {"true", "1", "yes", "on"}:
            return True
        if v in {"false", "0", "no", "off"}:
            return False
    return default


def render_basic_lifting_editor(selected_lifting):
    """
    Render basic-safe editable controls for graph-based liftings.

    Returns
    -------
    tuple[dict | None, list[str]]
        Effective edited config and validation errors.
    """
    if selected_lifting is None:
        return None, []

    editor_id = _selected_lifting_editor_id(selected_lifting)
    if st.session_state.get("_editor_lifting_id") != editor_id:
        st.session_state["_editor_lifting_id"] = editor_id
        st.session_state["editable_lifting_config"] = copy.deepcopy(
            selected_lifting.get("config", {})
        )

    if not _is_graph_family_lifting(selected_lifting):
        st.caption(
            "Config editing currently supports graph-based liftings only "
            "(graph2hypergraph / simplicial / cell / combinatorial)."
        )
        return copy.deepcopy(selected_lifting.get("config", {})), []

    cfg = st.session_state.get("editable_lifting_config", {})
    st.markdown("**Editable config (basic)**")

    feature_options = ["ProjectionSum", "Identity", "Concatenation", "Set"]
    editable_keys = [k for k in BASIC_EDITABLE_KEYS if k in cfg]
    for key in sorted(editable_keys):
        wkey = f"editcfg::{editor_id}::{key}"
        val = cfg.get(key)
        if key == "feature_lifting":
            opts = feature_options[:]
            if val not in opts and val is not None:
                opts.insert(0, val)
            idx = opts.index(val) if val in opts else 0
            cfg[key] = st.selectbox("feature_lifting", opts, index=idx, key=wkey)
        elif key in {"signed", "loop"}:
            cfg[key] = st.toggle(key, value=_safe_start_bool(val), key=wkey)
        elif key in {"distance_threshold"}:
            start = _safe_start_float(val, default=1.0)
            cfg[key] = st.number_input(key, value=start, min_value=0.000001, key=wkey)
        else:
            start = _safe_start_int(val, default=1)
            cfg[key] = int(st.number_input(key, value=start, min_value=1, step=1, key=wkey))

    st.session_state["editable_lifting_config"] = cfg
    errors = validate_basic_lifting_config(cfg)
    if errors:
        st.error("Invalid lifting config:\n- " + "\n- ".join(errors))

    if st.button(
        "Reset edited config to defaults",
        key=f"cfg_reset::{editor_id}",
        use_container_width=True,
    ):
        for k in BASIC_EDITABLE_KEYS:
            wk = f"editcfg::{editor_id}::{k}"
            if wk in st.session_state:
                del st.session_state[wk]
        st.session_state["editable_lifting_config"] = copy.deepcopy(
            selected_lifting.get("config", {})
        )
        st.rerun()

    with st.expander("Config preview", expanded=False):
        cfg_display = {
            k: v
            for k, v in cfg.items()
            if k not in ("neighborhoods",)
        }
        st.json(cfg_display)
    return copy.deepcopy(cfg), errors


# ============================================================================
# Streamlit App
# ============================================================================

D3_EMBED_HEIGHT = 760


def _render_left_config(available_datasets):
    """Render configuration controls in the Streamlit sidebar; return selections."""
    st.header("Data configuration")

    with st.expander("Dataset", expanded=True):
        selected_domain = st.selectbox(
            "Domain",
            options=list(available_datasets.keys()),
            index=0,
            help="Topological domain (folder under configs/dataset).",
        )
        datasets_in_domain = available_datasets.get(selected_domain, [])
        selected_dataset = st.selectbox(
            "Dataset",
            options=datasets_in_domain,
            index=0,
            help=f"YAML stem under configs/dataset/{selected_domain}/",
        )
        st.caption(f"**{selected_domain}** / **{selected_dataset}**")

    use_lifting = st.toggle(
        "Use lifting",
        value=st.session_state.get("use_lifting", False),
        key="use_lifting",
    )

    selected_lifting = None
    edited_lifting_config = None
    edited_lifting_errors = []
    if use_lifting:
        with st.expander("Transform / lifting (optional)", expanded=True):
            all_liftings = discover_available_liftings()
            available_for_domain = all_liftings.get(selected_domain, [])
            if not available_for_domain:
                st.caption(
                    f"No liftings configured for domain **{selected_domain}**."
                )
                st.session_state["selected_lifting_name"] = None
            else:
                targets = sorted(set(l["target"] for l in available_for_domain))
                previous_target = st.session_state.get("selected_lifting_target")
                target_index = (
                    targets.index(previous_target)
                    if previous_target in targets
                    else 0
                )
                selected_target = st.selectbox(
                    "Target domain",
                    options=targets,
                    index=target_index,
                    format_func=lambda x: x.capitalize(),
                )
                st.session_state["selected_lifting_target"] = selected_target
                liftings_for_target = [
                    l for l in available_for_domain if l["target"] == selected_target
                ]
                lifting_options = {l["name"]: l for l in liftings_for_target}
                lifting_names = list(lifting_options.keys())
                previous_name = st.session_state.get("selected_lifting_name")
                name_index = (
                    lifting_names.index(previous_name)
                    if previous_name in lifting_names
                    else 0
                )
                selected_lifting_name = st.selectbox(
                    "Lifting method",
                    options=lifting_names,
                    index=name_index,
                )
                st.session_state["selected_lifting_name"] = selected_lifting_name
                selected_lifting = lifting_options[selected_lifting_name]
                edited_lifting_config, edited_lifting_errors = render_basic_lifting_editor(
                    selected_lifting
                )

    st.session_state["selected_lifting"] = selected_lifting
    st.session_state["edited_lifting_config"] = edited_lifting_config
    st.session_state["edited_lifting_errors"] = edited_lifting_errors

    st.subheader("Graph sampling")
    max_nodes = st.slider(
        "Max nodes in graph",
        min_value=50,
        max_value=500,
        value=st.session_state.get("max_nodes", 150),
        help="Cap on nodes when building the NetworkX view for D3.",
        key="ui_max_nodes",
    )
    min_degree = st.slider(
        "Minimum degree",
        min_value=0,
        max_value=20,
        value=st.session_state.get("min_degree", 0),
        key="ui_min_degree",
    )
    st.session_state["max_nodes"] = max_nodes
    st.session_state["min_degree"] = min_degree

    st.subheader("Actions")
    load_clicked = st.button(
        "Load graph",
        type="primary",
        use_container_width=True,
        key="load_graph_btn",
    )

    return {
        "selected_domain": selected_domain,
        "selected_dataset": selected_dataset,
        "use_lifting": use_lifting,
        "selected_lifting": selected_lifting,
        "edited_lifting_config": edited_lifting_config,
        "edited_lifting_errors": edited_lifting_errors,
        "max_nodes": max_nodes,
        "min_degree": min_degree,
        "load_clicked": load_clicked,
    }


def _split_neighborhoods(available):
    """Bucket available neighborhoods into the three picker boxes."""
    graph_ids = [n["id"] for n in available if n["kind"] in ("graph", "hyperedges")]
    incidence = [n for n in available if n["kind"] == "incidence"]
    adjacency = [n for n in available if n["kind"] == "adjacency"]
    incidence.sort(key=lambda n: n.get("rank", 0))
    adjacency.sort(key=lambda n: n.get("rank", 0))
    return graph_ids, incidence, adjacency


def _render_neighborhood_picker():
    """Three boxes (Graph radio / Incidence checklist / Adjacency radio)."""
    st.subheader("Available neighborhoods")
    available = st.session_state.get("available_neighborhoods") or []
    if not available:
        st.info(
            "Load a dataset to enable neighborhood selection. After loading, "
            "you can pick one **Graph** entry, one or more **Incidence** "
            "entries (stacked bottom-to-top), or one **Adjacency** entry."
        )
        return

    graph_ids, incidence_items, adjacency_items = _split_neighborhoods(
        available
    )
    label_map = {n["id"]: n["label"] for n in available}
    selected_ids = list(st.session_state.get("selected_neighborhood_ids") or [])

    # Make sure every picker widget key exists in state BEFORE the widgets
    # render. _sync_picker_widget_state is idempotent and writes the values
    # consistent with the current selected_neighborhood_ids -- this avoids
    # ever falling back to st.radio's `index=` parameter on first render
    # (which has been observed to fire spurious on_change callbacks that
    # silently clobber the selection).
    _sync_picker_widget_state(selected_ids)

    with st.container(border=True):
        st.markdown("**Graph**")
        if graph_ids:
            options = ["(none)"] + graph_ids
            st.radio(
                "Graph view",
                options=options,
                format_func=lambda i: "(none)" if i == "(none)" else label_map.get(i, i),
                key="graph_radio",
                on_change=_on_graph_pick,
                label_visibility="collapsed",
            )
        else:
            st.caption("No graph / hyperedges neighborhood on this data.")

    with st.container(border=True):
        st.markdown("**Incidence** (stacked bottom-to-top by rank)")
        if incidence_items:
            for item in incidence_items:
                st.checkbox(
                    item["label"],
                    key=f"inc_{item['id']}_check",
                    on_change=_on_incidence_toggle,
                )
        else:
            st.caption("No incidence_k neighborhoods on this data.")

    with st.container(border=True):
        st.markdown("**Adjacency**")
        if adjacency_items:
            adj_ids = [n["id"] for n in adjacency_items]
            options = ["(none)"] + adj_ids
            st.radio(
                "Adjacency view",
                options=options,
                format_func=lambda i: "(none)" if i == "(none)" else label_map.get(i, i),
                key="adj_radio",
                on_change=_on_adjacency_pick,
                label_visibility="collapsed",
            )
        else:
            st.caption("No adjacency_k neighborhoods on this data.")


def _sync_picker_widget_state(selected_ids):
    """Force every picker widget key to match ``selected_ids``.

    This is the single source of truth for picker widget state. Calling this
    in every callback guarantees that on the next rerun the radios and
    checkboxes render with values consistent with ``selected_neighborhood_ids``,
    avoiding the fragile pop + ``index=`` re-initialization dance (which was
    causing incidence checkboxes to silently uncheck themselves).
    """
    available = st.session_state.get("available_neighborhoods") or []
    selected_set = set(selected_ids)

    graph_options = [
        n["id"] for n in available if n["kind"] in ("graph", "hyperedges")
    ]
    if graph_options:
        chosen = next((g for g in graph_options if g in selected_set), "(none)")
        st.session_state["graph_radio"] = chosen

    adj_options = [n["id"] for n in available if n["kind"] == "adjacency"]
    if adj_options:
        chosen = next((a for a in adj_options if a in selected_set), "(none)")
        st.session_state["adj_radio"] = chosen

    for n in available:
        if n["kind"] == "incidence":
            st.session_state[f"inc_{n['id']}_check"] = n["id"] in selected_set


def _on_graph_pick():
    """Selecting a Graph option becomes the sole selection.

    Picking ``(none)`` is treated as a no-op: the current selection (whatever
    it is) is preserved. This avoids accidentally wiping the selection when
    the radio re-renders with no graph option active.
    """
    if st.session_state.get("data") is None:
        return
    pick = st.session_state.get("graph_radio")
    if not pick or pick == "(none)":
        # Re-sync widgets to current selection (in case user tried to deselect).
        _sync_picker_widget_state(
            st.session_state.get("selected_neighborhood_ids") or []
        )
        return
    _commit_selection([pick])


def _on_incidence_toggle():
    """Toggling an incidence checkbox replaces the selection with the
    set of currently-checked incidences."""
    if st.session_state.get("data") is None:
        return
    incidence_ids = [
        n["id"] for n in st.session_state.get("available_neighborhoods") or []
        if n["kind"] == "incidence"
    ]
    new_ids = [i for i in incidence_ids if st.session_state.get(f"inc_{i}_check")]
    if not new_ids:
        # Don't allow ending with no selection -- restore one previously-checked
        # incidence so the layered view always has something to display.
        prev = list(st.session_state.get("selected_neighborhood_ids") or [])
        prev_inc = [p for p in prev if p in incidence_ids]
        if not prev_inc:
            return
        new_ids = [prev_inc[0]]
    _commit_selection(new_ids)


def _on_adjacency_pick():
    """Selecting an Adjacency option becomes the sole selection.

    Picking ``(none)`` is treated as a no-op: the current selection is
    preserved (see ``_on_graph_pick``).
    """
    if st.session_state.get("data") is None:
        return
    pick = st.session_state.get("adj_radio")
    if not pick or pick == "(none)":
        _sync_picker_widget_state(
            st.session_state.get("selected_neighborhood_ids") or []
        )
        return
    _commit_selection([pick])


def _commit_selection(new_ids):
    """Persist the selection, sync all picker widgets, rebuild the embed."""
    new_ids = list(new_ids)
    st.session_state["selected_neighborhood_ids"] = new_ids
    _sync_picker_widget_state(new_ids)
    _rebuild_embed_for_neighborhoods(new_ids)


def _rebuild_embed_for_neighborhoods(neigh_ids):
    """Rebuild the embedded D3 view for a list of neighborhood ids.

    A single non-incidence id falls through to the original single-matrix
    pipeline. Multiple incidence ids (or any incidence selection) goes
    through the layered pipeline (stacked ranks bottom-to-top).
    """
    if not neigh_ids:
        return False
    data = st.session_state.get("data")
    if data is None:
        st.error("Cannot rebuild graph: no dataset loaded.")
        return False
    dset0 = data[0] if hasattr(data, "__getitem__") else data

    rank_labels_for_payload = st.session_state.get("rank_labels") or {}
    applied_lift = st.session_state.get("lifting_applied")
    # Sampling sliders only apply on "Load graph"; neighborhood switches reuse
    # the snapshot from the last successful load.
    max_nodes = int(
        st.session_state.get(
            "_loaded_max_nodes", st.session_state.get("max_nodes", 150)
        )
    )
    min_degree = int(
        st.session_state.get(
            "_loaded_min_degree", st.session_state.get("min_degree", 0)
        )
    )

    lift_subtitle = (
        f"Lift: {applied_lift['name']} "
        f"({applied_lift['source']}→{applied_lift['target']})"
        if applied_lift is not None
        else "Lift: none (raw dataset)"
    )

    incidence_ids = [i for i in neigh_ids if i.startswith("incidence_")
                     and i != "incidence_hyperedges"]
    use_layered = (
        len(incidence_ids) >= 1
        and len(incidence_ids) == len(neigh_ids)
    )

    try:
        if use_layered:
            specs = []
            for nid in incidence_ids:
                k = int(nid.split("_", 1)[1])
                sp = incidence_rank_k_to_sparse(dset0, k)
                if sp is None:
                    st.error(f"Neighborhood '{nid}' is not available.")
                    return False
                specs.append((sp, k - 1, k))

            ranks_chosen = sorted({sr for _, sr, _ in specs}
                                  | {tr for _, _, tr in specs})
            expected = list(range(min(ranks_chosen), max(ranks_chosen) + 1))
            if ranks_chosen != expected:
                missing = sorted(set(expected) - set(ranks_chosen))
                st.info(
                    "Selected incidences are not contiguous in rank; "
                    f"missing rank(s): {missing}. Layers will still render "
                    "but unrelated ranks may not connect."
                )

            G_load, nd_load, layers = build_layered_networkx(
                specs, max_nodes=max_nodes, min_degree=min_degree
            )
            if not G_load or len(G_load.nodes()) == 0:
                st.error(
                    "Error building graph payload: "
                    "no nodes survive current filters for the selected layers."
                )
                return False

            sorted_inc = sorted(incidence_ids,
                                key=lambda x: int(x.split("_", 1)[1]))
            vdesc = "Layered: " + " | ".join(sorted_inc)
            payload = networkx_to_layered_d3_payload(
                G_load,
                nd_load,
                layers,
                rank_labels=rank_labels_for_payload,
                plot_title=vdesc,
                plot_subtitle=lift_subtitle,
            )
            caption_extra = (
                f"{len(G_load.nodes()):,} nodes across "
                f"{len(layers)} layer(s); {len(G_load.edges()):,} edges"
            )
        else:
            single_id = neigh_ids[0]
            matrix, vdesc, relation_ctx = get_named_visualization_matrix(
                dset0, single_id
            )
            if matrix is None:
                st.error(
                    f"Error building graph payload: neighborhood "
                    f"'{single_id}' is not available on the loaded data."
                )
                return False
            G_load, nd_load = sparse_to_networkx(
                matrix, max_nodes=max_nodes, min_degree=min_degree
            )
            if not G_load or len(G_load.nodes()) == 0:
                st.error(
                    "Error building graph payload: "
                    "graph is empty with current filters."
                )
                return False
            payload = networkx_to_d3_payload(
                G_load,
                nd_load,
                rank_labels=rank_labels_for_payload,
                plot_title=vdesc,
                relation_context=relation_ctx,
                plot_subtitle=lift_subtitle,
            )
            stats = get_matrix_stats(matrix)
            caption_extra = (
                f"shape {stats['shape']}, {stats['num_edges']:,} nonzeros, "
                f"density {stats['density']:.4%}"
                if stats
                else ""
            )

        if payload is None:
            st.error("Error building graph payload: payload is empty.")
            return False
    except Exception as e:
        st.error(f"Error building graph payload: {e}")
        return False

    cache_marker = "+".join(neigh_ids)
    embed_html = build_standalone_d3_html(
        payload,
        embed=True,
        chart_min_height=max(360, D3_EMBED_HEIGHT - 80),
        cache_marker=cache_marker,
    )
    download_html = build_standalone_d3_html(payload, cache_marker=cache_marker)
    st.session_state["_d3_embed_html"] = embed_html
    st.session_state["_d3_last_html"] = download_html
    st.session_state["_d3_payload"] = payload
    st.session_state["_d3_caption"] = (
        f"{vdesc} — {caption_extra}" if caption_extra else vdesc
    )
    st.session_state["selected_neighborhood_ids"] = list(neigh_ids)
    return True


def _do_load_graph(cfg):
    """Build dataset, optional lifting, and embed-ready D3 HTML.

    On success, populates ``st.session_state`` with the new payload/HTML/data
    and returns ``True``. On any failure, surfaces ``st.error`` and returns
    ``False`` (without clearing the previously embedded view).
    """
    try:
        raw, loaded_domain = load_dataset(
            domain=cfg["selected_domain"],
            dataset_name=cfg["selected_dataset"],
        )
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return False

    try:
        st.session_state["data_original"] = copy.deepcopy(raw)
    except Exception:
        st.session_state["data_original"] = raw

    current_data = raw
    applied_lift = None
    if cfg["use_lifting"] and cfg["selected_lifting"] is not None:
        if cfg["edited_lifting_errors"]:
            st.error(
                "Error applying lifting: invalid lifting config.\n- "
                + "\n- ".join(cfg["edited_lifting_errors"])
            )
            return False
        lifting_payload = copy.deepcopy(cfg["selected_lifting"])
        if isinstance(cfg["edited_lifting_config"], dict):
            lifting_payload["config"] = copy.deepcopy(cfg["edited_lifting_config"])
        try:
            current_data = [apply_lifting(raw, lifting_payload)]
            applied_lift = lifting_payload
        except Exception as e:
            st.error(f"Error applying lifting: {e}")
            return False

    st.session_state["data"] = current_data
    st.session_state["lifting_applied"] = applied_lift
    st.session_state["data_domain"] = loaded_domain
    st.session_state["dataset_name"] = cfg["selected_dataset"]
    dset0 = (
        current_data[0] if hasattr(current_data, "__getitem__") else current_data
    )
    rank_labels_for_payload = get_rank_labels(
        loaded_domain, cfg["selected_dataset"], dset0
    )
    st.session_state["rank_labels"] = rank_labels_for_payload

    available = enumerate_neighborhoods(dset0)
    st.session_state["available_neighborhoods"] = available
    if not available:
        st.error(
            "No incidence/adjacency neighborhoods are available on this data."
        )
        return False

    default_id = pick_default_neighborhood_id(available)
    default_ids = [default_id] if default_id else []
    st.session_state["selected_neighborhood_ids"] = default_ids
    st.session_state["_loaded_max_nodes"] = int(cfg["max_nodes"])
    st.session_state["_loaded_min_degree"] = int(cfg["min_degree"])
    # Sync picker widget keys to the freshly-loaded default selection
    # (also overwrites any stale values from a previous Load).
    _sync_picker_widget_state(default_ids)

    return _rebuild_embed_for_neighborhoods(default_ids)


def _render_right_view():
    """Render the embedded D3 graph (or a placeholder) in the main area."""
    st.header("Graph")
    _render_neighborhood_picker()
    embed_html = st.session_state.get("_d3_embed_html")
    if not embed_html:
        st.info(
            "Use the **sidebar** to configure the dataset, then click **Load graph**."
        )
        return

    data = st.session_state.get("data")
    dset0 = (
        data[0] if (data is not None and hasattr(data, "__getitem__")) else data
    )
    lifting_applied = st.session_state.get("lifting_applied")
    if dset0 is not None and hasattr(dset0, "num_nodes"):
        num_nodes = dset0.num_nodes
        if (
            num_nodes is None
            and getattr(dset0, "edge_index", None) is not None
        ):
            num_nodes = int(dset0.edge_index.max().item()) + 1
        num_features = (
            dset0.x.shape[1]
            if (
                hasattr(dset0, "x")
                and dset0.x is not None
                and len(dset0.x.shape) > 1
            )
            else 0
        )
        if lifting_applied:
            st.success(
                f"Viewing lifted data — **{lifting_applied['name']}** "
                f"({lifting_applied['source']} → {lifting_applied['target']}) | "
                f"Nodes: {num_nodes}, "
                f"Node features: {num_features if num_features else 'N/A'}"
            )
        else:
            st.info(
                f"Dataset loaded — Nodes: {num_nodes}, "
                f"Node features: {num_features if num_features else 'N/A'}"
            )

    caption = st.session_state.get("_d3_caption")
    if caption:
        st.caption(caption)

    sel_ids = list(st.session_state.get("selected_neighborhood_ids") or [])
    avail = st.session_state.get("available_neighborhoods") or []
    if sel_ids:
        labels = [next((n["label"] for n in avail if n["id"] == s), s) for s in sel_ids]
        joined_ids = ", ".join(f"`{s}`" for s in sel_ids)
        joined_labels = "; ".join(labels)
        st.markdown(f"**Currently displayed:** {joined_ids} — {joined_labels}")

    # ``components.html`` has no ``key=`` in Streamlit 1.50; wrap in a keyed
    # container so changing the neighborhood remounts the iframe subtree.
    neigh_key = "+".join(sel_ids) if sel_ids else "default"
    with st.container(key=f"d3_embed::{neigh_key}"):
        components.html(embed_html, height=D3_EMBED_HEIGHT, scrolling=False)

    payload = st.session_state.get("_d3_payload")
    if payload is not None and st.button(
        "Open graph in new browser window",
        key="d3_open_current",
        use_container_width=True,
    ):
        open_d3_graph_window(payload)

    last_html = st.session_state.get("_d3_last_html")
    if last_html:
        st.download_button(
            label="Download last D3 graph (HTML)",
            data=last_html,
            file_name="topobench_graph.html",
            mime="text/html",
            key="d3_download_main",
            use_container_width=True,
        )


def main():
    st.title("Hypergraph Neighborhood Explorer")
    st.caption(
        "Configure data and optional lifting in the **sidebar**, then **Load graph**. "
        "Use **Available neighborhoods** on this page to switch connectivity views."
    )

    flash = st.session_state.pop("_flash_ok", None)
    if flash:
        st.success(flash)

    available_datasets = discover_available_datasets()

    with st.sidebar:
        cfg = _render_left_config(available_datasets)

    if cfg["load_clicked"]:
        with st.spinner("Loading graph…"):
            ok = _do_load_graph(cfg)
        if ok:
            st.rerun()

    _render_right_view()


if __name__ == "__main__":
    main()