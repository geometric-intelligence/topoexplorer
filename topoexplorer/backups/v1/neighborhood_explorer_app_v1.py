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


def incidence_rank_one_to_sparse(data):
    """
    Sparse 0-cell to 1-cell incidence from simplicial / cell / combinatorial lifts.

    TopoBench stores these as ``incidence_1`` (see ``get_complex_connectivity``).
    """
    inc1 = None
    if hasattr(data, "incidence_1") and data.incidence_1 is not None:
        inc1 = data.incidence_1
    elif "incidence_1" in data:
        inc1 = data["incidence_1"]
    if inc1 is None:
        return None
    if hasattr(inc1, "layout") and inc1.layout == torch.sparse_coo:
        return inc1.coalesce()
    if torch.is_tensor(inc1) and inc1.dim() == 2 and inc1.size(0) == 2:
        row, col = inc1[0].long(), inc1[1].long()
        n0 = int(row.max().item()) + 1 if row.numel() else 0
        n1 = int(col.max().item()) + 1 if col.numel() else 0
        vals = torch.ones(row.size(0), dtype=torch.float32, device=row.device)
        return torch.sparse_coo_tensor(
            torch.stack([row, col]),
            vals,
            (n0, n1),
            device=row.device,
        ).coalesce()
    return None


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
    inc1 = incidence_rank_one_to_sparse(data)

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
        if inc1 is not None:
            return (
                inc1,
                "Rank-0 to Rank-1 incidence",
                {"type": "bipartite", "source_rank": 0, "target_rank": 1},
            )
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
        if inc1 is not None:
            return (
                inc1,
                "Rank-0 to Rank-1 incidence",
                {"type": "bipartite", "source_rank": 0, "target_rank": 1},
            )
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
            if node in valid_nodes or len(valid_nodes) == 0:
                G.add_node(f"src_{node}", bipartite=0, original_id=node)
        for node in tgt_nodes:
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
    html_doc = build_standalone_d3_html(payload)
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
    TopoBench feature liftings (e.g. ProjectionSum) use ``torch.matmul`` /
    ``sparse_mm`` with float sparse incidence; integer ``x`` (common for
    bag-of-words or raw counts) triggers ``expected scalar type Float but found Long``.
    """
    if data is None or not hasattr(data, "clone"):
        return data
    try:
        d = data.clone()
    except Exception:
        d = data
    x = getattr(d, "x", None)
    if x is not None and hasattr(x, "dtype") and not torch.is_floating_point(x):
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


# ============================================================================
# Streamlit App
# ============================================================================

def main():
    st.title("Hypergraph Neighborhood Explorer")
    st.caption("Configure data and optional lifting, then load and open the graph in D3.")

    flash = st.session_state.pop("_flash_ok", None)
    if flash:
        st.success(flash)

    available_datasets = discover_available_datasets()

    with st.sidebar:
        if st.session_state.get("_d3_last_html"):
            st.download_button(
                label="Download last D3 graph (HTML)",
                data=st.session_state["_d3_last_html"],
                file_name="topobench_graph.html",
                mime="text/html",
                key="d3_download_sidebar",
            )

    _, center, _ = st.columns([1, 6, 1])
    with center:
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

        with st.expander("Transform / lifting (optional)", expanded=True):
            all_liftings = discover_available_liftings()
            available_for_domain = all_liftings.get(selected_domain, [])
            selected_lifting = None
            if not available_for_domain:
                st.caption(f"No liftings configured for domain **{selected_domain}**.")
            else:
                targets = sorted(set(l["target"] for l in available_for_domain))
                selected_target = st.selectbox(
                    "Target domain",
                    options=targets,
                    format_func=lambda x: x.capitalize(),
                )
                liftings_for_target = [
                    l for l in available_for_domain if l["target"] == selected_target
                ]
                lifting_options = {l["name"]: l for l in liftings_for_target}
                selected_lifting_name = st.selectbox(
                    "Lifting method",
                    options=list(lifting_options.keys()),
                )
                selected_lifting = lifting_options[selected_lifting_name]
                with st.expander("Config preview", expanded=False):
                    cfg_display = {
                        k: v
                        for k, v in selected_lifting["config"].items()
                        if k not in ("neighborhoods",)
                    }
                    st.json(cfg_display)
            st.session_state["selected_lifting"] = selected_lifting

        st.subheader("Graph sampling")
        max_nodes = st.slider(
            "Max nodes in graph",
            min_value=50,
            max_value=500,
            value=150,
            help="Cap on nodes when building the NetworkX view for D3.",
        )
        min_degree = st.slider(
            "Minimum degree",
            min_value=0,
            max_value=20,
            value=0,
        )
        st.session_state["max_nodes"] = max_nodes
        st.session_state["min_degree"] = min_degree

        st.subheader("Actions")
        apply_lift_load_graph = st.toggle(
            "Apply selected lifting on load",
            value=False,
            disabled=st.session_state.get("selected_lifting") is None,
            key="apply_lift_load_graph",
        )
        if st.button("Load graph", type="primary", use_container_width=True):
            with st.spinner("Loading and opening graph…"):
                # Stage 1: dataset load
                try:
                    raw, loaded_domain = load_dataset(
                        domain=selected_domain, dataset_name=selected_dataset
                    )
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
                    return

                # Keep raw snapshot for diagnostics/possible future actions
                try:
                    st.session_state["data_original"] = copy.deepcopy(raw)
                except Exception:
                    st.session_state["data_original"] = raw

                # Stage 2: optional lifting
                sel_lift = st.session_state.get("selected_lifting")
                current_data = raw
                applied_lift = None
                if apply_lift_load_graph and sel_lift is not None:
                    try:
                        current_data = [apply_lifting(raw, sel_lift)]
                        applied_lift = sel_lift
                    except Exception as e:
                        st.error(f"Error applying lifting: {e}")
                        return

                st.session_state["data"] = current_data
                st.session_state["lifting_applied"] = applied_lift
                st.session_state["data_domain"] = loaded_domain
                st.session_state["dataset_name"] = selected_dataset
                dset0 = (
                    current_data[0]
                    if hasattr(current_data, "__getitem__")
                    else current_data
                )
                rank_labels_for_payload = get_rank_labels(
                    loaded_domain, selected_dataset, dset0
                )
                st.session_state["rank_labels"] = rank_labels_for_payload

                # Stage 3: graph payload build
                try:
                    matrix, vdesc, relation_ctx = get_primary_visualization_matrix(
                        dset0, "auto"
                    )
                    if matrix is None:
                        st.error(
                            "Error building graph payload: "
                            "no edge_index/incidence connectivity found."
                        )
                        return
                    mx = st.session_state.get("max_nodes", 150)
                    md = st.session_state.get("min_degree", 0)
                    G_load, nd_load = sparse_to_networkx(
                        matrix, max_nodes=mx, min_degree=md
                    )
                    if not G_load or len(G_load.nodes()) == 0:
                        st.error(
                            "Error building graph payload: "
                            "graph is empty with current filters."
                        )
                        return
                    lift_subtitle = (
                        f"Lift: {applied_lift['name']} "
                        f"({applied_lift['source']}→{applied_lift['target']})"
                        if applied_lift is not None
                        else "Lift: none (raw dataset)"
                    )
                    payload = networkx_to_d3_payload(
                        G_load,
                        nd_load,
                        rank_labels=rank_labels_for_payload,
                        plot_title=vdesc,
                        relation_context=relation_ctx,
                        plot_subtitle=lift_subtitle,
                    )
                    if payload is None:
                        st.error("Error building graph payload: payload is empty.")
                        return
                except Exception as e:
                    st.error(f"Error building graph payload: {e}")
                    return

                # Stage 4: open graph
                try:
                    open_d3_graph_window(payload)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error opening graph in browser: {e}")

    # ========================================================================
    # Main Content (after data is loaded)
    # ========================================================================

    if "data" not in st.session_state:
        st.info("Use **Load graph** in **Data configuration** above to begin.")
        return
    
    dataset = st.session_state["data"]
    max_nodes = st.session_state.get("max_nodes", 150)
    min_degree = st.session_state.get("min_degree", 0)

    data = dataset[0] if hasattr(dataset, "__getitem__") else dataset
    lifting_applied = st.session_state.get("lifting_applied")

    if not hasattr(data, "num_nodes"):
        st.warning("Could not determine dataset shape")
        return

    num_nodes = data.num_nodes
    if num_nodes is None and hasattr(data, "edge_index") and data.edge_index is not None:
        num_nodes = int(data.edge_index.max().item()) + 1
    num_features = (
        data.x.shape[1]
        if (hasattr(data, "x") and data.x is not None and len(data.x.shape) > 1)
        else 0
    )

    if lifting_applied:
        st.success(
            f"⚗️ Viewing lifted data — **{lifting_applied['name']}** "
            f"({lifting_applied['source']} → {lifting_applied['target']}) | "
            f"Nodes: {num_nodes}, Node features: {num_features if num_features else 'N/A'}"
        )
    else:
        st.info(
            f"✅ Dataset loaded! Nodes: {num_nodes}, "
            f"Node features: {num_features if num_features else 'N/A'}"
        )

    graph_ok = edge_index_to_sparse_adj(data) is not None
    inc_ok = (
        incidence_to_sparse_incidence(data) is not None
        or incidence_rank_one_to_sparse(data) is not None
    )
    if not (graph_ok or inc_ok):
        st.warning("No graph/incidence connectivity found for visualization.")
        return

    data_domain = st.session_state.get("data_domain", "graph")
    dataset_name = st.session_state.get("dataset_name", "dataset")
    rank_labels = st.session_state.get("rank_labels") or get_rank_labels(
        data_domain, dataset_name, data
    )

    if graph_ok and inc_ok:
        mode_options = ["auto", "graph", "incidence"]
    elif graph_ok:
        mode_options = ["graph"]
    else:
        mode_options = ["incidence"]
    mode_labels = {
        "auto": "Auto (prefer lifted incidence, else graph)",
        "graph": "Graph (adjacency from pairwise edges)",
        "incidence": "Incidence",
    }
    viz_mode = st.selectbox(
        "Connectivity view",
        options=mode_options,
        format_func=lambda m: mode_labels.get(m, m),
        key="simple_viz_mode",
    )
    matrix, vdesc, relation_ctx = get_primary_visualization_matrix(data, viz_mode)
    if matrix is None:
        st.warning("Could not build the selected connectivity view.")
        return

    stats = get_matrix_stats(matrix)
    if stats:
        st.caption(
            f"{vdesc} — shape {stats['shape']}, "
            f"{stats['num_edges']:,} nonzeros, density {stats['density']:.4%}"
        )
    G, node_degrees = sparse_to_networkx(matrix, max_nodes=max_nodes, min_degree=min_degree)
    if not G or len(G.nodes()) == 0:
        st.warning("Graph is empty with the current max-nodes / degree filters.")
        return

    payload = networkx_to_d3_payload(
        G,
        node_degrees,
        rank_labels=rank_labels,
        relation_context=relation_ctx,
        plot_title=vdesc,
        plot_subtitle=f"{dataset_name} ({data_domain})",
    )
    if st.button("Open graph in D3 (new window)", key="d3_open_current", use_container_width=True):
        open_d3_graph_window(payload)


if __name__ == "__main__":
    main()