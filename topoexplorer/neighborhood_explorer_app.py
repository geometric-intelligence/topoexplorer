"""
Interactive Hypergraph Neighborhood Explorer

Graph views are rendered inline as embedded D3 components, with an optional
download button that exports the same view as a standalone HTML file.

Run with: streamlit run topoexplorer/neighborhood_explorer_app.py
"""

import sys
import os
import copy
import json
import re
import math
from pathlib import Path
import yaml

# Add analysis directory to path for local module imports (e.g. d3_graph_html)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import torch

from d3_graph_html import build_standalone_d3_html
import graph_metrics as gm
import branding
import stats_html
from omegaconf import OmegaConf
from torch_geometric.utils import to_undirected
import rootutils
import configs as _topobench_configs

# Root of the installed topobench configs package (sibling of the topobench
# package in site-packages, e.g. <site-packages>/configs/).
_CONFIGS_ROOT = Path(_topobench_configs.__file__).parent

# When topobench is installed as a package (not cloned locally), rootutils
# cannot find the .project-root marker starting from the pip install location.
# Patch setup_root (the function run.py calls directly) to fall back to the
# topoexplorer repo root, which has the marker file.
_TOPOEXPLORER_ROOT = Path(__file__).parent.parent
_orig_setup_root = rootutils.setup_root

def _setup_root_with_fallback(search_from, indicator=".project-root", **kwargs):
    """Resolve the project root, falling back to the TopoExplorer repo root.

    Wraps ``rootutils.setup_root`` so that when topobench is installed as a pip
    package (and the ``.project-root`` marker cannot be found from the install
    location) the lookup degrades gracefully to this repository's root instead
    of raising.

    Args:
        search_from: Path from which ``rootutils`` starts searching upwards for
            the marker file.
        indicator: Name of the marker file identifying the project root.
        **kwargs: Extra keyword arguments forwarded to ``rootutils.setup_root``
            (e.g. ``pythonpath``).

    Returns:
        pathlib.Path: The resolved project root, or the TopoExplorer repo root
        when the marker cannot be located.
    """
    try:
        return _orig_setup_root(search_from, indicator, **kwargs)
    except FileNotFoundError:
        os.environ.setdefault("PROJECT_ROOT", str(_TOPOEXPLORER_ROOT))
        if kwargs.get("pythonpath", False):
            sys.path.insert(0, str(_TOPOEXPLORER_ROOT))
        return _TOPOEXPLORER_ROOT

rootutils.setup_root = _setup_root_with_fallback


# Domains whose datasets already carry higher-order connectivity (incidence,
# adjacency, Laplacian matrices, etc.) baked in by their loader (e.g. the
# MANTRA simplicial loader runs ``get_complex_connectivity`` during build).
# For these domains the "Use lifting" toggle defaults to off, because applying
# a transform would discard or rebuild the structure that's already there.
_NATIVE_HIGHER_ORDER_DOMAINS = frozenset(
    {"simplicial", "hypergraph", "cell", "combinatorial"}
)


# ============================================================================
# Dataset Discovery Functions
# ============================================================================

@st.cache_resource
def discover_available_datasets():
    """Scan configs/dataset folder and discover all available datasets."""
    datasets_by_domain = {}
    config_dir = _CONFIGS_ROOT / "dataset"
    
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
    liftings_dir = _CONFIGS_ROOT / "transforms" / "liftings"

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
    config_path = _CONFIGS_ROOT / "dataset" / domain / f"{dataset_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict


def _format_num_features(num_features):
    """Format num_features for display (scalar or list)."""
    if num_features is None:
        return "N/A"
    if isinstance(num_features, (list, tuple)):
        return ", ".join(str(x) for x in num_features)
    return str(num_features)


def extract_dataset_metadata(domain, dataset_name):
    """Read descriptive dataset fields from YAML (no loader invocation)."""
    try:
        dataset_yaml = load_dataset_config(domain, dataset_name)
    except (FileNotFoundError, yaml.YAMLError, OSError):
        return {}
    params = dataset_yaml.get("parameters") or {}
    split_params = dataset_yaml.get("split_params") or {}
    return {
        "task": params.get("task"),
        "task_level": params.get("task_level"),
        "learning_setting": split_params.get("learning_setting"),
        "num_features": params.get("num_features"),
        "num_classes": params.get("num_classes"),
        "split_type": split_params.get("split_type"),
    }

# ============================================================================
# Configuration and Constants
# ============================================================================

st.set_page_config(
    page_title="TopoExplorer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fixed, slightly-wider sidebar. The native collapse arrow is kept as a
# "focus mode" so the user can momentarily hide controls for a full-width graph.
_SIDEBAR_TARGET_WIDTH_PX = 560
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] {{
        min-width: {_SIDEBAR_TARGET_WIDTH_PX}px;
        max-width: {_SIDEBAR_TARGET_WIDTH_PX}px;
        width: {_SIDEBAR_TARGET_WIDTH_PX}px;
    }}
    /* Main canvas: tight top margin so headers/content sit near the top border.
       The fixed header is made transparent (below) so top-of-page banners and
       the load spinner remain visible despite the small padding. */
    [data-testid="stMainBlockContainer"], .block-container {{
        padding-top: 1rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100%;
    }}
    /* Transparent, zero-height header so it never covers top-of-page content. */
    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);
        height: 0;
    }}
    [data-testid="stSidebar"] {{
        position: relative;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {{
        position: absolute;
        top: 0.5rem;
        left: 0.35rem;
        z-index: 10;
        background: transparent;
    }}
    .topo-sidebar-brand {{
        text-align: center;
        margin: 0.5rem 2.25rem 0.75rem 2.25rem;
    }}
    .topo-sidebar-title {{
        text-align: center;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: {branding.BRAND["ink"]};
        line-height: 1.1;
    }}
    .topo-sidebar-tagline {{
        text-align: center;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-top: 0.15rem;
        color: {branding.BRAND["muted"]};
    }}
    [data-testid="stSidebarHeader"] {{
        min-height: 0;
        padding: 0;
        border: none;
    }}
    [data-testid="stSidebarUserContent"] {{ padding-top: 0.25rem; }}
    /* Large, full-width sidebar tab buttons only. */
    [data-testid="stSidebar"] .st-key-tab_load_btn button,
    [data-testid="stSidebar"] .st-key-tab_explore_btn button,
    [data-testid="stSidebar"] .st-key-tab_metrics_btn button {{
        height: 3.2rem;
        font-size: 1rem;
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_RANK_PALETTE = [
    "#1f77b4",  # rank 0 - blue   (Nodes)
    "#ff7f0e",  # rank 1 - orange (Edges)
    "#2ca02c",  # rank 2 - green  (Faces)
    "#d62728",  # rank 3 - red    (Volumes)
    "#ff2dbf",  # rank 4 - bright magenta (4-cells); was purple #9467bd,
                # swapped to avoid reading as bluish next to rank-1
                # orange adjacencies.
    "#8c564b",  # rank 5 - brown
    "#9467bd",  # rank 6 - purple (was pink #e377c2; rotated here so rank 4
                # can take the more eye-catching magenta slot).
    "#7f7f7f",  # rank 7 - gray
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


def blend_hex(*hex_colors):
    """Average several ``#rrggbb`` colors channel-wise.

    Returns ``"#888888"`` if nothing parseable is provided. Used to color
    adjacency edges that come from multiple via-ranks (overlap blend) and
    as the 3D solid fallback for incidence edges.
    """
    rs, gs, bs = [], [], []
    for c in hex_colors:
        if not isinstance(c, str):
            continue
        c = c.strip()
        if c.startswith("#"):
            c = c[1:]
        if len(c) != 6:
            continue
        try:
            rs.append(int(c[0:2], 16))
            gs.append(int(c[2:4], 16))
            bs.append(int(c[4:6], 16))
        except ValueError:
            continue
    if not rs:
        return "#888888"
    r = sum(rs) // len(rs)
    g = sum(gs) // len(gs)
    b = sum(bs) // len(bs)
    return f"#{r:02x}{g:02x}{b:02x}"


NEIGHBORHOOD_COLOR_MODES = ("rank_gradient", "unique_solid")


def _neighborhood_color_kind(nb_id):
    """Classify a neighborhood id for colormap sampling (adjacency vs incidence)."""
    if nb_id in ("graph",):
        return "adjacency"
    if nb_id in ("hyperedges", "incidence_hyperedges"):
        return "incidence"
    if nb_id.startswith("adjacency_"):
        return "adjacency"
    if nb_id.startswith("incidence_"):
        return "incidence"
    try:
        _r, _direction, ntype, _src = parse_neighborhood(nb_id)
        return "adjacency" if ntype == "adjacency" else "incidence"
    except Exception:
        return "incidence"


# Solid neighborhood colours: cool viridis band vs warm inferno band (no shared
# dark-purple anchors). Golden-ratio sampling spreads hues when many are selected.
_ADJacency_CMAP = "viridis"
_ADJacency_CMAP_RANGE = (0.38, 0.82)  # teal → green; skip pale yellow end
_INCIDENCE_CMAP = "inferno"
_INCIDENCE_CMAP_RANGE = (0.48, 0.86)  # red → orange; skip pale yellow end


def _sample_colormap_hex(cmap_name, n, *, start=0.12, end=0.88, spread="golden"):
    """Sample ``n`` distinct hex colors from a matplotlib colormap.

    With ``spread="golden"``, positions are golden-ratio spaced within
    ``[start, end]`` so consecutive neighborhoods are less likely to look alike.
    """
    if n <= 0:
        return []
    import matplotlib.colors as mcolors
    import numpy as np

    try:
        from matplotlib import colormaps

        cmap = colormaps[cmap_name]
    except Exception:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(cmap_name)
    if n == 1:
        mid = (start + end) / 2.0
        return [mcolors.to_hex(cmap(mid))]
    span = end - start
    if spread == "golden" and n > 2:
        # Golden-ratio conjugate (1/phi). Stepping the colormap by this
        # irrational fraction spreads successive colors far apart, avoiding the
        # adjacent near-duplicates that uniform sampling produces.
        phi = 0.618033988749895
        ts = sorted(start + ((i * phi) % 1.0) * span for i in range(n))
    else:
        ts = np.linspace(start, end, n)
    return [mcolors.to_hex(cmap(float(t))) for t in ts]


def _build_neighborhood_color_map(neigh_ids):
    """Assign one solid color per selected neighborhood (cool vs warm bands)."""
    adj_ids = [nid for nid in neigh_ids if _neighborhood_color_kind(nid) == "adjacency"]
    inc_ids = [nid for nid in neigh_ids if _neighborhood_color_kind(nid) == "incidence"]
    adj_lo, adj_hi = _ADJacency_CMAP_RANGE
    inc_lo, inc_hi = _INCIDENCE_CMAP_RANGE
    adj_colors = _sample_colormap_hex(
        _ADJacency_CMAP, len(adj_ids), start=adj_lo, end=adj_hi,
    )
    inc_colors = _sample_colormap_hex(
        _INCIDENCE_CMAP, len(inc_ids), start=inc_lo, end=inc_hi,
    )
    out = {}
    for nid, color in zip(adj_ids, adj_colors):
        out[nid] = color
    for nid, color in zip(inc_ids, inc_colors):
        out[nid] = color
    return out


def _expand_neighborhood_color_aliases(neigh_ids, color_map):
    """Map canonical reconstructed ids and basic ``incidence_k`` aliases to colors."""
    expanded = dict(color_map)
    for nid in neigh_ids:
        color = color_map.get(nid)
        if not color:
            continue
        expanded[nid] = color
        if nid.startswith("incidence_"):
            try:
                k = int(nid.split("_", 1)[1])
            except ValueError:
                continue
            src = max(k - 1, 0)
            expanded[f"1-up_incidence-{src}"] = color
        elif nid.startswith("adjacency_"):
            try:
                k = int(nid.split("_", 1)[1])
            except ValueError:
                continue
            expanded[f"1-up_adjacency-{k}"] = color
            expanded[f"adjacency-{k}"] = color
    return expanded


def _resolve_link_neighborhood_id(link, neigh_ids):
    """Best-effort match from a rendered link back to a selected neighborhood id."""
    selected = set(neigh_ids)
    if link.get("neighborhoodId") in selected:
        return link["neighborhoodId"]

    kind = link.get("kind") or "incidence"
    if kind == "adjacency":
        via_ranks = link.get("viaRanks") or []
        via = int(via_ranks[0]) if via_ranks else link.get("srcRank")
        src = link.get("srcRank")
        canon = _neighborhood_id_for_adjacency(src, via)
        if canon in selected:
            return canon
        for nid in neigh_ids:
            if _neighborhood_color_kind(nid) != "adjacency":
                continue
            cls = _classify_id_simple(nid)
            if cls and cls[1] == src and cls[2] == via:
                return nid
    else:
        direction = link.get("direction") or "up"
        src = link.get("srcRank")
        tgt = link.get("tgtRank")
        canon = _neighborhood_id_for_incidence(src, tgt, direction)
        if canon in selected:
            return canon
        if direction == "both":
            lo, hi = sorted((int(src), int(tgt)))
            for cand_dir in ("up", "down"):
                alt = _neighborhood_id_for_incidence(lo, hi, cand_dir)
                if alt in selected:
                    return alt
        for nid in neigh_ids:
            if _neighborhood_color_kind(nid) != "incidence":
                continue
            cls = _classify_id_simple(nid)
            if not cls or cls[0] != "incidence":
                continue
            lo, hi = sorted((int(src), int(tgt)))
            if cls[1] == lo and cls[2] == hi:
                return nid

    if len(neigh_ids) == 1:
        return neigh_ids[0]
    return None


def _classify_id_simple(nid):
    """Lightweight ``(kind, a, b)`` classifier for color aliasing (no data required)."""
    if nid.startswith("incidence_") and nid != "incidence_hyperedges":
        try:
            k = int(nid.split("_", 1)[1])
        except ValueError:
            return None
        if k <= 0:
            return None
        return ("incidence", max(k - 1, 0), k)
    if nid.startswith("adjacency_"):
        try:
            k = int(nid.split("_", 1)[1])
        except ValueError:
            return None
        return ("adjacency", k, k + 1)
    if "-" in nid:
        try:
            _r, _direction, ntype, _src = parse_neighborhood(nid)
        except Exception:
            return None
        if ntype == "adjacency":
            via = _src + _r if _direction == "up" else _src - _r
            return ("adjacency", _src, via)
        tgt = _neighborhood_target_rank(nid)
        lo, hi = sorted((_src, tgt))
        return ("incidence", lo, hi)
    if nid in ("graph",):
        return ("adjacency", 0, 1)
    if nid in ("hyperedges", "incidence_hyperedges"):
        return ("incidence", 0, 1)
    return None


def _apply_neighborhood_color_mode(payload, neigh_ids, color_mode):
    """Apply rank-gradient (default) or unique-solid neighborhood edge colors."""
    if not payload:
        return
    if color_mode != "unique_solid" or not neigh_ids:
        payload["neighborhoodColorMode"] = "rank_gradient"
        return

    color_map = _build_neighborhood_color_map(neigh_ids)
    expanded = _expand_neighborhood_color_aliases(neigh_ids, color_map)

    for link in payload.get("links") or []:
        nb_id = _resolve_link_neighborhood_id(link, neigh_ids)
        solid = expanded.get(nb_id) if nb_id else None
        if solid is None and len(neigh_ids) == 1:
            solid = color_map.get(neigh_ids[0])
        if not solid:
            continue
        link["color"] = solid
        link["neighborhoodId"] = nb_id or neigh_ids[0]
        link.pop("colorStart", None)
        link.pop("colorEnd", None)

    payload["neighborhoodColorMode"] = "unique_solid"
    payload["relationsLegend"] = _build_relations_legend(
        payload.get("links"),
        color_mode="unique_solid",
        solid_color_map=expanded,
    )

_FRIENDLY_RANK_NAMES = {0: "Nodes", 1: "Edges", 2: "Faces", 3: "Volumes"}


def friendly_rank_label(rank, rank_labels=None) -> str:
    """Short human-friendly cell label for a rank.

    Prefers an explicit override from ``rank_labels`` (when not the generic
    ``"Rank N"`` placeholder), then the canonical Nodes/Edges/Faces/Volumes
    naming, then ``"k-cells"`` for higher ranks.
    """
    if rank is None:
        return "Cells"
    rk = int(rank)
    if rank_labels:
        custom = rank_labels.get(rk)
        if isinstance(custom, str) and custom and custom != f"Rank {rk}":
            return custom
    if rk in _FRIENDLY_RANK_NAMES:
        return _FRIENDLY_RANK_NAMES[rk]
    return f"{rk}-cells"


_CELLS_TERM_BY_RANK = {0: "Nodes", 1: "Edges", 2: "Faces", 3: "Volumes"}


def _cells_term(rank, *, target_kind=None):
    """Return the universal cell-dimension label for a rank.

    Uses the simplicial/cell-complex terminology requested in the legend:
    ``0`` → ``Nodes``, ``1`` → ``Edges``, ``2`` → ``Faces``, ``3`` →
    ``Volumes``, and ``k`` → ``"<k>-cells"`` for ``k ≥ 4``. The
    ``target_kind="hyperedge"`` override is used for the special
    ``incidence_hyperedges`` view.
    """
    if target_kind == "hyperedge":
        return "Hyperedges"
    if rank is None:
        return "Cells"
    try:
        r = int(rank)
    except Exception:
        return str(rank)
    if r in _CELLS_TERM_BY_RANK:
        return _CELLS_TERM_BY_RANK[r]
    return f"{r}-cells"


def _neighborhood_id_for_incidence(src_rank, tgt_rank, direction):
    """Reconstruct the canonical ``r-direction_incidence-s`` id.

    Returns ``None`` if the inputs don't form a valid neighborhood spec.
    """
    if src_rank is None or tgt_rank is None or direction not in ("up", "down"):
        return None
    s = int(src_rank)
    t = int(tgt_rank)
    if direction == "up":
        r = t - s
        return f"{r}-up_incidence-{s}" if r > 0 else None
    r = s - t
    return f"{r}-down_incidence-{s}" if r > 0 else None


def _neighborhood_id_for_adjacency(src_rank, via_rank):
    """Reconstruct the canonical ``r-direction_adjacency-s`` id."""
    if src_rank is None or via_rank is None:
        return None
    s = int(src_rank)
    v = int(via_rank)
    if v > s:
        return f"{v - s}-up_adjacency-{s}"
    if v < s:
        return f"{s - v}-down_adjacency-{s}"
    return f"adjacency-{s}"


def _build_relations_legend(
    links_out,
    rank_labels=None,
    *,
    color_mode="rank_gradient",
    solid_color_map=None,
):
    """Aggregate per-link metadata in ``links_out`` into legend rows.

    Returns one entry per unique relation rendered, with enough data for
    the JS renderer to draw a small two-node + edge glyph that mirrors
    the actual plot (gradient + directional arrow for incidences in
    rank-gradient mode; solid colour per neighborhood in unique-solid mode).
    Each row's label combines the canonical neighborhood id with
    cell-dimension terminology, e.g.
    ``"2-down_incidence-2: Faces → Nodes"`` or
    ``"1-up_adjacency-0: Nodes ↔ Nodes via Edges"``.
    """
    # ``rank_labels`` is accepted for API parity with the rank legend but
    # the relations legend deliberately uses the universal cell-dimension
    # terminology requested by the user, so the parameter is unused here.
    del rank_labels

    solid = color_mode == "unique_solid"
    solid_color_map = solid_color_map or {}

    seen = set()
    out = []

    def _solid_for_nb(nb_id, fallback):
        """Return the per-neighborhood solid color, or ``fallback`` if unset."""
        if nb_id and nb_id in solid_color_map:
            return solid_color_map[nb_id]
        return fallback

    def _emit_incidence(src_rank, tgt_rank, direction, *,
                        color, color_start, color_end, target_kind):
        """Append a deduplicated incidence entry to the relations legend."""
        key = ("incidence", src_rank, tgt_rank, direction, target_kind)
        if key in seen:
            return
        seen.add(key)
        nb_id = _neighborhood_id_for_incidence(src_rank, tgt_rank, direction)
        src_term = _cells_term(src_rank)
        tgt_term = _cells_term(tgt_rank, target_kind=target_kind)
        body = f"{src_term} → {tgt_term}"
        label = f"{nb_id}: {body}" if nb_id else body
        # Endpoint circles always mirror the rank colors of the actual
        # plot nodes (e.g. rank-0 blue, rank-1 orange). Only the *line*
        # color differs between modes: a per-neighborhood solid in
        # unique-solid mode, the rank-to-rank gradient otherwise. This
        # keeps the glyph readable against the plot in both modes.
        src_endpoint = rank_color(src_rank) if src_rank is not None else "#888888"
        tgt_endpoint = rank_color(tgt_rank) if tgt_rank is not None else "#888888"
        if solid:
            solid_c = _solid_for_nb(nb_id, color)
            out.append({
                "kind": "incidence",
                "srcRank": src_rank,
                "tgtRank": tgt_rank,
                "srcColor": src_endpoint,
                "tgtColor": tgt_endpoint,
                "color": solid_c,
                "direction": direction,
                "neighborhoodId": nb_id,
                "label": label,
            })
        else:
            out.append({
                "kind": "incidence",
                "srcRank": src_rank,
                "tgtRank": tgt_rank,
                "srcColor": src_endpoint,
                "tgtColor": tgt_endpoint,
                "colorStart": color_start,
                "colorEnd": color_end,
                "color": color,
                "direction": direction,
                "neighborhoodId": nb_id,
                "label": label,
            })

    for link in links_out or []:
        kind = link.get("kind") or "incidence"
        if kind == "adjacency":
            via_ranks = link.get("viaRanks") or []
            via = int(via_ranks[0]) if via_ranks else None
            # The visual src/tgt rank of an adjacency edge is the same -
            # both endpoints live at the same rank. Try to recover it
            # from ``srcRank``/``tgtRank`` first, then fall back to the
            # via-rank as a defensive default.
            src_rank = link.get("srcRank")
            if src_rank is None:
                src_rank = via
            # Pseudo-neighborhoods (e.g. ``"graph"`` for the pairwise
            # edge_index backbone) carry an explicit legend label and
            # disable the technical neighborhood-id prefix so the entry
            # reads as e.g. ``"Graph backbone"`` instead of the
            # awkward ``"0-up_adjacency-0: Nodes ↔ Nodes via Nodes"``.
            override_label = link.get("legendLabel")
            suppress_nb_id = bool(link.get("suppressNeighborhoodId"))
            key = (
                "adjacency",
                int(src_rank) if src_rank is not None else None,
                via,
                override_label or "",
            )
            if key in seen:
                continue
            seen.add(key)
            if suppress_nb_id:
                nb_id = None
            else:
                nb_id = link.get("neighborhoodId") or _neighborhood_id_for_adjacency(src_rank, via)
            # Endpoint circles in the legend always mirror the actual
            # plot's rank colour for ``src_rank`` (adjacencies are
            # within-rank, so both endpoints share the same colour). Only
            # the line colour changes between modes: solid per-neighborhood
            # in unique-solid mode, mediating via-rank colour otherwise.
            src_color = rank_color(src_rank) if src_rank is not None else "#888888"
            if solid:
                color = _solid_for_nb(nb_id, link.get("color") or "#888888")
            else:
                color = link.get("color") or (rank_color(via) if via is not None else "#888888")
            if override_label:
                label = override_label
            else:
                term = _cells_term(src_rank)
                via_term = _cells_term(via)
                body = f"{term} ↔ {term} via {via_term}"
                label = f"{nb_id}: {body}" if nb_id else body
            out.append({
                "kind": "adjacency",
                "srcRank": src_rank,
                "tgtRank": src_rank,
                "srcColor": src_color,
                "tgtColor": src_color,
                "color": color,
                "viaRank": via,
                "neighborhoodId": nb_id,
                "label": label,
            })
        else:
            src_rank = link.get("srcRank")
            tgt_rank = link.get("tgtRank")
            direction = link.get("direction") or "up"
            target_kind = link.get("targetKind")
            color = link.get("color")
            color_start = link.get("colorStart")
            color_end = link.get("colorEnd")
            if direction == "both" and src_rank is not None and tgt_rank is not None:
                # Both an up- and a down-incidence spec contributed to
                # this edge. Show the user each direction separately so
                # the canonical ids still match what they selected.
                lo = min(int(src_rank), int(tgt_rank))
                hi = max(int(src_rank), int(tgt_rank))
                _emit_incidence(
                    lo, hi, "up",
                    color=color, color_start=color_start,
                    color_end=color_end, target_kind=target_kind,
                )
                _emit_incidence(
                    hi, lo, "down",
                    color=color, color_start=color_end,
                    color_end=color_start, target_kind=target_kind,
                )
            else:
                _emit_incidence(
                    src_rank, tgt_rank, direction,
                    color=color, color_start=color_start,
                    color_end=color_end, target_kind=target_kind,
                )
    return out

def _build_legend(ranks, rank_labels=None):
    """Legend entries: list of ``{rank, color, label}`` for the given ranks.

    When ``rank_labels`` describes more ranks than ``ranks`` does (e.g. the
    dataset has cells up to faces but the current view only contains
    rank-0 nodes), we *always* include every known rank in the legend.
    Adjacency edges colour themselves by their mediating via-rank, so the
    user can encounter rank colours that don't correspond to any node
    layer currently on screen; showing all dataset ranks keeps those
    colour-to-rank associations visible.
    """
    rank_set = {int(x) for x in ranks}
    if rank_labels:
        for r in rank_labels.keys():
            try:
                rank_set.add(int(r))
            except (TypeError, ValueError):
                continue
    out = []
    for r in sorted(rank_set):
        out.append({
            "rank": r,
            "color": rank_color(r),
            "label": friendly_rank_label(r, rank_labels),
        })
    return out

# ============================================================================
# Data Loading Functions
# ============================================================================

def _strip_unresolvable_interpolations(d):
    """Recursively replace unresolvable ${...} strings with None in a dict."""
    if isinstance(d, dict):
        return {k: _strip_unresolvable_interpolations(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_strip_unresolvable_interpolations(v) for v in d]
    if isinstance(d, str) and d.strip().startswith('${'):
        return None
    return d


def load_dataset(domain, dataset_name):
    """Load a dataset by properly resolving config interpolations."""
    config_path = _CONFIGS_ROOT / "dataset" / domain / f"{dataset_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    with open(config_path, 'r') as f:
        dataset_yaml = yaml.safe_load(f)

    project_root = Path(__file__).parent.parent

    base_config = {
        'paths': {
            'root_dir': str(project_root),
            'data_dir': str(project_root / 'datasets'),
            'log_dir': str(project_root / 'logs'),
        },
        # Minimal stand-ins for keys that dataset configs sometimes interpolate
        # from the model side of a full training config (e.g. MANTRA simplicial
        # configs use ``model_domain: ${model.model_domain}``). We default
        # ``model_domain`` to the dataset's own domain so the loader treats the
        # data as native (no implicit lifting).
        'model': {
            'model_name': '',
            'model_domain': domain,
            'backbone': {'neighborhoods': None},
        },
    }

    cfg = OmegaConf.create(base_config)
    dataset_cfg = OmegaConf.create({'dataset': dataset_yaml})
    cfg = OmegaConf.merge(cfg, dataset_cfg)

    try:
        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        # Some configs reference keys we don't model here (e.g. split_params
        # referencing ``${dataset.loader.parameters.data_name}`` is fine, but
        # arbitrary ``${something.else}`` won't resolve). Resolve the loader
        # (which is what we actually need) on its own, with the same base
        # ``paths``/``model`` available, and fill the rest with best-effort
        # unresolved values so the descriptive UI still has something to show.
        loader_cfg = OmegaConf.create({
            'paths': base_config['paths'],
            'model': base_config['model'],
            'dataset': {'loader': dataset_yaml.get('loader', {})},
        })
        try:
            loader_resolved = OmegaConf.to_container(loader_cfg, resolve=True)
        except Exception:
            # As a last resort, drop the still-unresolvable interpolations.
            loader_only_unresolved = OmegaConf.to_container(loader_cfg, resolve=False)
            loader_only_unresolved = _strip_unresolvable_interpolations(loader_only_unresolved)
            loader_resolved = loader_only_unresolved
        full_unresolved = OmegaConf.to_container(cfg, resolve=False)
        full_unresolved = _strip_unresolvable_interpolations(full_unresolved)
        cfg_resolved = full_unresolved
        cfg_resolved['dataset']['loader'] = loader_resolved['dataset']['loader']

    resolved_dataset = cfg_resolved['dataset']

    loader_target = resolved_dataset['loader'].get('_target_')
    if not loader_target:
        raise ValueError("No loader target found in config")

    class_name = loader_target.split('.')[-1]

    from topobench.data import loaders as loaders_module
    loader_class = getattr(loaders_module, class_name)

    loader_params = OmegaConf.create(resolved_dataset['loader']['parameters'])

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
    """Count the stored (non-zero) entries of a sparse COO tensor.

    Args:
        sp: A ``torch.sparse_coo_tensor``, or ``None``.

    Returns:
        int: Number of stored values after coalescing, or ``0`` if ``sp`` is
        ``None``.
    """
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
    """List the ranks ``k`` for which an ``incidence_k`` matrix exists.

    Args:
        data: A topobench data object whose keys may include ``incidence_k``
            entries.

    Returns:
        list[int]: The ranks ``k`` present, sorted in descending order (highest
        rank first).
    """
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
            src_rank = max(k - 1, 0)
            return (
                sp,
                f"Rank-{src_rank} to rank-{k} incidence",
                {"type": "bipartite", "source_rank": src_rank, "target_rank": k},
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


def detect_max_rank(data):
    """Return the maximum cell rank present on ``data``, or -1 if none found."""
    max_k = -1
    for key in _data_keys(data):
        m = re.match(r"^incidence_(\d+)$", key)
        if m:
            max_k = max(max_k, int(m.group(1)))
        m = re.match(r"^adjacency_(\d+)$", key)
        if m:
            max_k = max(max_k, int(m.group(1)))
    return max_k


_RAW_CONNECTIVITY_PREFIXES = (
    "incidence", "adjacency", "coadjacency",
    "up_laplacian", "down_laplacian", "hodge_laplacian",
)


def build_raw_connectivity_dict(data, max_rank):
    """Collect raw connectivity matrices from ``data`` into a dict.

    Matches the key schema produced by ``get_complex_connectivity``:
    ``{prefix}_{k}`` for each prefix and rank 0..max_rank.
    """
    connectivity = {}
    for prefix in _RAW_CONNECTIVITY_PREFIXES:
        for k in range(max_rank + 1):
            key = f"{prefix}_{k}"
            attr = _get_data_attr(data, key)
            if attr is not None:
                sp = _to_sparse_coo(attr)
                if sp is not None:
                    connectivity[key] = sp
    return connectivity


def generate_all_neighborhoods(max_rank):
    """Generate all valid TopoBench neighborhood strings for a given max_rank.

    Returns a flat list of strings in ``r-direction_type-src_rank`` format.
    """
    neighborhoods = []
    for src_rank in range(max_rank + 1):
        for r in range(1, max_rank + 1):
            if src_rank + r <= max_rank:
                neighborhoods.append(f"{r}-up_adjacency-{src_rank}")
                neighborhoods.append(f"{r}-up_incidence-{src_rank}")
    for src_rank in range(1, max_rank + 1):
        for r in range(1, max_rank + 1):
            if src_rank - r >= 0:
                neighborhoods.append(f"{r}-down_adjacency-{src_rank}")
                neighborhoods.append(f"{r}-down_incidence-{src_rank}")
    return neighborhoods


def parse_neighborhood(neighborhood_str):
    """Parse ``r-direction_type-src_rank`` into ``(r, direction, ntype, src_rank)``.

    Also accepts the short form ``direction_type-src_rank`` (r defaults to 1).
    """
    parts = neighborhood_str.split("-")
    if len(parts) == 3:
        r = int(parts[0])
        direction, ntype = parts[1].split("_", 1)
        src_rank = int(parts[2])
    elif len(parts) == 2:
        r = 1
        direction, ntype = parts[0].split("_", 1)
        src_rank = int(parts[1])
    else:
        raise ValueError(f"Cannot parse neighborhood string: {neighborhood_str!r}")
    return r, direction, ntype, src_rank


def _neighborhood_target_rank(neighborhood_str):
    """Return the target rank for a TopoBench neighborhood string."""
    r, direction, _ntype, src_rank = parse_neighborhood(neighborhood_str)
    if direction == "up":
        return src_rank + r
    return src_rank - r


def _describe_neighborhood(neighborhood_str, rank_labels=None):
    """Human-readable label for a TopoBench neighborhood string."""
    r, direction, ntype, src_rank = parse_neighborhood(neighborhood_str)
    arrow = "↑" if direction == "up" else "↓"
    if rank_labels:
        src_label = rank_labels.get(src_rank, f"Rank {src_rank}")
    else:
        src_label = f"Rank {src_rank}"

    if ntype == "adjacency":
        via_rank = src_rank + r if direction == "up" else src_rank - r
        if rank_labels:
            via_label = rank_labels.get(via_rank, f"Rank {via_rank}")
        else:
            via_label = f"Rank {via_rank}"
        return f"{src_label} ↔ {src_label} (via {via_label}, r={r}) {arrow}"
    else:
        tgt_rank = src_rank + r if direction == "up" else src_rank - r
        if rank_labels:
            tgt_label = rank_labels.get(tgt_rank, f"Rank {tgt_rank}")
        else:
            tgt_label = f"Rank {tgt_rank}"
        return f"{src_label} → {tgt_label} (r={r}) {arrow}"


def compute_all_topobench_neighborhoods(data):
    """Compute all valid non-empty TopoBench neighborhoods for ``data``.

    Returns a dict mapping neighborhood strings to sparse COO tensors,
    or an empty dict if the data has no lifted connectivity.
    """
    max_rank = detect_max_rank(data)
    if max_rank < 1:
        return {}

    connectivity = build_raw_connectivity_dict(data, max_rank)
    if not connectivity:
        return {}

    all_nbrs = generate_all_neighborhoods(max_rank)
    if not all_nbrs:
        return {}

    try:
        from topobench.data.utils import select_neighborhoods_of_interest
    except ImportError:
        return {}

    result = {}
    for nb_str in all_nbrs:
        try:
            computed = select_neighborhoods_of_interest(
                connectivity, [nb_str]
            )
        except Exception:
            continue
        tensor = computed.get(nb_str)
        if tensor is None:
            continue
        try:
            sp = tensor.coalesce() if tensor.is_sparse else tensor
            if sp._nnz() > 0:
                result[nb_str] = sp
        except Exception:
            continue
    return result


def enumerate_neighborhoods(data):
    """List non-empty TopoBench-format neighborhoods on a lifted ``Data`` object.

    Returns
    -------
    list[dict]
        Items shaped like
        ``{"id": str, "label": str, "kind": str, "rank": int | None}``
        where ``kind`` is ``tb_incidence`` or ``tb_adjacency``.
    """
    out = []

    # Base structures: the plain graph (pairwise adjacency) and, when present,
    # the node x hyperedge incidence. These let the user view only the
    # underlying (hyper)graph without any higher-order neighborhood overlay.
    if edge_index_to_sparse_adj(data) is not None:
        out.append({
            "id": "graph",
            "label": "Graph — pairwise adjacency",
            "kind": "graph",
            "rank": 0,
        })
    if incidence_to_sparse_incidence(data) is not None:
        out.append({
            "id": "hyperedges",
            "label": "Hyperedges — node × hyperedge incidence",
            "kind": "hyperedges",
            "rank": 0,
        })

    tb_cache = st.session_state.get("_topobench_neighborhoods") or {}
    type_order = ["up_incidence", "down_incidence", "up_adjacency", "down_adjacency"]
    for ntype in type_order:
        matching = sorted(
            (k for k in tb_cache if ntype in k),
            key=lambda s: (parse_neighborhood(s)[3], parse_neighborhood(s)[0]),
        )
        for nb_str in matching:
            _r, _direction, kind_str, src_rank = parse_neighborhood(nb_str)
            kind_tag = "tb_adjacency" if kind_str == "adjacency" else "tb_incidence"
            out.append({
                "id": nb_str,
                "label": f"{nb_str} — {_describe_neighborhood(nb_str)}",
                "kind": kind_tag,
                "rank": src_rank,
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
        # ``"graph"`` is a pseudo-neighborhood (the raw pairwise edge_index)
        # rather than a topobench operator. The default adjacency legend
        # template ("0-up_adjacency-0: Nodes ↔ Nodes via Nodes") reads
        # oddly here because there's no real mediating rank, so we pin
        # a custom label and signal that no canonical id should be
        # displayed for the entry.
        return (
            adj,
            "Graph (adjacency from edge_index)",
            {
                "type": "adjacency",
                "source_rank": 0,
                "legend_label": "Graph backbone",
                "suppress_neighborhood_id": True,
            },
        )

    if neigh_id == "hyperedges":
        inc = incidence_to_sparse_incidence(data)
        if inc is None:
            return None, None, None
        return (
            inc,
            "incidence_hyperedges (Rank 0 → hyperedges)",
            {
                "type": "bipartite",
                "source_rank": 0,
                "target_kind": "hyperedge",
                "incidence_direction": "up",
            },
        )

    m = re.match(r"^incidence_(\d+)$", neigh_id)
    if m:
        k = int(m.group(1))
        if k <= 0:
            return None, None, None
        sp = incidence_rank_k_to_sparse(data, k)
        if sp is None:
            return None, None, None
        src_rank = max(k - 1, 0)
        return (
            sp,
            f"incidence_{k} (Rank {src_rank} → Rank {k})",
            {
                "type": "bipartite",
                "source_rank": src_rank,
                "target_rank": k,
                "incidence_direction": "up",
            },
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
            {"type": "adjacency", "source_rank": k, "via_rank": k + 1},
        )

    tb_cache = st.session_state.get("_topobench_neighborhoods") or {}
    if neigh_id in tb_cache:
        sp = tb_cache[neigh_id]
        r, direction, ntype, src_rank = parse_neighborhood(neigh_id)
        desc = _describe_neighborhood(neigh_id)
        if ntype == "adjacency":
            via_rank = src_rank + r if direction == "up" else src_rank - r
            relation_ctx = {
                "type": "adjacency",
                "source_rank": src_rank,
                "via_rank": via_rank,
            }
        else:
            tgt_rank = _neighborhood_target_rank(neigh_id)
            # TopoBench's ``select_neighborhoods_of_interest`` always stores
            # incidence matrices in ``(N_target, N_source)`` shape -- both for
            # ``up`` and ``down`` neighborhoods. For the bipartite renderer
            # to label/colour each side correctly *and* draw the arrow in the
            # correct direction (up -> arrow toward highest rank, down ->
            # arrow toward lowest rank), we canonicalise to
            # ``(N_source_rank, N_target_rank)`` so that ``idx[0]`` maps to
            # the operator's source side and ``idx[1]`` to its target side.
            try:
                sp = sp.coalesce().transpose(0, 1).coalesce()
            except Exception:
                pass
            relation_ctx = {
                "type": "bipartite",
                "source_rank": src_rank,
                "target_rank": tgt_rank,
                "incidence_direction": direction,
            }
        return sp, f"{neigh_id} ({desc})", relation_ctx

    return None, None, None


def _incidence_direction_for_id(nid: str) -> str:
    """Return ``'up'`` or ``'down'`` for an incidence neighborhood id.

    Canonical ``incidence_k`` and ``incidence_hyperedges`` matrices are stored
    in ``(N_lower, N_higher)`` orientation, i.e. an *up* incidence. The
    TopoBench-style ``r-direction_incidence-s`` ids encode their direction
    explicitly. Unknown ids fall back to ``"up"`` (the convention used
    throughout the visualiser).
    """
    if not nid:
        return "up"
    if nid in ("hyperedges", "incidence_hyperedges"):
        return "up"
    if nid.startswith("incidence_"):
        return "up"
    if "-" in nid:
        try:
            _r, direction, ntype, _src = parse_neighborhood(nid)
            if ntype == "incidence":
                return direction
        except Exception:
            pass
    return "up"


def pick_default_neighborhood_id(available):
    """Pick a sensible default neighborhood from the catalog.

    Preference: the smallest-r ``up_incidence`` at the lowest source rank
    (typically ``1-up_incidence-0``), then any incidence item, then any item.
    """
    if not available:
        return None

    def _key(n):
        """Sort key preferring low source rank, low r, and up-incidence."""
        try:
            r, direction, ntype, src = parse_neighborhood(n["id"])
        except Exception:
            return (9, 9, 9)
        return (src, r, 0 if direction == "up" else 1)

    incidences = sorted(
        [n for n in available if n["kind"] == "tb_incidence"], key=_key
    )
    if incidences:
        return incidences[0]["id"]
    return available[0]["id"]


def _discover_rank_populations(data):
    """Infer per-rank populations (N_k) and optional hyperedge count."""
    populations = {}

    def _set_rank(rank, count):
        """Record ``count`` as the population of ``rank`` (keeping the max)."""
        try:
            rank_i = int(rank)
            count_i = int(count)
        except Exception:
            return
        if count_i < 0:
            return
        populations[rank_i] = max(populations.get(rank_i, 0), count_i)

    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is not None:
        _set_rank(0, num_nodes)

    graph_adj = edge_index_to_sparse_adj(data)
    if graph_adj is not None:
        _set_rank(0, int(graph_adj.shape[0]))

    for k in _connectivity_keys_present(data, "incidence"):
        sp = incidence_rank_k_to_sparse(data, k)
        if sp is None:
            continue
        sp = sp.coalesce()
        _set_rank(max(k - 1, 0), int(sp.shape[0]))
        _set_rank(k, int(sp.shape[1]))

    for k in _connectivity_keys_present(data, "adjacency"):
        sp = adjacency_rank_k_to_sparse(data, k)
        if sp is None:
            continue
        sp = sp.coalesce()
        _set_rank(k, int(sp.shape[0]))

    hyper = incidence_to_sparse_incidence(data)
    num_hyperedges = None
    if hyper is not None:
        hyper = hyper.coalesce()
        _set_rank(0, int(hyper.shape[0]))
        num_hyperedges = int(hyper.shape[1])
    elif getattr(data, "num_hyperedges", None) is not None:
        try:
            num_hyperedges = int(getattr(data, "num_hyperedges"))
        except Exception:
            num_hyperedges = None

    return dict(sorted(populations.items())), num_hyperedges


def compute_shared_node_sampling(data, *, caps_by_rank, cap_hyperedges=None):
    """Compute canonical sampled node ids per rank for cross-view consistency."""
    rank_pops, num_hyperedges = _discover_rank_populations(data)
    degree_by_rank = {r: {} for r in rank_pops}
    hyper_degree = {}

    def _add_rank_degree(rank, node_id, delta=1):
        """Accumulate ``delta`` into the degree of ``node_id`` at ``rank``."""
        rank = int(rank)
        node_id = int(node_id)
        mp = degree_by_rank.setdefault(rank, {})
        mp[node_id] = mp.get(node_id, 0) + int(delta)

    def _count_adjacency(sp, rank):
        """Add each undirected adjacency edge once to both endpoints' degrees."""
        sp = sp.coalesce()
        idx = sp.indices().numpy()
        seen_pairs = set()
        for i in range(idx.shape[1]):
            u = int(idx[0, i])
            v = int(idx[1, i])
            if u == v:
                continue
            pair = (u, v) if u < v else (v, u)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            _add_rank_degree(rank, pair[0], 1)
            _add_rank_degree(rank, pair[1], 1)

    graph_adj = edge_index_to_sparse_adj(data)
    if graph_adj is not None:
        _count_adjacency(graph_adj, 0)

    for k in _connectivity_keys_present(data, "adjacency"):
        sp = adjacency_rank_k_to_sparse(data, k)
        if sp is None:
            continue
        _count_adjacency(sp, k)

    for k in _connectivity_keys_present(data, "incidence"):
        sp = incidence_rank_k_to_sparse(data, k)
        if sp is None:
            continue
        sp = sp.coalesce()
        idx = sp.indices().numpy()
        src_rank = max(k - 1, 0)
        for i in range(idx.shape[1]):
            _add_rank_degree(src_rank, int(idx[0, i]), 1)
            _add_rank_degree(k, int(idx[1, i]), 1)

    hyper = incidence_to_sparse_incidence(data)
    if hyper is not None:
        hyper = hyper.coalesce()
        idx = hyper.indices().numpy()
        for i in range(idx.shape[1]):
            src = int(idx[0, i])
            he = int(idx[1, i])
            _add_rank_degree(0, src, 1)
            hyper_degree[he] = hyper_degree.get(he, 0) + 1

    selected_by_rank = {}
    normalized_caps = {}
    for rank, pop in rank_pops.items():
        cap = caps_by_rank.get(rank, min(150, pop))
        cap = max(0, min(int(cap), int(pop)))
        normalized_caps[rank] = cap

        deg_map = degree_by_rank.get(rank, {})
        candidates = list(range(int(pop)))
        candidates.sort(key=lambda node_id: (-deg_map.get(node_id, 0), int(node_id)))
        selected_by_rank[rank] = frozenset(candidates[:cap])

    selected_hyperedges = None
    if num_hyperedges is not None:
        if cap_hyperedges is None:
            cap_hyperedges = min(150, int(num_hyperedges))
        cap_hyperedges = max(0, min(int(cap_hyperedges), int(num_hyperedges)))
        hyper_candidates = list(range(int(num_hyperedges)))
        hyper_candidates.sort(key=lambda he: (-hyper_degree.get(he, 0), int(he)))
        selected_hyperedges = frozenset(hyper_candidates[:cap_hyperedges])

    return {
        "selected_by_rank": selected_by_rank,
        "aggregate_degree_by_rank": degree_by_rank,
        "selected_hyperedges": selected_hyperedges,
        "caps_by_rank": normalized_caps,
        "rank_populations": rank_pops,
        "num_hyperedges": num_hyperedges,
    }


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

def sparse_to_networkx(
    sparse_tensor,
    max_nodes=200,
    min_degree=0,
    *,
    allowed_source=None,
    allowed_target=None,
):
    """Convert sparse tensor to NetworkX graph with filtering and optional masks."""
    if sparse_tensor is None:
        return None, {}

    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices().numpy()
    n_edges = indices.shape[1]
    if n_edges == 0:
        return None, {}

    is_adjacency = sparse_tensor.shape[0] == sparse_tensor.shape[1]
    source_allowed = set(allowed_source) if allowed_source is not None else None
    target_allowed = set(allowed_target) if allowed_target is not None else None
    if is_adjacency and target_allowed is None:
        target_allowed = source_allowed

    if is_adjacency:
        node_degrees = {}
        seen_pairs = set()
        kept_pairs = []
        for i in range(n_edges):
            src = int(indices[0, i])
            tgt = int(indices[1, i])
            if src == tgt:
                continue
            if source_allowed is not None and src not in source_allowed:
                continue
            if target_allowed is not None and tgt not in target_allowed:
                continue
            pair = (src, tgt) if src < tgt else (tgt, src)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            kept_pairs.append(pair)
            u, v = pair
            node_degrees[u] = node_degrees.get(u, 0) + 1
            node_degrees[v] = node_degrees.get(v, 0) + 1

        if min_degree > 0:
            node_degrees = {k: v for k, v in node_degrees.items() if v >= min_degree}

        # When the caller passes an explicit mask (i.e. the user
        # deliberately sampled which ids to show), seed every allowed
        # node into the degree map - even with degree 0. Otherwise an
        # isolated vertex (no neighbours in the current adjacency) would
        # silently disappear from the plot, which is wrong for
        # adjacency-only views where degree-0 cells are legitimate.
        if source_allowed is not None:
            for nid in source_allowed:
                node_degrees.setdefault(int(nid), 0)
        if target_allowed is not None:
            for nid in target_allowed:
                node_degrees.setdefault(int(nid), 0)

        if not node_degrees:
            return None, {}

        # Legacy fallback for callers not using canonical masks.
        if source_allowed is None and target_allowed is None and len(node_degrees) > max_nodes:
            top_nodes = sorted(
                node_degrees.keys(),
                key=lambda node_id: (-node_degrees[node_id], int(node_id)),
            )[:max_nodes]
            node_degrees = {k: node_degrees[k] for k in top_nodes}

        valid_nodes = set(node_degrees.keys())
        G = nx.Graph()
        for node in valid_nodes:
            G.add_node(node, degree=node_degrees[node])
        for src, tgt in kept_pairs:
            if src in valid_nodes and tgt in valid_nodes:
                G.add_edge(src, tgt)

        if len(G.nodes()) == 0:
            return None, {}
        return G, node_degrees

    # Bipartite (incidence) path: source and target ids live in disjoint id
    # spaces (e.g. rank-0 cells vs rank-1 cells) and can numerically collide.
    # Track degrees in two separate dicts so they don't get conflated, and
    # always feed back a single combined ``node_degrees`` mapping keyed by the
    # raw original id of each side -- callers tag each node with its layer
    # (``src_*``/``tgt_*``) when looking up the degree.
    src_degrees: dict = {}
    tgt_degrees: dict = {}
    kept_pairs = []
    for i in range(n_edges):
        src = int(indices[0, i])
        tgt = int(indices[1, i])
        if source_allowed is not None and src not in source_allowed:
            continue
        if target_allowed is not None and tgt not in target_allowed:
            continue
        kept_pairs.append((src, tgt))
        src_degrees[src] = src_degrees.get(src, 0) + 1
        tgt_degrees[tgt] = tgt_degrees.get(tgt, 0) + 1

    if min_degree > 0:
        src_degrees = {k: v for k, v in src_degrees.items() if v >= min_degree}
        tgt_degrees = {k: v for k, v in tgt_degrees.items() if v >= min_degree}
    if not src_degrees and not tgt_degrees:
        return None, {}

    # Legacy fallback for callers not using canonical masks. Cap each side
    # separately so we don't accidentally drop all of one side.
    if source_allowed is None and target_allowed is None:
        if len(src_degrees) > max_nodes:
            top_src = sorted(
                src_degrees.keys(),
                key=lambda nid: (-src_degrees[nid], int(nid)),
            )[:max_nodes]
            src_degrees = {k: src_degrees[k] for k in top_src}
        if len(tgt_degrees) > max_nodes:
            top_tgt = sorted(
                tgt_degrees.keys(),
                key=lambda nid: (-tgt_degrees[nid], int(nid)),
            )[:max_nodes]
            tgt_degrees = {k: tgt_degrees[k] for k in top_tgt}

    valid_src = set(src_degrees.keys())
    valid_tgt = set(tgt_degrees.keys())

    G = nx.DiGraph()
    src_present = set()
    tgt_present = set()
    for src, tgt in kept_pairs:
        if src not in valid_src or tgt not in valid_tgt:
            continue
        src_present.add(src)
        tgt_present.add(tgt)

    for node in src_present:
        G.add_node(
            f"src_{node}", bipartite=0, original_id=node,
            degree=src_degrees.get(node, 0),
        )
    for node in tgt_present:
        G.add_node(
            f"tgt_{node}", bipartite=1, original_id=node,
            degree=tgt_degrees.get(node, 0),
        )

    for src, tgt in kept_pairs:
        src_key = f"src_{src}"
        tgt_key = f"tgt_{tgt}"
        if src_key in G and tgt_key in G:
            G.add_edge(src_key, tgt_key)

    if len(G.nodes()) == 0:
        return None, {}

    # Combined degree map: caller looks up by ``(bipartite_side, original_id)``
    # via the per-node ``degree`` attribute; we also return a flat dict keyed
    # by string node id for ``networkx_to_d3_payload`` compatibility (it falls
    # back to the per-node attribute when the raw id isn't found).
    node_degrees = {n: G.nodes[n]["degree"] for n in G.nodes()}
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

        # Prefer the per-node ``degree`` attribute set by the builder (always
        # disambiguated by side for bipartite graphs); fall back to the legacy
        # raw-id lookup or the live graph degree.
        raw_deg = G.nodes[node].get("degree")
        if raw_deg is None:
            raw_deg = node_degrees.get(node, node_degrees.get(deg_key, G.degree(node)))
        degree = _json_safe_float(raw_deg)
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

    if graph_type == "adjacency":
        via_rank = (relation_context or {}).get("via_rank", src_rank)
        adj_c = rank_color(via_rank)
        legend_label_override = (relation_context or {}).get("legend_label")
        suppress_nb_id = bool((relation_context or {}).get("suppress_neighborhood_id"))
        links_out = [
            {
                "source": str(u),
                "target": str(v),
                "color": adj_c,
                "kind": "adjacency",
                "srcRank": int(src_rank) if src_rank is not None else None,
                "viaRanks": [int(via_rank)],
                **({"legendLabel": legend_label_override} if legend_label_override else {}),
                **({"suppressNeighborhoodId": True} if suppress_nb_id else {}),
            }
            for u, v in G.edges()
        ]
    else:
        # Reversed gradient between the two ranks: at the source endpoint
        # paint with the target rank color and vice versa.
        color_at_src = rank_color(target_rank)
        color_at_tgt = rank_color(src_rank)
        midpoint = blend_hex(color_at_src, color_at_tgt)
        incidence_direction = (relation_context or {}).get(
            "incidence_direction", "up"
        )
        incidence_target_kind = (relation_context or {}).get("target_kind")
        links_out = [
            {
                "source": str(u),
                "target": str(v),
                "color": midpoint,
                "colorStart": color_at_src,
                "colorEnd": color_at_tgt,
                "kind": "incidence",
                "srcRank": src_rank,
                "tgtRank": target_rank,
                "direction": incidence_direction,
                "targetKind": incidence_target_kind,
            }
            for u, v in G.edges()
        ]

    if graph_type == "adjacency":
        legend_ranks = [src_rank]
    else:
        legend_ranks = [src_rank, target_rank]

    return {
        "graphType": graph_type,
        "title": title,
        "subtitle": subtitle,
        "nodes": nodes_out,
        "links": links_out,
        "legend": _build_legend(legend_ranks, rank_labels),
        "relationsLegend": _build_relations_legend(links_out, rank_labels),
    }


# ============================================================================
# Layered (multi-incidence) builders
# ============================================================================

def _layered_node_id(rank: int, original_id) -> str:
    """Build a stable, rank-qualified node id for the layered (multi-rank) view.

    Prefixing the original id with its rank keeps ids unique across layers, so
    the same cell index appearing at different ranks does not collide.

    Args:
        rank: The cell rank (layer) the node belongs to.
        original_id: The node's index within its rank.

    Returns:
        str: An id of the form ``"r{rank}_n{original_id}"``.
    """
    return f"r{int(rank)}_n{int(original_id)}"


def build_layered_networkx(
    incidence_specs,
    *,
    max_nodes: int = 200,
    min_degree: int = 0,
    selected_by_rank=None,
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

    specs = []
    for spec in incidence_specs:
        if spec is None or spec[0] is None:
            continue
        sp = spec[0].coalesce()
        sr = int(spec[1])
        tr = int(spec[2])
        direction = str(spec[3]) if len(spec) >= 4 else "up"
        specs.append((sp, sr, tr, direction))
    specs.sort(key=lambda t: t[1])
    if not specs:
        return None, {}, []

    # ``raw_edges`` carries the per-occurrence direction so we can later
    # mark each unique edge with the direction(s) of the incidence
    # operator(s) that produced it. ``layer_nodes`` is purely the (rank,
    # node-id) catalogue.
    raw_edges = []
    layer_nodes = {}
    for sp, sr, tr, direction in specs:
        idx = sp.indices().numpy()
        for i in range(idx.shape[1]):
            src_orig = int(idx[0, i])
            tgt_orig = int(idx[1, i])
            u = _layered_node_id(sr, src_orig)
            v = _layered_node_id(tr, tgt_orig)
            raw_edges.append((u, v, direction))
            layer_nodes.setdefault(sr, {})[u] = src_orig
            layer_nodes.setdefault(tr, {})[v] = tgt_orig

    if not raw_edges:
        return None, {}, []

    degree = {}
    for u, v, _d in raw_edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1

    if min_degree > 0:
        degree = {n: d for n, d in degree.items() if d >= min_degree}
    if not degree:
        return None, {}, []

    layers_sorted = sorted(layer_nodes.keys())

    selected = set()
    if selected_by_rank is not None:
        for rank in layers_sorted:
            chosen_ids = set(selected_by_rank.get(int(rank), set()))
            for node_id in chosen_ids:
                lid = _layered_node_id(rank, node_id)
                if lid in degree:
                    selected.add(lid)
    else:
        num_layers = max(len(layers_sorted), 1)
        per_layer_cap = max(1, math.ceil(max_nodes / num_layers))
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

    # First-pass: collapse per-occurrence directions into a per-edge set so
    # that an edge contributed to by both an up- and a down-incidence ends
    # up tagged ``"both"`` (we then default to "up" when drawing arrows).
    pair_dirs = {}
    for u, v, direction in raw_edges:
        if u not in selected or v not in selected:
            continue
        pair = (u, v) if u < v else (v, u)
        pair_dirs.setdefault(pair, set()).add(direction)

    for (u, v), dirs in pair_dirs.items():
        if len(dirs) == 1:
            edge_direction = next(iter(dirs))
        else:
            edge_direction = "both"
        G.add_edge(u, v, kind="incidence", direction=edge_direction)

    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)
    if len(G.nodes()) == 0:
        return None, {}, []

    node_degrees = {n: G.degree(n) for n in G.nodes()}
    final_layers = sorted({G.nodes[n]["layer"] for n in G.nodes()})
    return G, node_degrees, final_layers


def build_layered_adjacency_networkx(
    adjacency_specs,
    *,
    max_nodes: int = 200,
    min_degree: int = 0,
    selected_by_rank=None,
):
    """Stack multiple adjacency matrices into a layered rank-wise graph.

    Each spec is ``(sparse_matrix, src_rank, via_rank)``. ``via_rank`` is the
    rank that mediates the adjacency (e.g. up_adjacency-s through ``s+r``).
    Edges accumulate the set of via-ranks seen across all specs; downstream
    payload code blends colors when a pair appears in multiple specs.
    """
    if not adjacency_specs:
        return None, {}, []

    normalized = []
    for spec in adjacency_specs:
        sp = spec[0]
        if sp is None:
            continue
        src_rank = int(spec[1])
        via_rank = int(spec[2]) if len(spec) >= 3 else src_rank
        normalized.append((sp.coalesce(), src_rank, via_rank))
    specs = sorted(normalized, key=lambda t: (t[1], t[2]))
    if not specs:
        return None, {}, []

    # pair_via_ranks: (rank, (u, v)) -> set[int] of via-ranks
    pair_via_ranks = {}
    layer_nodes = {}
    for sp, rank, via in specs:
        idx = sp.indices().numpy()
        for i in range(idx.shape[1]):
            src_orig = int(idx[0, i])
            tgt_orig = int(idx[1, i])
            if src_orig == tgt_orig:
                continue
            a, b = (src_orig, tgt_orig) if src_orig < tgt_orig else (tgt_orig, src_orig)
            u = _layered_node_id(rank, a)
            v = _layered_node_id(rank, b)
            key = (rank, (u, v))
            pair_via_ranks.setdefault(key, set()).add(via)
            layer_nodes.setdefault(rank, {})[u] = a
            layer_nodes.setdefault(rank, {})[v] = b

    if not pair_via_ranks:
        return None, {}, []

    degree = {}
    for (_rank, (u, v)) in pair_via_ranks:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1

    if min_degree > 0:
        degree = {n: d for n, d in degree.items() if d >= min_degree}
    if not degree:
        return None, {}, []

    layers_sorted = sorted(layer_nodes.keys())
    selected = set()
    # When the caller passes ``selected_by_rank`` the node ids were chosen
    # deliberately by the user (e.g. a per-rank sample of the current
    # sample's full population). Those nodes must appear in the graph
    # even when they have no adjacency edges - otherwise an "isolated"
    # node is silently hidden, which is confusing for adjacency-only
    # views where degree-0 vertices are legitimate.
    selection_is_explicit = selected_by_rank is not None
    if selection_is_explicit:
        layers_from_selection = {
            int(r) for r, ids in selected_by_rank.items() if ids
        }
        layers_sorted = sorted(set(layers_sorted) | layers_from_selection)
        for rank in layers_sorted:
            chosen_ids = set(selected_by_rank.get(int(rank), set()))
            for node_id in chosen_ids:
                lid = _layered_node_id(rank, node_id)
                selected.add(lid)
                # Make sure the (rank, original_id) mapping exists even
                # for isolated picks that never showed up as an edge
                # endpoint.
                layer_nodes.setdefault(int(rank), {}).setdefault(lid, int(node_id))
    else:
        num_layers = max(len(layers_sorted), 1)
        per_layer_cap = max(1, math.ceil(max_nodes / num_layers))
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

    for (_rank, (u, v)), via_set in pair_via_ranks.items():
        if u not in selected or v not in selected:
            continue
        G.add_edge(
            u, v,
            kind="adjacency",
            via_ranks=sorted(int(x) for x in via_set),
        )

    if not selection_is_explicit:
        # Auto-pick path: still strip degree-0 leftovers so the view
        # doesn't fill up with random isolated picks the user did not ask
        # for. Explicit selection always keeps every picked node.
        isolated = [n for n in G.nodes() if G.degree(n) == 0]
        G.remove_nodes_from(isolated)
    if len(G.nodes()) == 0:
        return None, {}, []

    node_degrees = {n: G.degree(n) for n in G.nodes()}
    final_layers = sorted({G.nodes[n]["layer"] for n in G.nodes()})
    return G, node_degrees, final_layers


def build_combined_layered_networkx(
    incidence_specs,
    adjacency_specs,
    *,
    max_nodes: int = 200,
    min_degree: int = 0,
    selected_by_rank=None,
):
    """Build a layered graph from incidence (cross-rank) + adjacency (within-rank).

    Adjacency specs are 3-tuples ``(sparse, src_rank, via_rank)``; edges
    accumulate the set of via-ranks observed across all selected adjacencies.
    """
    specs_inc = []
    for spec in incidence_specs:
        if spec is None or spec[0] is None:
            continue
        sp = spec[0].coalesce()
        sr = int(spec[1])
        tr = int(spec[2])
        direction = str(spec[3]) if len(spec) >= 4 else "up"
        specs_inc.append((sp, sr, tr, direction))
    specs_inc.sort(key=lambda t: t[1])

    normalized_adj = []
    for spec in adjacency_specs:
        sp = spec[0]
        if sp is None:
            continue
        src_rank = int(spec[1])
        via_rank = int(spec[2]) if len(spec) >= 3 else src_rank
        normalized_adj.append((sp.coalesce(), src_rank, via_rank))
    specs_adj = sorted(normalized_adj, key=lambda t: (t[1], t[2]))

    if not specs_inc and not specs_adj:
        return None, {}, []

    raw_inc_edges = []  # list of (u, v, direction)
    adj_pair_via = {}  # (u, v) -> set[int]  with u<v lexicographically
    layer_nodes = {}

    for sp, sr, tr, direction in specs_inc:
        idx = sp.indices().numpy()
        for i in range(idx.shape[1]):
            src_orig = int(idx[0, i])
            tgt_orig = int(idx[1, i])
            u = _layered_node_id(sr, src_orig)
            v = _layered_node_id(tr, tgt_orig)
            raw_inc_edges.append((u, v, direction))
            layer_nodes.setdefault(sr, {})[u] = src_orig
            layer_nodes.setdefault(tr, {})[v] = tgt_orig

    for sp, rank, via in specs_adj:
        idx = sp.indices().numpy()
        for i in range(idx.shape[1]):
            src_orig = int(idx[0, i])
            tgt_orig = int(idx[1, i])
            if src_orig == tgt_orig:
                continue
            a, b = (src_orig, tgt_orig) if src_orig < tgt_orig else (tgt_orig, src_orig)
            u = _layered_node_id(rank, a)
            v = _layered_node_id(rank, b)
            adj_pair_via.setdefault((u, v), set()).add(via)
            layer_nodes.setdefault(rank, {})[u] = a
            layer_nodes.setdefault(rank, {})[v] = b

    if not raw_inc_edges and not adj_pair_via:
        return None, {}, []

    degree = {}
    for u, v, _d in raw_inc_edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    for (u, v) in adj_pair_via:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1

    if min_degree > 0:
        degree = {n: d for n, d in degree.items() if d >= min_degree}
    if not degree:
        return None, {}, []

    layers_sorted = sorted(layer_nodes.keys())
    selected = set()
    # When the user has explicitly sampled which ids to show per rank,
    # those nodes must always make it into the graph - even if they
    # don't currently participate in any selected incidence or
    # adjacency. Otherwise picking only adjacencies (or any narrow
    # neighbourhood subset) silently hides every isolated vertex.
    selection_is_explicit = selected_by_rank is not None
    if selection_is_explicit:
        layers_from_selection = {
            int(r) for r, ids in selected_by_rank.items() if ids
        }
        layers_sorted = sorted(set(layers_sorted) | layers_from_selection)
        for rank in layers_sorted:
            chosen_ids = set(selected_by_rank.get(int(rank), set()))
            for node_id in chosen_ids:
                lid = _layered_node_id(rank, node_id)
                selected.add(lid)
                # Ensure the (rank, original_id) mapping exists even for
                # picks that never appeared as an edge endpoint.
                layer_nodes.setdefault(int(rank), {}).setdefault(lid, int(node_id))
    else:
        num_layers = max(len(layers_sorted), 1)
        per_layer_cap = max(1, math.ceil(max_nodes / num_layers))
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

    # Collect every direction that contributed to each unique incidence
    # pair so we can render the arrow in the correct direction later.
    inc_pair_dirs = {}
    for u, v, direction in raw_inc_edges:
        if u not in selected or v not in selected:
            continue
        pair = (u, v) if u < v else (v, u)
        inc_pair_dirs.setdefault(pair, set()).add(direction)

    for (u, v), dirs in inc_pair_dirs.items():
        if len(dirs) == 1:
            edge_direction = next(iter(dirs))
        else:
            edge_direction = "both"
        G.add_edge(u, v, kind="incidence", direction=edge_direction)

    for (u, v), via_set in adj_pair_via.items():
        if u not in selected or v not in selected:
            continue
        G.add_edge(
            u, v,
            kind="adjacency",
            via_ranks=sorted(int(x) for x in via_set),
        )

    if not selection_is_explicit:
        # Auto-pick path keeps the old "trim isolated" safety net so the
        # view isn't cluttered with leftovers. Explicit selection always
        # keeps every picked node, isolated or not.
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

    links_out = []
    for u, v, data in G.edges(data=True):
        kind = data.get("kind")
        u_layer = int(G.nodes[u].get("layer", 0))
        v_layer = int(G.nodes[v].get("layer", 0))
        if kind == "adjacency":
            # If the same node-pair adjacency arises from multiple via-ranks
            # (e.g. selected ``1-up_adjacency-0`` *and* ``2-up_adjacency-0``)
            # we no longer blend their colours into one; instead we emit
            # one link per via-rank, each keeping its own rank colour. The
            # JS renderer offsets duplicate-endpoint links perpendicular
            # to the line direction so they draw as visible parallel edges
            # rather than overlapping.
            via_ranks = data.get("via_ranks") or [u_layer]
            for via in via_ranks:
                v_int = int(via)
                links_out.append({
                    "source": str(u),
                    "target": str(v),
                    "color": rank_color(v_int),
                    "kind": "adjacency",
                    "srcRank": int(u_layer),
                    "viaRanks": [v_int],
                })
        else:
            # Orient the link so the arrow tip points the way the user
            # expects: up-incidences flow lower -> higher rank (arrow toward
            # highest rank), down-incidences flow higher -> lower (arrow
            # toward lowest rank). When an edge was contributed to by both
            # up and down specs we default to "up". For edges that don't
            # carry a direction tag (legacy builders) we also default to
            # "up", which matches the canonical ``incidence_k`` orientation.
            direction = data.get("direction", "up")
            if u_layer == v_layer:
                src_node, tgt_node = u, v
                src_layer, tgt_layer = u_layer, v_layer
            else:
                if u_layer < v_layer:
                    lower_node, lower_layer = u, u_layer
                    higher_node, higher_layer = v, v_layer
                else:
                    lower_node, lower_layer = v, v_layer
                    higher_node, higher_layer = u, u_layer
                if direction == "down":
                    src_node, src_layer = higher_node, higher_layer
                    tgt_node, tgt_layer = lower_node, lower_layer
                else:
                    src_node, src_layer = lower_node, lower_layer
                    tgt_node, tgt_layer = higher_node, higher_layer
            # Reversed gradient: near each endpoint we paint with the *other*
            # endpoint's rank color (per user spec).
            color_at_src = rank_color(tgt_layer)
            color_at_tgt = rank_color(src_layer)
            midpoint = blend_hex(color_at_src, color_at_tgt)
            links_out.append({
                "source": str(src_node),
                "target": str(tgt_node),
                "color": midpoint,
                "colorStart": color_at_src,
                "colorEnd": color_at_tgt,
                "kind": kind or "incidence",
                "srcRank": src_layer,
                "tgtRank": tgt_layer,
                "direction": direction,
            })

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
        "legend": _build_legend(layers_sorted, rank_labels),
        "relationsLegend": _build_relations_legend(links_out, rank_labels),
    }


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


def _resolve_oc_select(value, dataset_params):
    """Resolve a ${oc.select:key,default} interpolation against dataset parameters.

    Returns the resolved value, or None if unresolvable and no default exists.
    """
    m = re.match(r"^\$\{oc\.select:([^,}]+)(?:,([^}]*))?\}$", value.strip())
    if not m:
        return None
    key_path = m.group(1).strip()
    default_str = m.group(2)

    parts = key_path.split(".")
    if parts[:2] == ["dataset", "parameters"] and len(parts) == 3:
        param_name = parts[2]
        if param_name in dataset_params:
            return dataset_params[param_name]

    if default_str is not None:
        default_str = default_str.strip()
        if default_str.lower() == "null" or default_str.lower() == "none":
            return None
        if default_str.lower() == "true":
            return True
        if default_str.lower() == "false":
            return False
        try:
            return int(default_str)
        except ValueError:
            pass
        try:
            return float(default_str)
        except ValueError:
            pass
        return default_str
    return None


def _get_dataset_parameters_for_lifting(lifting_info, domain=None, dataset_name=None):
    """Load dataset parameters to use when resolving lifting config interpolations."""
    source_domain = domain or lifting_info.get("source", "graph")
    dset_name = dataset_name or st.session_state.get("dataset_name")
    if not dset_name:
        return {}
    try:
        cfg = load_dataset_config(source_domain, dset_name)
        return cfg.get("parameters") or {}
    except (FileNotFoundError, Exception):
        return {}


def apply_lifting(data, lifting_info, *, graph_index=0, domain=None, dataset_name=None):
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

    SKIP_KEYS = {'transform_type', 'transform_name'}

    dataset_params = _get_dataset_parameters_for_lifting(
        lifting_info, domain=domain, dataset_name=dataset_name
    )

    params = {}
    for k, v in config.items():
        if k in SKIP_KEYS:
            continue
        if isinstance(v, str) and v.startswith('${'):
            resolved = _resolve_oc_select(v, dataset_params)
            if resolved is not None:
                params[k] = resolved
            continue
        params[k] = v

    transform = DataTransform(transform_name=transform_name, **params)

    if hasattr(data, '__getitem__'):
        single = data[graph_index]
    else:
        single = data

    single = _ensure_float_node_features(single)

    transformed = transform(single)
    return transformed


@st.cache_resource(show_spinner=False)
def _load_dataset_cached(domain, dataset_name):
    """Cache raw dataset objects by domain/name."""
    return load_dataset(domain=domain, dataset_name=dataset_name)


@st.cache_resource(show_spinner=False)
def _apply_lifting_cached(
    domain,
    dataset_name,
    lifting_name,
    lifting_source,
    lifting_target,
    lifting_config_json,
    graph_index=0,
):
    """Cache lifted data by dataset + lifting identity + config + graph index."""
    raw, _ = _load_dataset_cached(domain=domain, dataset_name=dataset_name)
    raw_copy = copy.deepcopy(raw)
    payload = {
        "name": lifting_name,
        "source": lifting_source,
        "target": lifting_target,
        "config": json.loads(lifting_config_json),
    }
    return apply_lifting(
        raw_copy, payload, graph_index=graph_index,
        domain=domain, dataset_name=dataset_name,
    )


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
        """Return True if ``v`` is a positive integer (rejecting bools)."""
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
        width="stretch",
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

# Graph block height for ``components.html`` (header + chart inside the iframe).
D3_EMBED_HEIGHT = 820


def _format_graph_header_title(vdesc, dataset_name, lift_line, num_nodes) -> str:
    """Single-line title for the D3 embed header (View, Dataset, Lift, Nodes)."""
    nodes_str = str(num_nodes) if num_nodes is not None else "N/A"
    return (
        f"View: {vdesc}  Dataset: {dataset_name}  |  {lift_line}  |  Nodes: {nodes_str}"
    )


def _node_count_from_dataset(dset0):
    """Return node count for header display, or None if unavailable."""
    if dset0 is None or not hasattr(dset0, "num_nodes"):
        return None
    num_nodes = dset0.num_nodes
    if num_nodes is None and getattr(dset0, "edge_index", None) is not None:
        num_nodes = int(dset0.edge_index.max().item()) + 1
    return num_nodes


def _render_dataset_metadata_card(domain, dataset_name):
    """Read-only dataset metadata from configs/dataset YAML."""
    meta = extract_dataset_metadata(domain, dataset_name)
    st.markdown("**Dataset metadata**")
    col_left, col_right = st.columns(2)
    with col_left:
        st.caption(f"**task:** {meta.get('task') or 'N/A'}")
        st.caption(f"**task_level:** {meta.get('task_level') or 'N/A'}")
        st.caption(
            f"**num_features:** {_format_num_features(meta.get('num_features'))}"
        )
    with col_right:
        st.caption(
            f"**learning_setting:** {meta.get('learning_setting') or 'N/A'}"
        )
        st.caption(f"**split_type:** {meta.get('split_type') or 'N/A'}")
        num_classes = meta.get("num_classes")
        st.caption(
            f"**num_classes:** {num_classes if num_classes is not None else 'N/A'}"
        )


def _render_left_config(available_datasets):
    """Render configuration controls in the Streamlit sidebar; return selections."""
    st.header("Data configuration")

    with st.expander("Dataset", expanded=True):
        # Persist domain/dataset across sidebar tab switches via shadow keys
        # (these selectboxes can't use a widget key because the Dataset options
        # change with the Domain, which would invalidate a stored key value).
        domain_options = list(available_datasets.keys())
        prev_domain = st.session_state.get("cfg_domain")
        domain_index = (
            domain_options.index(prev_domain)
            if prev_domain in domain_options
            else 0
        )
        selected_domain = st.selectbox(
            "Domain",
            options=domain_options,
            index=domain_index,
            help="Topological domain (folder under configs/dataset).",
        )
        st.session_state["cfg_domain"] = selected_domain

        datasets_in_domain = available_datasets.get(selected_domain, [])
        prev_dataset = st.session_state.get("cfg_dataset")
        dataset_index = (
            datasets_in_domain.index(prev_dataset)
            if prev_dataset in datasets_in_domain
            else 0
        )
        selected_dataset = st.selectbox(
            "Dataset",
            options=datasets_in_domain,
            index=dataset_index,
            help=f"YAML stem under configs/dataset/{selected_domain}/",
        )
        st.session_state["cfg_dataset"] = selected_dataset
        st.caption(f"**{selected_domain}** / **{selected_dataset}**")
        _render_dataset_metadata_card(selected_domain, selected_dataset)

        prev_key = st.session_state.get("_last_dataset_key")
        dataset_key = (selected_domain, selected_dataset)
        if prev_key != dataset_key:
            st.session_state["active_graph_index"] = 0
            st.session_state["loaded_dataset_size"] = 0
            st.session_state["_last_dataset_key"] = dataset_key
            if "main_graph_index_input" in st.session_state:
                del st.session_state["main_graph_index_input"]
            # When switching to a dataset whose domain has native higher-order
            # connectivity (e.g. MANTRA simplicial complexes already come with
            # full incidence/adjacency/laplacian matrices), default the
            # "Use lifting" toggle to off — applying a transform would only
            # discard or overwrite that structure. Users can still opt in.
            if selected_domain in _NATIVE_HIGHER_ORDER_DOMAINS:
                st.session_state["use_lifting"] = False
            else:
                st.session_state["use_lifting"] = True

    # ``value=`` is intentionally omitted: ``st.session_state["use_lifting"]``
    # is seeded in ``main()`` before this widget renders, and Streamlit warns
    # if a keyed widget receives both a default ``value`` and a pre-existing
    # session-state value.
    use_lifting = st.toggle(
        "Use lifting",
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

    st.subheader("Actions")
    load_clicked = st.button(
        "Load graph",
        type="primary",
        width="stretch",
        key="load_graph_btn",
    )

    return {
        "selected_domain": selected_domain,
        "selected_dataset": selected_dataset,
        "use_lifting": use_lifting,
        "selected_lifting": selected_lifting,
        "edited_lifting_config": edited_lifting_config,
        "edited_lifting_errors": edited_lifting_errors,
        "load_clicked": load_clicked,
    }


def _split_neighborhoods(available):
    """Bucket available neighborhoods into picker boxes."""
    graph_ids = [n["id"] for n in available if n["kind"] in ("graph", "hyperedges")]
    incidence = [n for n in available if n["kind"] == "incidence"]
    adjacency = [n for n in available if n["kind"] == "adjacency"]
    tb_incidence = [n for n in available if n["kind"] == "tb_incidence"]
    tb_adjacency = [n for n in available if n["kind"] == "tb_adjacency"]
    incidence.sort(key=lambda n: n.get("rank", 0))
    adjacency.sort(key=lambda n: n.get("rank", 0))
    tb_incidence.sort(key=lambda n: (n.get("rank", 0), n["id"]))
    tb_adjacency.sort(key=lambda n: (n.get("rank", 0), n["id"]))
    return graph_ids, incidence, adjacency, tb_incidence, tb_adjacency


_ADD_PLACEHOLDER = "➕ Add neighborhood…"


def _render_neighborhood_picker():
    """Picker showing all non-empty TopoBench neighborhoods on the loaded data."""
    st.subheader("Available neighborhoods")
    available = st.session_state.get("available_neighborhoods") or []
    if not available:
        st.info(
            "Apply a lifting in the **Load** tab to compute "
            "`r-direction_type-src_rank` neighborhoods."
        )
        return

    (_graph_ids, _incidence_items, _adjacency_items,
     tb_incidence_items, tb_adjacency_items) = _split_neighborhoods(available)
    selected_ids = list(st.session_state.get("selected_neighborhood_ids") or [])

    base_items = [n for n in available if n["kind"] in ("graph", "hyperedges")]
    _render_base_group(base_items, selected_ids)
    _render_nbhd_group("Incidence", "inc", tb_incidence_items, selected_ids)
    _render_nbhd_group("Adjacency", "adj", tb_adjacency_items, selected_ids)


def _render_base_group(items, selected_ids):
    """Render the base-structure view options (graph / hyperedges).

    Picking one shows only that structure (exclusive selection), which is how a
    user views the plain (hyper)graph without any neighborhood overlay.
    """
    if not items:
        return
    selected_set = set(selected_ids)
    with st.container(border=True):
        st.markdown("**Base structure**")
        for item in items:
            active = selected_set == {item["id"]}
            st.button(
                item["label"],
                key=f"base_{item['id']}_btn",
                help="Show only this structure",
                on_click=_on_base_pick,
                args=(item["id"],),
                type="primary" if active else "secondary",
                width="stretch",
            )


def _render_nbhd_group(title, prefix, items, selected_ids):
    """Render one neighborhood group: selected tags + an add-only dropdown.

    ``items`` are the available neighborhoods of this group (each a dict with
    ``id``/``label``). Selected items appear as removable badges above a
    dropdown that lists only the not-yet-selected options. Empty groups show a
    short note instead of a dropdown.
    """
    selected_set = set(selected_ids)
    with st.container(border=True):
        st.markdown(f"**{title}** (up / down)")

        if not items:
            st.caption(f"No {title.lower()} neighborhoods on this data")
            return

        labels = {it["id"]: it["label"] for it in items}
        chosen = [it["id"] for it in items if it["id"] in selected_set]

        # Selected options as removable tag badges (above the dropdown).
        if chosen:
            try:
                tag_row = st.container(horizontal=True)
            except TypeError:  # older Streamlit without horizontal containers
                tag_row = st.container()
            with tag_row:
                for nb_id in chosen:
                    st.button(
                        f"{nb_id}  ✕",
                        key=f"rm_{nb_id}",
                        help=labels.get(nb_id, nb_id),
                        on_click=_on_nbhd_remove,
                        args=(nb_id,),
                        width="content",
                    )

        # Add-only dropdown listing options not yet selected.
        remaining = [it["id"] for it in items if it["id"] not in selected_set]
        add_key = f"{prefix}_add_select"
        if remaining:
            if st.session_state.get(add_key) not in remaining:
                st.session_state[add_key] = _ADD_PLACEHOLDER
            st.selectbox(
                f"Add {title.lower()} neighborhood",
                options=[_ADD_PLACEHOLDER] + remaining,
                format_func=lambda v: _ADD_PLACEHOLDER if v == _ADD_PLACEHOLDER else labels.get(v, v),
                key=add_key,
                on_change=_on_nbhd_add,
                args=(prefix,),
                label_visibility="collapsed",
            )
        else:
            st.caption("All options selected")


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

    for n in available:
        if n["kind"] == "incidence":
            st.session_state[f"inc_{n['id']}_check"] = n["id"] in selected_set
        elif n["kind"] == "adjacency":
            st.session_state[f"adj_{n['id']}_check"] = n["id"] in selected_set
        elif n["kind"] in ("tb_incidence", "tb_adjacency"):
            st.session_state[f"tb_{n['id']}_check"] = n["id"] in selected_set


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


def _collect_layered_selection():
    """Read every layered-kind checkbox and return the union of checked ids.

    Combines basic incidence_k / adjacency_k checkboxes with the multi-hop
    TopoBench checkboxes (``tb_incidence`` / ``tb_adjacency``).
    """
    available = st.session_state.get("available_neighborhoods") or []
    layered_kinds = ("incidence", "adjacency", "tb_incidence", "tb_adjacency")
    picked = []
    for n in available:
        kind = n["kind"]
        if kind not in layered_kinds:
            continue
        if kind == "incidence":
            key = f"inc_{n['id']}_check"
        elif kind == "adjacency":
            key = f"adj_{n['id']}_check"
        else:
            key = f"tb_{n['id']}_check"
        if st.session_state.get(key):
            picked.append(n["id"])
    return picked


def _on_incidence_toggle():
    """Toggling any layered checkbox keeps the union of all checked items."""
    if st.session_state.get("data") is None:
        return
    new_ids = _collect_layered_selection()
    if not new_ids:
        graph_pick = st.session_state.get("graph_radio")
        if graph_pick and graph_pick != "(none)":
            _commit_selection([graph_pick])
            return
        prev = list(st.session_state.get("selected_neighborhood_ids") or [])
        if not prev:
            return
        new_ids = [prev[0]]
    _commit_selection(new_ids)


def _on_layered_3d_toggle():
    """Re-embed the graph when toggling 2D vs 3D for the current selection."""
    if st.session_state.get("data") is None:
        return
    ids = list(st.session_state.get("selected_neighborhood_ids") or [])
    if not ids:
        return
    _rebuild_embed_for_neighborhoods(ids)


def _on_color_by_rank_toggle():
    """Re-embed when toggling rank-gradient vs unique-solid neighborhood colors."""
    if st.session_state.get("data") is None:
        return
    ids = list(st.session_state.get("selected_neighborhood_ids") or [])
    if not ids:
        return
    _rebuild_embed_for_neighborhoods(ids)


def _neighborhood_color_mode_from_session():
    """Return ``rank_gradient`` or ``unique_solid`` from the Explore-tab toggle."""
    if st.session_state.get("color_neighborhoods_by_rank", False):
        return "rank_gradient"
    return "unique_solid"

def _on_metrics_option_change():
    """Re-embed when a metrics toggle changes (HUD visibility / advanced compute)."""
    if st.session_state.get("data") is None:
        return
    ids = list(st.session_state.get("selected_neighborhood_ids") or [])
    if not ids:
        return
    _rebuild_embed_for_neighborhoods(ids)


def _on_adjacency_toggle():
    """Toggling an adjacency checkbox keeps the union of all layered checks."""
    _on_incidence_toggle()


def _on_tb_toggle():
    """Toggling a TopoBench checkbox keeps the union of all layered checks."""
    _on_incidence_toggle()


_BASE_IDS = ("graph", "hyperedges")


def _on_base_pick(base_id):
    """Select a base structure exclusively (view only the plain (hyper)graph)."""
    if st.session_state.get("data") is None:
        return
    _commit_selection([base_id])


def _on_nbhd_add(prefix):
    """Add the neighborhood chosen in a group's add-dropdown to the selection."""
    if st.session_state.get("data") is None:
        return
    add_key = f"{prefix}_add_select"
    pick = st.session_state.get(add_key)
    st.session_state[add_key] = _ADD_PLACEHOLDER
    if not pick or pick == _ADD_PLACEHOLDER:
        return
    current = list(st.session_state.get("selected_neighborhood_ids") or [])
    # Adding a neighborhood leaves any exclusive base-structure view.
    current = [c for c in current if c not in _BASE_IDS]
    if pick in current:
        return
    _commit_selection(current + [pick])


def _on_nbhd_remove(neigh_id):
    """Remove a neighborhood tag from the selection (keeping at least one)."""
    if st.session_state.get("data") is None:
        return
    current = list(st.session_state.get("selected_neighborhood_ids") or [])
    new_ids = [i for i in current if i != neigh_id]
    if not new_ids:
        # Preserve the non-empty guarantee the embed relies on.
        return
    _commit_selection(new_ids)


def _commit_selection(new_ids):
    """Persist the selection, sync all picker widgets, rebuild the embed."""
    new_ids = list(new_ids)
    st.session_state["selected_neighborhood_ids"] = new_ids
    _sync_picker_widget_state(new_ids)
    _rebuild_embed_for_neighborhoods(new_ids)


def _metrics_marker(neigh_ids, expensive):
    """Cache key for displayed-graph metrics.

    Includes everything that changes the displayed graph (selection, sampling
    snapshot, graph index, lifting) plus the expensive flag. Deliberately
    excludes the 2D/3D toggle and the HUD on/off toggle so those never trigger
    a recompute.
    """
    return "|".join(
        [
            "+".join(neigh_ids),
            f"max={st.session_state.get('_loaded_max_nodes')}",
            f"min={st.session_state.get('_loaded_min_degree')}",
            f"idx={st.session_state.get('active_graph_index')}",
            f"lift={st.session_state.get('lifting_applied')}",
            f"exp={bool(expensive)}",
        ]
    )


def _compute_and_attach_metrics(payload, G_load, neigh_ids):
    """Compute displayed-graph metrics (cached) and attach them to the payload."""
    if not isinstance(payload, dict):
        return

    expensive = bool(st.session_state.get("metrics_expensive", False))
    marker = _metrics_marker(neigh_ids, expensive)
    cached = st.session_state.get("_graph_metrics")
    if cached and cached.get("marker") == marker:
        metrics = cached.get("data")
    else:
        try:
            metrics = gm.compute_graph_metrics(
                G_load,
                view_label=payload.get("graphType", "graph"),
                expensive=expensive,
            )
            st.session_state.pop("_graph_metrics_error", None)
        except Exception as e:  # never break the embed over metrics
            metrics = None
            st.session_state["_graph_metrics_error"] = str(e)
        st.session_state["_graph_metrics"] = {"marker": marker, "data": metrics}

    if not metrics:
        return

    payload["graphMetrics"] = gm.build_hud_rows(metrics)
    payload["showMetricsHud"] = bool(
        st.session_state.get("show_metrics_hud", True)
    )
    for nd in payload.get("nodes", []):
        nm = gm.node_payload_metrics(metrics, nd.get("id"))
        if nm:
            nd["metrics"] = nm
    for ln in payload.get("links", []):
        em = gm.edge_payload_metrics(metrics, ln.get("source"), ln.get("target"))
        if em:
            ln["metrics"] = em


def _rebuild_embed_for_neighborhoods(neigh_ids):
    """Rebuild the embedded D3 view for a list of neighborhood ids.

    A single non-incidence id falls through to the original single-matrix
    pipeline. Multiple incidence ids or multiple adjacency ids go through
    layered pipelines (stacked ranks bottom-to-top).
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
    shared_sampling = st.session_state.get("_shared_sampling") or {}
    shared_by_rank = shared_sampling.get("selected_by_rank") or {}
    shared_hyperedges = shared_sampling.get("selected_hyperedges")
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

    def _classify_id(nid):
        """Return classification tuple for an id.

        - ``('incidence', src_sorted, tgt_sorted)`` for bipartite cross-rank edges.
        - ``('adjacency', src_rank, via_rank)`` for within-rank edges, where
          ``via_rank`` is the rank that mediates the adjacency
          (basic ``adjacency_k`` → k+1; TB ``r-up_adjacency-s`` → s+r;
          TB ``r-down_adjacency-s`` → s-r).
        - ``(None, None, None)`` if unknown.
        """
        if nid.startswith("incidence_") and nid != "incidence_hyperedges":
            try:
                k = int(nid.split("_", 1)[1])
            except ValueError:
                return (None, None, None)
            if k <= 0:
                return (None, None, None)
            return ("incidence", max(k - 1, 0), k)
        if nid.startswith("adjacency_"):
            try:
                k = int(nid.split("_", 1)[1])
            except ValueError:
                return (None, None, None)
            # Basic adjacency_k matches TopoBench up_adjacency-k, mediated by (k+1)-cells.
            return ("adjacency", k, k + 1)
        if "-" in nid:
            try:
                _r, _dir, _nt, _src = parse_neighborhood(nid)
            except Exception:
                return (None, None, None)
            if _nt == "adjacency":
                _via = _src + _r if _dir == "up" else _src - _r
                return ("adjacency", _src, _via)
            if _nt == "incidence":
                _tgt = _neighborhood_target_rank(nid)
                src_sorted, tgt_sorted = sorted((_src, _tgt))
                return ("incidence", src_sorted, tgt_sorted)
        return (None, None, None)

    def _resolve_matrix(nid):
        """Return a sparse COO matrix for an id (basic or TopoBench), or None."""
        if nid.startswith("incidence_") and nid != "incidence_hyperedges":
            try:
                k = int(nid.split("_", 1)[1])
            except ValueError:
                return None
            return incidence_rank_k_to_sparse(dset0, k)
        if nid.startswith("adjacency_"):
            try:
                k = int(nid.split("_", 1)[1])
            except ValueError:
                return None
            return adjacency_rank_k_to_sparse(dset0, k)
        tb_cache = st.session_state.get("_topobench_neighborhoods") or {}
        if nid in tb_cache:
            sp = tb_cache[nid]
            try:
                _r, _dir, _nt, _src = parse_neighborhood(nid)
            except Exception:
                return sp
            if _nt == "incidence" and _dir == "up":
                # TopoBench's up_incidence-s yields a (N_tgt, N_src) matrix
                # (transposed by select_neighborhoods_of_interest). For
                # layered visualization we need (N_lower, N_higher), so
                # transpose back to canonical (src × tgt) orientation.
                try:
                    return sp.transpose(0, 1).coalesce()
                except Exception:
                    return sp
            return sp
        return None

    classified = [(nid, _classify_id(nid)) for nid in neigh_ids]
    incidence_ids = [nid for nid, (k, _s, _t) in classified if k == "incidence"]
    adjacency_ids = [nid for nid, (k, _s, _t) in classified if k == "adjacency"]
    non_layered_ids = [
        nid for nid, (k, _s, _t) in classified if k is None
    ]
    use_combined = (
        len(incidence_ids) >= 1
        and len(adjacency_ids) >= 1
        and not non_layered_ids
    )
    use_layered = (
        not use_combined
        and
        len(incidence_ids) >= 1
        and len(incidence_ids) == len(neigh_ids)
    )
    use_layered_adj = (
        not use_combined
        and
        len(adjacency_ids) >= 2
        and len(adjacency_ids) == len(neigh_ids)
    )
    is_3d_eligible = (
        use_combined
        or use_layered_adj
        or (use_layered and len(incidence_ids) > 1)
    )

    try:
        if use_combined:
            incidence_specs = []
            for nid in incidence_ids:
                kind, src, tgt = _classify_id(nid)
                sp = _resolve_matrix(nid)
                if sp is None:
                    st.error(f"Neighborhood '{nid}' is not available.")
                    return False
                incidence_specs.append(
                    (sp, src, tgt, _incidence_direction_for_id(nid))
                )

            adjacency_specs = []
            for nid in adjacency_ids:
                kind, src, via = _classify_id(nid)
                sp = _resolve_matrix(nid)
                if sp is None:
                    st.error(f"Neighborhood '{nid}' is not available.")
                    return False
                adjacency_specs.append((sp, src, via))

            if not incidence_specs or not adjacency_specs:
                st.error(
                    "Combined rendering requires at least one valid incidence_k "
                    "and one valid adjacency_k."
                )
                return False

            ranks_chosen = sorted(
                {spec[1] for spec in incidence_specs}
                | {spec[2] for spec in incidence_specs}
                | {rank for _, rank, _ in adjacency_specs}
            )
            G_load, nd_load, layers = build_combined_layered_networkx(
                incidence_specs,
                adjacency_specs,
                max_nodes=max_nodes,
                min_degree=min_degree,
                selected_by_rank=(
                    {int(r): set(shared_by_rank.get(int(r), set())) for r in ranks_chosen}
                    if shared_by_rank
                    else None
                ),
            )
            if not G_load or len(G_load.nodes()) == 0:
                st.error(
                    "Error building graph payload: "
                    "no nodes survive current filters for the selected combined layers."
                )
                return False

            sorted_inc = sorted(
                incidence_ids,
                key=lambda x: _classify_id(x)[1] if _classify_id(x)[1] is not None else 0,
            )
            sorted_adj = sorted(
                adjacency_ids,
                key=lambda x: _classify_id(x)[1] if _classify_id(x)[1] is not None else 0,
            )
            vdesc = (
                "Layered: "
                + ", ".join(sorted_inc)
                + " + "
                + ", ".join(sorted_adj)
            )
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
        elif use_layered:
            specs = []
            for nid in incidence_ids:
                kind, src, tgt = _classify_id(nid)
                sp = _resolve_matrix(nid)
                if sp is None:
                    st.error(f"Neighborhood '{nid}' is not available.")
                    return False
                specs.append((sp, src, tgt, _incidence_direction_for_id(nid)))

            if not specs:
                st.error("No valid incidence neighborhoods selected for layered rendering.")
                return False

            ranks_chosen = sorted({sr for _, sr, _, _ in specs}
                                  | {tr for _, _, tr, _ in specs})
            expected = list(range(min(ranks_chosen), max(ranks_chosen) + 1))
            if ranks_chosen != expected:
                missing = sorted(set(expected) - set(ranks_chosen))
                st.info(
                    "Selected incidences are not contiguous in rank; "
                    f"missing rank(s): {missing}. Layers will still render "
                    "but unrelated ranks may not connect."
                )

            G_load, nd_load, layers = build_layered_networkx(
                specs,
                max_nodes=max_nodes,
                min_degree=min_degree,
                selected_by_rank=(
                    {int(r): set(shared_by_rank.get(int(r), set())) for r in ranks_chosen}
                    if shared_by_rank
                    else None
                ),
            )
            if not G_load or len(G_load.nodes()) == 0:
                st.error(
                    "Error building graph payload: "
                    "no nodes survive current filters for the selected layers."
                )
                return False

            sorted_inc = sorted(
                incidence_ids,
                key=lambda x: _classify_id(x)[1] if _classify_id(x)[1] is not None else 0,
            )
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
        elif use_layered_adj:
            specs = []
            for nid in adjacency_ids:
                kind, src, via = _classify_id(nid)
                sp = _resolve_matrix(nid)
                if sp is None:
                    st.error(f"Neighborhood '{nid}' is not available.")
                    return False
                specs.append((sp, src, via))

            if not specs:
                st.error("No valid adjacency neighborhoods selected for layered rendering.")
                return False

            ranks_chosen = sorted({rank for _, rank, _ in specs})
            G_load, nd_load, layers = build_layered_adjacency_networkx(
                specs,
                max_nodes=max_nodes,
                min_degree=min_degree,
                selected_by_rank=(
                    {int(r): set(shared_by_rank.get(int(r), set())) for r in ranks_chosen}
                    if shared_by_rank
                    else None
                ),
            )
            if not G_load or len(G_load.nodes()) == 0:
                st.error(
                    "Error building graph payload: "
                    "no nodes survive current filters for the selected adjacency layers."
                )
                return False

            sorted_adj = sorted(
                adjacency_ids,
                key=lambda x: _classify_id(x)[1] if _classify_id(x)[1] is not None else 0,
            )
            vdesc = "Layered adjacency: " + " | ".join(sorted_adj)
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

            allowed_source = None
            allowed_target = None
            if shared_by_rank:
                if single_id == "graph":
                    allowed_source = set(shared_by_rank.get(0, set()))
                    allowed_target = set(shared_by_rank.get(0, set()))
                elif single_id.startswith("adjacency_"):
                    k = int(single_id.split("_", 1)[1])
                    allowed_source = set(shared_by_rank.get(k, set()))
                    allowed_target = set(shared_by_rank.get(k, set()))
                elif single_id.startswith("incidence_"):
                    k = int(single_id.split("_", 1)[1])
                    allowed_source = set(shared_by_rank.get(max(k - 1, 0), set()))
                    allowed_target = set(shared_by_rank.get(k, set()))
                elif single_id == "hyperedges":
                    allowed_source = set(shared_by_rank.get(0, set()))
                    if shared_hyperedges is not None:
                        allowed_target = set(shared_hyperedges)
                elif "-" in single_id:
                    try:
                        _r, _dir, _nt, _src = parse_neighborhood(single_id)
                        allowed_source = set(shared_by_rank.get(_src, set()))
                        if _nt == "adjacency":
                            allowed_target = set(shared_by_rank.get(_src, set()))
                        else:
                            _tgt = _neighborhood_target_rank(single_id)
                            allowed_target = set(shared_by_rank.get(_tgt, set()))
                    except Exception:
                        pass

            G_load, nd_load = sparse_to_networkx(
                matrix,
                max_nodes=max_nodes,
                min_degree=min_degree,
                allowed_source=allowed_source,
                allowed_target=allowed_target,
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
        if (
            st.session_state.get("layered_3d_view")
            and is_3d_eligible
            and isinstance(payload, dict)
        ):
            payload["graphType"] = "layered3d"
            payload["dagLevelDistance"] = 140
    except Exception as e:
        st.error(f"Error building graph payload: {e}")
        return False

    cache_marker = "+".join(neigh_ids)
    if st.session_state.get("layered_3d_view"):
        cache_marker += ":3d"
    color_mode = _neighborhood_color_mode_from_session()
    cache_marker += f":colors:{color_mode}"

    dataset_name = st.session_state.get("dataset_name") or "—"
    num_nodes = _node_count_from_dataset(dset0)
    payload["title"] = _format_graph_header_title(
        vdesc, dataset_name, lift_subtitle, num_nodes
    )
    payload["subtitle"] = ""

    _apply_neighborhood_color_mode(payload, neigh_ids, color_mode)
    _compute_and_attach_metrics(payload, G_load, neigh_ids)

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
    if st.session_state.get("layered_3d_view") and not is_3d_eligible:
        st.caption(
            "3D layered view applies to multi-rank selections only "
            "(multi-incidence, 2+ adjacency, or combined incidence+adjacency). "
            "Current view is 2D."
        )
    return True


def _finalize_loaded_sample(dset0, cfg, loaded_domain, dataset_name,
                            sync_widgets=True):
    """Populate neighborhoods, sampling, and embed for a working sample.

    ``sync_widgets`` controls whether the sampling widget keys (``ui_min_degree``,
    ``ui_rank_cap_*``, ``ui_hyperedge_cap``) are written. This is safe only on
    the initial load or from on_change callbacks (before widgets are
    re-instantiated). It must be ``False`` when called inline after those
    widgets already exist in the current run (e.g. the "Apply to all" button),
    otherwise Streamlit raises "cannot be modified after the widget is
    instantiated".
    """
    rank_labels_for_payload = get_rank_labels(
        loaded_domain, dataset_name, dset0
    )
    st.session_state["rank_labels"] = rank_labels_for_payload

    st.session_state["_topobench_neighborhoods"] = (
        compute_all_topobench_neighborhoods(dset0)
    )

    available = enumerate_neighborhoods(dset0)
    st.session_state["available_neighborhoods"] = available
    if not available:
        st.error(
            "No non-empty neighborhoods are available on this data. "
            "Apply a lifting to compute higher-order neighborhoods."
        )
        return False

    available_ids = {n["id"] for n in available}
    prev_ids = list(st.session_state.get("selected_neighborhood_ids") or [])
    if prev_ids and all(pid in available_ids for pid in prev_ids):
        default_ids = prev_ids
    else:
        default_id = pick_default_neighborhood_id(available)
        default_ids = [default_id] if default_id else []
    st.session_state["selected_neighborhood_ids"] = default_ids

    rank_pops, num_hyperedges = _discover_rank_populations(dset0)
    cfg_caps = cfg.get("caps_by_rank") or {}
    caps_by_rank = {}
    for rank, pop in rank_pops.items():
        pop_int = int(pop)
        if rank in cfg_caps or str(rank) in cfg_caps:
            raw_cap = cfg_caps.get(rank, cfg_caps.get(str(rank)))
        else:
            raw_cap = _default_cap_for_population(pop_int)
        cap = max(0, min(int(raw_cap), pop_int))
        caps_by_rank[int(rank)] = cap
    if not caps_by_rank:
        fallback_cap = _default_cap_for_population(
            int(cfg.get("max_nodes", DEFAULT_LARGE_CAP))
        )
        caps_by_rank = {0: fallback_cap}

    hyperedge_cap = cfg.get("hyperedge_cap")
    if num_hyperedges is not None and hyperedge_cap is None:
        hyperedge_cap = _default_cap_for_population(int(num_hyperedges))
    if hyperedge_cap is not None and num_hyperedges is not None:
        hyperedge_cap = max(0, min(int(hyperedge_cap), int(num_hyperedges)))

    # Remember the per-rank populations of *this* sample so that a later
    # sample switch can tell whether each carried-over cap was at the
    # previous sample's maximum (and therefore should be refreshed to the
    # new sample's maximum) or was explicitly clamped below by the user.
    st.session_state["_loaded_rank_pops"] = {int(r): int(p) for r, p in rank_pops.items()}
    st.session_state["_loaded_hyperedge_pop"] = (
        int(num_hyperedges) if num_hyperedges is not None else None
    )

    min_degree = int(cfg.get("min_degree", 0))
    if sync_widgets:
        for rank, cap in caps_by_rank.items():
            st.session_state[f"ui_rank_cap_{rank}"] = int(cap)
        if (
            hyperedge_cap is not None
            and num_hyperedges is not None
            and int(num_hyperedges) > 1
        ):
            st.session_state["ui_hyperedge_cap"] = int(hyperedge_cap)
        st.session_state["ui_min_degree"] = min_degree

    st.session_state["rank_populations"] = rank_pops
    st.session_state["hyperedge_population"] = num_hyperedges
    st.session_state["rank_caps"] = {str(k): int(v) for k, v in caps_by_rank.items()}
    st.session_state["_loaded_rank_caps"] = caps_by_rank
    st.session_state["_loaded_hyperedge_cap"] = hyperedge_cap
    st.session_state["_shared_sampling"] = compute_shared_node_sampling(
        dset0,
        caps_by_rank=caps_by_rank,
        cap_hyperedges=hyperedge_cap,
    )

    st.session_state["_loaded_max_nodes"] = int(
        caps_by_rank.get(0, cfg.get("max_nodes", DEFAULT_LARGE_CAP))
    )
    st.session_state["_loaded_min_degree"] = min_degree
    _sync_picker_widget_state(default_ids)
    return _rebuild_embed_for_neighborhoods(default_ids)


def _build_sample_at_index(raw, graph_index, cfg):
    """Return (current_data, applied_lift) for one graph index."""
    applied_lift = None
    if cfg["use_lifting"] and cfg["selected_lifting"] is not None:
        lifting_payload = copy.deepcopy(cfg["selected_lifting"])
        if isinstance(cfg.get("edited_lifting_config"), dict):
            lifting_payload["config"] = copy.deepcopy(cfg["edited_lifting_config"])
        lifting_cfg_json = json.dumps(
            lifting_payload.get("config") or {},
            sort_keys=True,
            separators=(",", ":"),
        )
        lifted = _apply_lifting_cached(
            cfg["selected_domain"],
            cfg["selected_dataset"],
            lifting_payload.get("name", ""),
            lifting_payload.get("source", ""),
            lifting_payload.get("target", ""),
            lifting_cfg_json,
            graph_index=graph_index,
        )
        return [copy.deepcopy(lifted)], lifting_payload
    if hasattr(raw, "__getitem__"):
        return [copy.deepcopy(raw[graph_index])], None
    return raw, None


def _reload_graph_at_index(graph_index):
    """Reload visualization for a different inductive graph sample."""
    raw = st.session_state.get("data_original")
    if raw is None:
        st.error("Cannot switch graph sample: no dataset loaded.")
        return False

    total = int(st.session_state.get("loaded_dataset_size", 1) or 1)
    max_idx = max(0, total - 1)
    idx = int(graph_index)
    if idx < 0 or idx > max_idx:
        st.error(f"Graph index must be between 0 and {max_idx}.")
        return False

    cfg = st.session_state.get("_load_cfg_snapshot") or {}
    try:
        current_data, applied_lift = _build_sample_at_index(raw, idx, cfg)
    except Exception as e:
        st.error(f"Error loading graph sample {idx}: {e}")
        return False

    st.session_state["data"] = current_data
    st.session_state["lifting_applied"] = applied_lift
    dset0 = (
        current_data[0] if hasattr(current_data, "__getitem__") else current_data
    )

    # Build the carry-over caps for the new sample. A cap stored against
    # the *previous* sample is carried over only if the user had clamped
    # it strictly below that sample's full population. Caps that were at
    # the previous sample's max are dropped so the new sample's auto-
    # default (= its own full population, up to DEFAULT_LARGE_CAP) takes
    # effect -- this is what fixes the "MANTRA samples all look the same"
    # issue where sample 0's small population would otherwise pin every
    # subsequent sample to the same tiny cap.
    prev_pops = st.session_state.get("_loaded_rank_pops") or {}
    loaded_caps = st.session_state.get("_loaded_rank_caps") or {}
    carry_caps = {}
    for rank, cap in loaded_caps.items():
        prev_pop = prev_pops.get(int(rank))
        if prev_pop is None:
            carry_caps[int(rank)] = int(cap)
        elif int(cap) < int(prev_pop):
            carry_caps[int(rank)] = int(cap)
        # else: was at max on previous sample, drop so finalize re-derives

    prev_hyper_pop = st.session_state.get("_loaded_hyperedge_pop")
    loaded_hyper = st.session_state.get("_loaded_hyperedge_cap")
    carry_hyper = None
    if loaded_hyper is not None:
        if prev_hyper_pop is None or int(loaded_hyper) < int(prev_hyper_pop):
            carry_hyper = int(loaded_hyper)

    finalize_cfg = {
        "caps_by_rank": carry_caps,
        "hyperedge_cap": carry_hyper,
        "max_nodes": st.session_state.get("_loaded_max_nodes", DEFAULT_LARGE_CAP),
        "min_degree": st.session_state.get("_loaded_min_degree", 0),
    }
    loaded_domain = st.session_state.get("data_domain")
    dataset_name = st.session_state.get("dataset_name")
    ok = _finalize_loaded_sample(dset0, finalize_cfg, loaded_domain, dataset_name)
    if ok:
        st.session_state["active_graph_index"] = idx
    return ok


# Default per-rank node cap. Populations at or below this are shown in full;
# larger ranks are sub-sampled to this many nodes to keep the D3 layout
# responsive in the browser.
DEFAULT_LARGE_CAP = 150


def _default_cap_for_population(pop):
    """Cap a rank population to ``DEFAULT_LARGE_CAP``, leaving small ranks whole.

    Args:
        pop: Number of cells present at a given rank.

    Returns:
        int: ``pop`` if it does not exceed ``DEFAULT_LARGE_CAP``, otherwise the
        cap.
    """
    pop = int(pop)
    return pop if pop <= DEFAULT_LARGE_CAP else DEFAULT_LARGE_CAP


def _auto_caps_by_rank(rank_populations):
    """Derive a default per-rank sampling cap from measured rank populations.

    Args:
        rank_populations: Mapping of ``rank -> population`` (cell count).

    Returns:
        dict[int, int]: Mapping of ``rank -> cap`` obtained by applying
        :func:`_default_cap_for_population` to each population.
    """
    return {
        int(r): _default_cap_for_population(p) for r, p in rank_populations.items()
    }


def _default_load_sampling_cfg():
    """Return the default sampling configuration used when loading a dataset.

    Returns:
        dict: Sampling config with empty per-rank caps, no hyperedge cap, a
        global ``max_nodes`` of ``DEFAULT_LARGE_CAP`` and no minimum-degree
        filter.
    """
    return {
        "caps_by_rank": {},
        "hyperedge_cap": None,
        "max_nodes": DEFAULT_LARGE_CAP,
        "min_degree": 0,
    }


def _get_sampling_context():
    """Return dset0, rank_populations, hyperedge_population, is_loaded."""
    loaded_data = st.session_state.get("data")
    if loaded_data is not None and hasattr(loaded_data, "__getitem__"):
        dset0 = loaded_data[0]
    else:
        dset0 = loaded_data
    is_loaded = dset0 is not None
    rank_populations = {}
    hyperedge_population = None
    if dset0 is not None:
        rank_populations, hyperedge_population = _discover_rank_populations(dset0)
    return dset0, rank_populations, hyperedge_population, is_loaded


def _format_rank_cap_summary(rank_populations, hyperedge_population):
    """One-line summary of current cap settings for the popover trigger."""
    rank_labels = st.session_state.get("rank_labels") or {}
    parts = []
    for rank, pop in sorted(rank_populations.items()):
        pop_int = int(pop)
        label = rank_labels.get(rank, f"Rank {rank}")
        if pop_int <= 1:
            parts.append(f"{label}: {pop_int}")
        else:
            key = f"ui_rank_cap_{rank}"
            cap = int(
                st.session_state.get(
                    key, _default_cap_for_population(pop_int)
                )
            )
            parts.append(f"{label}: {cap}/{pop_int}")
    if hyperedge_population is not None:
        hyper_pop_int = int(hyperedge_population)
        if hyper_pop_int <= 1:
            parts.append(f"Hyperedges: {hyper_pop_int}")
        else:
            cap = int(
                st.session_state.get(
                    "ui_hyperedge_cap", _default_cap_for_population(hyper_pop_int)
                )
            )
            parts.append(f"Hyperedges: {cap}/{hyper_pop_int}")
    return " · ".join(parts) if parts else "No ranks discovered"


def _render_rank_cap_sliders(rank_populations, hyperedge_population):
    """Render per-rank and hyperedge cap sliders inside popover or inline."""
    ui_rank_caps = {}
    ui_hyperedge_cap = None
    if not rank_populations:
        return ui_rank_caps, ui_hyperedge_cap

    max_rank_pop = max(rank_populations.values()) if rank_populations else 0
    if int(max_rank_pop) >= 2:
        set_col, btn_col = st.columns([3, 2])
        with set_col:
            all_cap_default = int(
                st.session_state.get("ui_set_all_rank_caps", DEFAULT_LARGE_CAP)
            )
            st.session_state["ui_set_all_rank_caps"] = max(
                0, min(all_cap_default, int(max_rank_pop))
            )
            all_cap = st.number_input(
                "Set all ranks to",
                min_value=0,
                max_value=int(max_rank_pop),
                step=1,
                key="ui_set_all_rank_caps",
            )
        with btn_col:
            st.write("")
            if st.button(
                "Apply to all",
                width="stretch",
                key="ui_apply_all_rank_caps",
            ):
                for rank, pop in rank_populations.items():
                    if int(pop) >= 2:
                        st.session_state[f"ui_rank_cap_{rank}"] = max(
                            0, min(int(all_cap), int(pop))
                        )
                _on_sampling_control_change()

    rank_labels = st.session_state.get("rank_labels") or {}
    for rank, pop in sorted(rank_populations.items()):
        pop_int = int(pop)
        label_prefix = rank_labels.get(rank, f"Rank {rank}")
        if pop_int <= 1:
            ui_rank_caps[int(rank)] = pop_int
            st.caption(f"{label_prefix}: {pop_int} node(s) — fixed.")
            continue

        key = f"ui_rank_cap_{rank}"
        default_cap = _default_cap_for_population(pop_int)
        value_cap = int(st.session_state.get(key, default_cap))
        st.session_state[key] = max(0, min(value_cap, pop_int))
        ui_rank_caps[int(rank)] = st.slider(
            f"{label_prefix} cap",
            min_value=0,
            max_value=pop_int,
            key=key,
            on_change=_on_sampling_control_change,
        )

    if hyperedge_population is not None:
        hyper_pop_int = int(hyperedge_population)
        if hyper_pop_int <= 1:
            ui_hyperedge_cap = hyper_pop_int
            st.caption(f"Hyperedges: {hyper_pop_int} — fixed.")
        else:
            hyper_default = _default_cap_for_population(hyper_pop_int)
            hyper_value = int(
                st.session_state.get("ui_hyperedge_cap", hyper_default)
            )
            st.session_state["ui_hyperedge_cap"] = max(
                0, min(hyper_value, hyper_pop_int)
            )
            ui_hyperedge_cap = st.slider(
                "Hyperedge cap",
                min_value=0,
                max_value=hyper_pop_int,
                key="ui_hyperedge_cap",
                on_change=_on_sampling_control_change,
            )
    return ui_rank_caps, ui_hyperedge_cap


def _render_inline_rank_cap(rank_populations, hyperedge_population):
    """Single-rank (or one configurable rank) inline cap slider."""
    ui_rank_caps = {}
    ui_hyperedge_cap = None
    rank_labels = st.session_state.get("rank_labels") or {}
    for rank, pop in sorted(rank_populations.items()):
        pop_int = int(pop)
        if pop_int <= 1:
            ui_rank_caps[int(rank)] = pop_int
            continue
        label_prefix = rank_labels.get(rank, f"Rank {rank}")
        key = f"ui_rank_cap_{rank}"
        default_cap = _default_cap_for_population(pop_int)
        value_cap = int(st.session_state.get(key, default_cap))
        st.session_state[key] = max(0, min(value_cap, pop_int))
        ui_rank_caps[int(rank)] = st.slider(
            f"{label_prefix} cap",
            min_value=0,
            max_value=pop_int,
            key=key,
            on_change=_on_sampling_control_change,
        )
    if hyperedge_population is not None:
        hyper_pop_int = int(hyperedge_population)
        if hyper_pop_int <= 1:
            ui_hyperedge_cap = hyper_pop_int
        else:
            hyper_default = _default_cap_for_population(hyper_pop_int)
            hyper_value = int(
                st.session_state.get("ui_hyperedge_cap", hyper_default)
            )
            st.session_state["ui_hyperedge_cap"] = max(
                0, min(hyper_value, hyper_pop_int)
            )
            ui_hyperedge_cap = st.slider(
                "Hyperedge cap",
                min_value=0,
                max_value=hyper_pop_int,
                key="ui_hyperedge_cap",
                on_change=_on_sampling_control_change,
            )
    return ui_rank_caps, ui_hyperedge_cap


def _render_rank_cap_controls(rank_populations, hyperedge_population):
    """Render rank-cap UI (inline or expander); return (ui_rank_caps, ui_hyperedge_cap)."""
    if not rank_populations:
        return {}, None

    configurable = [
        rank for rank, pop in rank_populations.items() if int(pop) >= 2
    ]
    use_expander = len(configurable) >= 2

    if use_expander:
        with st.expander("Per-rank caps", expanded=True):
            return _render_rank_cap_sliders(rank_populations, hyperedge_population)

    return _render_inline_rank_cap(rank_populations, hyperedge_population)


def _collect_sampling_cfg(ui_rank_caps, ui_hyperedge_cap, min_degree):
    """Build sampling fields for load / apply from widget values."""
    st.session_state["rank_caps"] = {str(k): int(v) for k, v in ui_rank_caps.items()}
    max_nodes = int(
        ui_rank_caps.get(0, st.session_state.get("max_nodes", DEFAULT_LARGE_CAP))
    )
    st.session_state["max_nodes"] = max_nodes
    st.session_state["min_degree"] = int(min_degree)
    return {
        "caps_by_rank": ui_rank_caps,
        "hyperedge_cap": ui_hyperedge_cap,
        "max_nodes": max_nodes,
        "min_degree": int(min_degree),
    }


def _read_sampling_cfg_from_session():
    """Read cap/min-degree settings from widget session keys."""
    rank_pops = st.session_state.get("rank_populations") or {}
    ui_rank_caps = {}
    for rank, pop in rank_pops.items():
        pop_int = int(pop)
        if pop_int <= 1:
            ui_rank_caps[int(rank)] = pop_int
        else:
            key = f"ui_rank_cap_{rank}"
            default = _default_cap_for_population(pop_int)
            ui_rank_caps[int(rank)] = int(st.session_state.get(key, default))

    hyperedge_population = st.session_state.get("hyperedge_population")
    ui_hyperedge_cap = None
    if hyperedge_population is not None:
        hyper_pop_int = int(hyperedge_population)
        if hyper_pop_int <= 1:
            ui_hyperedge_cap = hyper_pop_int
        else:
            ui_hyperedge_cap = int(
                st.session_state.get(
                    "ui_hyperedge_cap",
                    _default_cap_for_population(hyper_pop_int),
                )
            )

    min_degree = int(st.session_state.get("ui_min_degree", 0))
    return _collect_sampling_cfg(ui_rank_caps, ui_hyperedge_cap, min_degree)


def _on_sampling_control_change():
    """Rebuild embed when a sampling slider changes (post-load live update)."""
    data = st.session_state.get("data")
    if data is None:
        return

    cfg = _read_sampling_cfg_from_session()
    loaded_caps = st.session_state.get("_loaded_rank_caps") or {}
    loaded_hyper = st.session_state.get("_loaded_hyperedge_cap")
    loaded_min = int(st.session_state.get("_loaded_min_degree", 0))
    if (
        cfg["caps_by_rank"] == loaded_caps
        and cfg.get("hyperedge_cap") == loaded_hyper
        and cfg["min_degree"] == loaded_min
    ):
        return

    dset0 = data[0] if hasattr(data, "__getitem__") else data
    loaded_domain = st.session_state.get("data_domain")
    dataset_name = st.session_state.get("dataset_name")
    ok = _finalize_loaded_sample(
        dset0, cfg, loaded_domain, dataset_name, sync_widgets=False
    )
    if ok:
        snap = st.session_state.get("_load_cfg_snapshot") or {}
        snap["caps_by_rank"] = copy.deepcopy(cfg.get("caps_by_rank") or {})
        snap["hyperedge_cap"] = cfg.get("hyperedge_cap")
        snap["max_nodes"] = cfg.get("max_nodes", DEFAULT_LARGE_CAP)
        snap["min_degree"] = cfg.get("min_degree", 0)
        st.session_state["_load_cfg_snapshot"] = snap
        # No explicit ``st.rerun()``: this runs as a widget ``on_change``
        # callback and Streamlit reruns the script automatically after the
        # callback returns. Calling ``st.rerun()`` here would emit
        # "Calling st.rerun() within a callback is a no-op."


def _on_main_graph_index_change():
    """Reload when the user submits a graph index from the main-panel input."""
    active = int(st.session_state.get("active_graph_index", 0))
    raw_val = str(st.session_state.get("main_graph_index_input", "0")).strip()
    total = int(st.session_state.get("loaded_dataset_size", 1) or 1)
    max_idx = max(0, total - 1)

    try:
        idx = int(raw_val)
    except ValueError:
        st.session_state["main_graph_index_input"] = str(active)
        st.session_state["_graph_index_error"] = "Enter an integer."
        return

    if idx < 0 or idx > max_idx:
        st.session_state["main_graph_index_input"] = str(active)
        st.session_state["_graph_index_error"] = (
            f"Enter an integer between 0 and {max_idx}."
        )
        return

    if idx == active:
        st.session_state.pop("_graph_index_error", None)
        return

    if _reload_graph_at_index(idx):
        st.session_state["main_graph_index_input"] = str(idx)
        st.session_state.pop("_graph_index_error", None)
        # No explicit ``st.rerun()``: this runs as a widget ``on_change``
        # callback and Streamlit reruns the script automatically afterwards.


def _render_graph_index_input(active, error=None):
    """Graph index text input with compact inline validation styling."""
    if "main_graph_index_input" not in st.session_state:
        st.session_state["main_graph_index_input"] = str(active)

    with st.container(key="graph_index_input_block"):
        if error:
            st.markdown(
                """
                <style>
                div[data-testid="stTextInput"]:has(input[aria-label="Graph index"]) input {
                    border-color: #ff4b4b !important;
                    box-shadow: 0 0 0 1px #ff4b4b !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
        st.text_input(
            "Graph index",
            key="main_graph_index_input",
            on_change=_on_main_graph_index_change,
        )
        if error:
            st.markdown(
                f'<p style="color:#ff4b4b;font-size:0.8rem;margin:-0.4rem 0 0.25rem 0;">'
                f"{error}</p>",
                unsafe_allow_html=True,
            )


def _render_graph_sample_section():
    """Post-load graph sample index and live sampling controls."""
    if st.session_state.get("data") is None:
        return

    _dset0, rank_populations, hyperedge_population, _is_loaded = (
        _get_sampling_context()
    )
    meta = st.session_state.get("dataset_metadata") or {}
    is_inductive = (meta.get("learning_setting") or "").lower() == "inductive"

    if is_inductive:
        total = int(st.session_state.get("loaded_dataset_size", 1) or 1)
        max_idx = max(0, total - 1)
        active = int(st.session_state.get("active_graph_index", 0))
        err = st.session_state.get("_graph_index_error")
        _render_graph_index_input(active, error=err)
        st.caption(
            f"**Available graphs:** `0` to `{max_idx}` ({total} total)"
        )
    else:
        st.caption("Single graph (transductive dataset)")

    # ``value=`` omitted on purpose: ``_persist_widget_state`` re-seeds the
    # session-state entry on every run, so passing ``value=`` would trigger
    # Streamlit's "default value but also set via Session State" warning.
    st.session_state.setdefault("ui_min_degree", 0)
    st.slider(
        "Minimum degree",
        min_value=0,
        max_value=20,
        key="ui_min_degree",
        help="Drop nodes below this degree after rank caps are applied.",
        on_change=_on_sampling_control_change,
    )

    if rank_populations:
        summary = _format_rank_cap_summary(rank_populations, hyperedge_population)
        st.caption(summary)
        _render_rank_cap_controls(rank_populations, hyperedge_population)


def _do_load_graph(cfg, progress=None):
    """Build dataset, optional lifting, and embed-ready D3 HTML.

    On success, populates ``st.session_state`` with the new payload/HTML/data
    and returns ``True``. On any failure, surfaces ``st.error`` and returns
    ``False`` (without clearing the previously embedded view).
    """
    if progress is None:
        progress = lambda _msg: None

    progress("Loading dataset")
    try:
        raw_cached, loaded_domain = _load_dataset_cached(
            domain=cfg["selected_domain"],
            dataset_name=cfg["selected_dataset"],
        )
        raw = copy.deepcopy(raw_cached)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return False

    try:
        st.session_state["data_original"] = copy.deepcopy(raw)
    except Exception:
        st.session_state["data_original"] = raw

    total = len(raw) if hasattr(raw, "__len__") else 1
    st.session_state["loaded_dataset_size"] = total
    graph_index = 0

    st.session_state["dataset_metadata"] = extract_dataset_metadata(
        cfg["selected_domain"], cfg["selected_dataset"]
    )
    st.session_state["_load_cfg_snapshot"] = copy.deepcopy(cfg)

    current_data = raw
    applied_lift = None
    if cfg["use_lifting"] and cfg["selected_lifting"] is not None:
        if cfg["edited_lifting_errors"]:
            st.error(
                "Error applying lifting: invalid lifting config.\n- "
                + "\n- ".join(cfg["edited_lifting_errors"])
            )
            return False
        progress("Applying lifting transform")
        try:
            current_data, applied_lift = _build_sample_at_index(raw, graph_index, cfg)
        except Exception as e:
            st.error(f"Error applying lifting: {e}")
            return False
    elif hasattr(raw, "__getitem__"):
        current_data = [copy.deepcopy(raw[graph_index])]

    st.session_state["data"] = current_data
    st.session_state["lifting_applied"] = applied_lift
    st.session_state["data_domain"] = loaded_domain
    st.session_state["dataset_name"] = cfg["selected_dataset"]
    st.session_state["active_graph_index"] = 0
    st.session_state["main_graph_index_input"] = "0"

    dset0 = (
        current_data[0] if hasattr(current_data, "__getitem__") else current_data
    )

    progress("Enumerating neighborhoods")
    progress("Computing shared sampling")
    progress("Building graph payload")
    return _finalize_loaded_sample(
        dset0, cfg, loaded_domain, cfg["selected_dataset"]
    )


def _render_sidebar_tab_selector(data_loaded):
    """Large three-tab selector backed by session state (survives reruns)."""
    options = ["Load graph", "Explore", "Metrics"]

    # Apply a programmatic switch requested on a previous run (e.g. after a
    # successful load). Must happen BEFORE the tab buttons are rendered.
    pending = st.session_state.pop("_pending_tab", None)
    if pending in options:
        st.session_state["active_sidebar_tab"] = pending
    if "active_sidebar_tab" not in st.session_state:
        st.session_state["active_sidebar_tab"] = "Load graph"

    active = st.session_state.get("active_sidebar_tab") or "Load graph"
    col_load, col_explore, col_metrics = st.columns(3)
    with col_load:
        if st.button(
            "Load",
            key="tab_load_btn",
            type="primary" if active == "Load graph" else "secondary",
            width="stretch",
        ):
            st.session_state["active_sidebar_tab"] = "Load graph"
            st.rerun()
    with col_explore:
        if st.button(
            "Explore",
            key="tab_explore_btn",
            type="primary" if active == "Explore" else "secondary",
            width="stretch",
        ):
            st.session_state["active_sidebar_tab"] = "Explore"
            st.rerun()
    with col_metrics:
        if st.button(
            "Metrics",
            key="tab_metrics_btn",
            type="primary" if active == "Metrics" else "secondary",
            width="stretch",
        ):
            st.session_state["active_sidebar_tab"] = "Metrics"
            st.rerun()

    return active


def _render_explore_tab():
    """Post-load controls: neighborhoods (top), 3D view, and graph sample."""
    if st.session_state.get("data") is None:
        st.info("Load a graph from the **Load** tab to explore neighborhoods.")
        return

    _render_neighborhood_picker()

    st.divider()
    st.session_state.setdefault("color_neighborhoods_by_rank", False)
    st.toggle(
        "3D layered view (orbit/zoom)",
        help=(
            "Stack ranks as horizontal planes in 3D (multi-incidence, 2+ adjacency, "
            "or combined incidence+adjacency). Single-matrix views stay 2D."
        ),
        key="layered_3d_view",
        on_change=_on_layered_3d_toggle,
    )
    st.toggle(
        "Color neighborhoods by rank information",
        help=(
            "When off (default), each selected neighborhood is drawn in one solid "
            "colour (adjacency: cool greens/teals; incidence: warm oranges/yellows). "
            "When on, edges use rank-based gradients between source and target ranks."
        ),
        key="color_neighborhoods_by_rank",
        on_change=_on_color_by_rank_toggle,
    )

    st.divider()
    st.subheader("Graph sample")
    _render_graph_sample_section()


def _static_dataframe(rows):
    """Render a table at full height (no inner scrollbar, never clipped).

    ``st.table`` grows to fit every row, unlike ``st.dataframe`` which caps its
    height and scrolls. The index is hidden to keep the layout clean.
    """
    if not rows:
        return
    import pandas as pd

    df = pd.DataFrame(rows)
    try:
        st.table(df.style.hide(axis="index"))
    except Exception:
        st.table(df)


def _sparse_to_scipy(sparse_tensor):
    """Convert a coalesced torch sparse tensor to a scipy COO matrix."""
    import scipy.sparse as sp

    t = sparse_tensor.coalesce()
    idx = t.indices().cpu().numpy()
    vals = t.values().cpu().numpy()
    shape = tuple(t.shape)
    return sp.coo_matrix((vals, (idx[0], idx[1])), shape=shape)


def _spectral_radius(sparse_tensor):
    """Largest |eigenvalue| (square) or largest singular value (rectangular).

    For symmetric adjacency matrices these coincide. Small matrices are handled
    densely; larger ones use sparse iterative solvers with a dense fallback.
    """
    if sparse_tensor is None:
        return None
    try:
        import numpy as np

        mat = _sparse_to_scipy(sparse_tensor).asfptype()
        if min(mat.shape) == 0:
            return None
        square = mat.shape[0] == mat.shape[1]

        if min(mat.shape) <= 3:
            dense = mat.toarray()
            if square:
                return float(np.max(np.abs(np.linalg.eigvals(dense))))
            return float(np.max(np.linalg.svd(dense, compute_uv=False)))

        import scipy.sparse.linalg as sla

        try:
            if square:
                ev = sla.eigs(mat, k=1, which="LM", return_eigenvectors=False)
                return float(np.abs(ev[0]))
            sv = sla.svds(mat, k=1, return_singular_vectors=False)
            return float(sv[0])
        except Exception:
            dense = mat.toarray()
            if square:
                return float(np.max(np.abs(np.linalg.eigvals(dense))))
            return float(np.max(np.linalg.svd(dense, compute_uv=False)))
    except Exception:
        return None


def _neighborhood_clustering(sparse_tensor):
    """Average clustering coefficient for a square (adjacency) neighborhood."""
    if sparse_tensor is None:
        return None
    try:
        t = sparse_tensor.coalesce()
        if t.shape[0] != t.shape[1]:
            return None
        dim = int(t.shape[0])
        G, _nd = sparse_to_networkx(t, max_nodes=dim, min_degree=0)
        if G is None or G.number_of_nodes() == 0:
            return None
        UG = G.to_undirected() if G.is_directed() else G
        return float(nx.average_clustering(UG))
    except Exception:
        return None


def _compute_selected_structural_metrics():
    """Spectral radius (all selected) + clustering (adjacency only) per neighborhood.

    Computed on the full neighborhood matrices (not the sampled display) and
    cached against the current selection / graph index / lifting.
    """
    dset0 = st.session_state.get("data")
    if dset0 is None:
        return []
    selected_ids = list(st.session_state.get("selected_neighborhood_ids") or [])
    if not selected_ids:
        return []

    marker = _metrics_marker(selected_ids, expensive=False)
    cached = st.session_state.get("_struct_metrics")
    if cached and cached.get("marker") == marker:
        return cached.get("rows") or []

    rows = []
    for neigh_id in selected_ids:
        matrix, _desc, relation_ctx = get_named_visualization_matrix(dset0, neigh_id)
        if matrix is None:
            continue
        radius = _spectral_radius(matrix)
        is_adjacency = (relation_ctx or {}).get("type") == "adjacency"
        clustering = _neighborhood_clustering(matrix) if is_adjacency else None
        rows.append({
            "Neighborhood": neigh_id,
            "Spectral radius": gm.fmt_value(radius),
            "Clustering (adjacency only)": (
                gm.fmt_value(clustering) if clustering is not None else "—"
            ),
        })

    st.session_state["_struct_metrics"] = {"marker": marker, "rows": rows}
    return rows


def _render_dataset_info_section():
    """Whole-dataset info: YAML metadata + current graph node count + size."""
    domain = st.session_state.get("cfg_domain")
    dataset_name = st.session_state.get("dataset_name")
    meta = extract_dataset_metadata(domain, dataset_name) if domain and dataset_name else {}

    num_nodes = _node_count_from_dataset(st.session_state.get("data"))
    num_graphs = int(st.session_state.get("loaded_dataset_size", 1) or 1)

    st.subheader("Dataset")
    if dataset_name:
        st.caption(f"**{domain}** / **{dataset_name}**")
    col_left, col_right = st.columns(2)
    with col_left:
        st.caption(
            f"**number of nodes:** {num_nodes if num_nodes is not None else 'N/A'}"
        )
        st.caption(f"**number of graphs:** {num_graphs}")
        st.caption(f"**task:** {meta.get('task') or 'N/A'}")
        st.caption(f"**task_level:** {meta.get('task_level') or 'N/A'}")
    with col_right:
        st.caption(
            f"**num_features:** {_format_num_features(meta.get('num_features'))}"
        )
        st.caption(
            f"**learning_setting:** {meta.get('learning_setting') or 'N/A'}"
        )
        st.caption(f"**split_type:** {meta.get('split_type') or 'N/A'}")
        num_classes = meta.get("num_classes")
        st.caption(
            f"**num_classes:** {num_classes if num_classes is not None else 'N/A'}"
        )
    if num_nodes is not None and num_graphs > 1:
        st.caption("Node count is for the currently displayed graph.")
    st.divider()


def _render_metrics_tab():
    """Displayed-graph metrics tab: HUD toggle, whole-graph table, top elements."""
    if st.session_state.get("data") is None:
        st.info("Load a graph from the **Load** tab to see metrics.")
        return

    _render_dataset_info_section()

    err = st.session_state.get("_graph_metrics_error")
    if err:
        st.caption(f"Metrics unavailable: {err}")
        return

    cached = st.session_state.get("_graph_metrics") or {}
    metrics = cached.get("data")
    if not metrics:
        st.caption("Metrics will appear once a graph is displayed.")
        return

    st.session_state.setdefault("show_metrics_hud", True)
    st.toggle(
        "Show metrics overlay",
        key="show_metrics_hud",
        help="Compact whole-graph metrics in the top-left of the graph canvas.",
        on_change=_on_metrics_option_change,
    )

    flags = metrics.get("flags") or {}
    for note in flags.get("notes", []):
        st.caption(note)

    graph = metrics.get("graph") or {}
    scope = metrics.get("graph_scope") or {}
    label_map = dict(gm.GRAPH_METRIC_LABELS)
    rows = []
    for key, _label in gm.GRAPH_METRIC_LABELS:
        if key not in graph or graph.get(key) is None:
            continue
        label = label_map.get(key, key)
        if key in scope:
            label = f"{label} ({scope[key]})"
        rows.append({"Metric": label, "Value": gm.fmt_value(graph[key])})
    if rows:
        st.caption("Displayed graph (after sampling)")
        _static_dataframe(rows)

    struct_rows = _compute_selected_structural_metrics()
    if struct_rows:
        st.caption("Selected neighborhoods (structural)")
        _static_dataframe(struct_rows)

    _render_metrics_top_elements(metrics)

    with st.expander("Advanced metrics"):
        st.session_state.setdefault("metrics_expensive", False)
        st.checkbox(
            "Compute advanced metrics (betweenness, closeness, eccentricity, …)",
            key="metrics_expensive",
            help=(
                "Heavier centralities. May be slow or approximated on large "
                "graphs; betweenness uses sampling above ~800 nodes."
            ),
            on_change=_on_metrics_option_change,
        )
        if flags.get("betweenness_approx"):
            st.caption("Betweenness is approximated (k-sampled) for this graph.")
        if not flags.get("expensive"):
            st.caption("Enable to add per-node/edge centralities to tooltips and tables.")

    stats_doc = _build_stats_html(metrics)
    if stats_doc:
        st.download_button(
            label="Download stats (HTML)",
            data=stats_doc,
            file_name="topobench_metrics.html",
            mime="text/html",
            key="stats_download_main",
            width="stretch",
        )


def _build_stats_html(metrics):
    """Assemble the standalone stats HTML for the current view.

    Reuses the already-computed metric tables (whole-graph summary, selected
    neighborhood structural metrics, top nodes and Forman-Ricci extremes) plus
    the dataset metadata, and renders them as a branded, self-contained page via
    :func:`stats_html.build_stats_html`.

    Args:
        metrics: The metrics dict from ``st.session_state["_graph_metrics"]``.

    Returns:
        str | None: The HTML document, or ``None`` if no metrics are available.
    """
    if not metrics:
        return None

    domain = st.session_state.get("cfg_domain")
    dataset_name = st.session_state.get("dataset_name")
    meta = (
        extract_dataset_metadata(domain, dataset_name)
        if domain and dataset_name
        else {}
    )
    num_nodes = _node_count_from_dataset(st.session_state.get("data"))
    num_graphs = int(st.session_state.get("loaded_dataset_size", 1) or 1)

    payload = st.session_state.get("_d3_payload") or {}
    title = payload.get("title") or (dataset_name or "TopoExplorer")
    subtitle = payload.get("subtitle") or ""

    num_classes = meta.get("num_classes")
    sections = [
        {
            "heading": "Dataset",
            "columns": ["Field", "Value"],
            "rows": [
                ["Domain", domain or "N/A"],
                ["Dataset", dataset_name or "N/A"],
                ["Nodes (displayed graph)",
                 num_nodes if num_nodes is not None else "N/A"],
                ["Number of graphs", num_graphs],
                ["Task", meta.get("task") or "N/A"],
                ["Task level", meta.get("task_level") or "N/A"],
                ["Learning setting", meta.get("learning_setting") or "N/A"],
                ["Num features", _format_num_features(meta.get("num_features"))],
                ["Num classes", num_classes if num_classes is not None else "N/A"],
                ["Split type", meta.get("split_type") or "N/A"],
            ],
        }
    ]

    graph = metrics.get("graph") or {}
    scope = metrics.get("graph_scope") or {}
    label_map = dict(gm.GRAPH_METRIC_LABELS)
    g_rows = []
    for key, _label in gm.GRAPH_METRIC_LABELS:
        if key not in graph or graph.get(key) is None:
            continue
        label = label_map.get(key, key)
        if key in scope:
            label = f"{label} ({scope[key]})"
        g_rows.append([label, gm.fmt_value(graph[key])])
    sections.append({
        "heading": "Displayed graph (after sampling)",
        "columns": ["Metric", "Value"],
        "rows": g_rows,
    })

    struct_rows = _compute_selected_structural_metrics() or []
    if struct_rows:
        sections.append({
            "heading": "Selected neighborhoods (structural)",
            "columns": ["Neighborhood", "Spectral radius",
                        "Clustering (adjacency only)"],
            "rows": [
                [r.get("Neighborhood"), r.get("Spectral radius"),
                 r.get("Clustering (adjacency only)")]
                for r in struct_rows
            ],
        })

    nodes = metrics.get("nodes") or {}
    if nodes:
        centrality = st.session_state.get("metrics_top_centrality") or "degree_centrality"
        sample = next(iter(nodes.values()), {})
        if sample.get(centrality) is None:
            centrality = "degree_centrality"
        summary = gm.summarize_metrics(metrics, centrality=centrality, top_k=5)
        top_rows = [[nid, gm.fmt_value(val)] for nid, val in summary["top_nodes"]]
        if top_rows:
            sections.append({
                "heading": f"Top nodes by {centrality}",
                "columns": ["Node", "Value"],
                "rows": top_rows,
            })
        neg = summary["forman_most_negative"]
        if neg:
            sections.append({
                "heading": "Most negative Forman-Ricci edges",
                "columns": ["Edge", "Forman"],
                "rows": [[f"{u} — {v}", gm.fmt_value(val)] for (u, v), val in neg],
            })

    return stats_html.build_stats_html(sections, title=title, subtitle=subtitle)


def _render_metrics_top_elements(metrics):
    """Top-k nodes by a chosen centrality plus Forman-Ricci extremes."""
    nodes = metrics.get("nodes") or {}
    if not nodes:
        return

    sample = next(iter(nodes.values()), {})
    centrality_opts = [
        (k, lbl)
        for k, lbl in [
            ("degree_centrality", "Degree centrality"),
            ("betweenness", "Betweenness"),
            ("closeness", "Closeness"),
            ("eigenvector", "Eigenvector"),
            ("pagerank", "PageRank"),
            ("clustering", "Clustering"),
        ]
        if k in sample and sample.get(k) is not None
    ]
    if centrality_opts:
        keys = [k for k, _ in centrality_opts]
        labels = {k: lbl for k, lbl in centrality_opts}
        choice = st.selectbox(
            "Top nodes by",
            options=keys,
            format_func=lambda k: labels.get(k, k),
            key="metrics_top_centrality",
        )
        summary = gm.summarize_metrics(metrics, centrality=choice, top_k=5)
        top_rows = [
            {"Node": nid, labels.get(choice, choice): gm.fmt_value(val)}
            for nid, val in summary["top_nodes"]
        ]
        if top_rows:
            _static_dataframe(top_rows)

    edges = metrics.get("edges") or {}
    if edges:
        summary = gm.summarize_metrics(metrics, top_k=5)
        neg = summary["forman_most_negative"]
        if neg:
            st.caption("Most negative Forman-Ricci edges")
            _static_dataframe(
                [
                    {"Edge": f"{u} — {v}", "Forman": gm.fmt_value(val)}
                    for (u, v), val in neg
                ]
            )


def _render_graph_canvas():
    """Main area: D3 embed (title + graph) and download directly below."""
    embed_html = st.session_state.get("_d3_embed_html")
    if not embed_html:
        return

    sel_ids = list(st.session_state.get("selected_neighborhood_ids") or [])
    neigh_key = "+".join(sel_ids) if sel_ids else "default"
    with st.container(key=f"d3_embed::{neigh_key}"):
        st.iframe(embed_html, height=D3_EMBED_HEIGHT)

    last_html = st.session_state.get("_d3_last_html")
    if last_html:
        st.download_button(
            label="Download graph (HTML)",
            data=last_html,
            file_name="topobench_graph.html",
            mime="text/html",
            key="d3_download_main",
            width="stretch",
        )


_PERSIST_WIDGET_KEYS = frozenset(
    {
        "use_lifting",
        "ui_min_degree",
        "ui_hyperedge_cap",
        "ui_set_all_rank_caps",
        "layered_3d_view",
        "main_graph_index_input",
        "show_metrics_hud",
        "metrics_expensive",
    }
)


def _persist_widget_state():
    """Keep sidebar widget selections alive across tab switches.

    Streamlit drops the session_state entry of any keyed widget that is not
    rendered on a run (e.g. Explore/Metrics widgets while the Load graph tab is
    showing). Re-assigning those keys here -- before any widget is instantiated
    this run -- opts them out of that cleanup so selections survive when the
    user switches sidebar tabs.
    """
    for key in list(st.session_state.keys()):
        if key in _PERSIST_WIDGET_KEYS or key.startswith("ui_rank_cap_"):
            st.session_state[key] = st.session_state[key]


def main():
    """Render the full Streamlit app: sidebar controls and the graph view.

    This is the entry point invoked on every Streamlit rerun. It restores
    persisted widget state, seeds first-load defaults (domain ``graph`` /
    dataset ``MUTAG``, lifting enabled), builds the sidebar configuration
    (dataset, lifting, neighborhoods, sampling and visualization options) and
    renders the resulting Hasse-graph view together with the metrics panel.

    Side effects:
        Reads and writes ``st.session_state`` and emits Streamlit UI elements.
    """
    _persist_widget_state()

    flash = st.session_state.pop("_flash_ok", None)
    if flash:
        st.success(flash)

    available_datasets = discover_available_datasets()

    # Default selection on first load: domain "graph", dataset "MUTAG".
    if "cfg_domain" not in st.session_state and "graph" in available_datasets:
        st.session_state["cfg_domain"] = "graph"
        if "MUTAG" in available_datasets.get("graph", []):
            st.session_state["cfg_dataset"] = "MUTAG"

    # "Use lifting" toggle defaults to on for first-time visitors.
    if "use_lifting" not in st.session_state:
        st.session_state["use_lifting"] = True

    if "layered_3d_view" not in st.session_state:
        st.session_state["layered_3d_view"] = True

    data_loaded = st.session_state.get("data") is not None

    sidebar_cfg = None
    with st.sidebar:
        st.markdown(
            """
            <div class="topo-sidebar-brand">
              <div class="topo-sidebar-title">TopoExplorer</div>
              <div class="topo-sidebar-tagline">built on TopoBench</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        tab = _render_sidebar_tab_selector(data_loaded)
        st.divider()
        if tab == "Load graph":
            sidebar_cfg = _render_left_config(available_datasets)
        elif tab == "Metrics":
            _render_metrics_tab()
        else:
            _render_explore_tab()

    if sidebar_cfg is not None and sidebar_cfg["load_clicked"]:
        load_cfg = {**sidebar_cfg, **_default_load_sampling_cfg()}
        with st.status("Loading graph…", expanded=True) as status:
            def _progress(msg):
                """Relay a load-progress message to the Streamlit status box."""
                status.update(label=f"Loading graph… {msg}", state="running")
                status.write(msg)

            ok = _do_load_graph(load_cfg, progress=_progress)
            if ok:
                status.update(label="Graph loaded", state="complete")
            else:
                status.update(label="Load stopped", state="error")
        if ok:
            st.session_state["_pending_tab"] = "Explore"
            st.rerun()

    _render_graph_canvas()


if __name__ == "__main__":
    main()