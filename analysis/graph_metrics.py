"""Graph metrics for the Neighborhood Explorer (displayed-graph only).

All metrics are computed on the *displayed* NetworkX graph (after sampling and
lifting), on an undirected view so the same code path covers adjacency,
incidence/bipartite, and layered views.

Metric tiers
------------
- cheap (always): counts, density, degree stats, components, clustering,
  transitivity, per-node degree centrality, per-edge Forman-Ricci.
- default (size-gated, largest connected component): diameter, radius.
- advanced (opt-in, expensive): betweenness, closeness, eccentricity,
  eigenvector, pagerank, edge betweenness, degree assortativity.

Forman-Ricci uses the Weber-Jost-Saucan (2018) definition; with unit weights
(the displayed graph carries none) it reduces exactly to ``4 - deg(u) - deg(v)``
(no triangle term), matching TopoBench's implementation.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

# Above these sizes, distance measures (diameter/radius) are skipped by default
# and betweenness is approximated via k-sampling.
DISTANCE_NODE_LIMIT = 1200
DISTANCE_EDGE_LIMIT = 6000
BETWEENNESS_EXACT_LIMIT = 800
BETWEENNESS_SAMPLE_K = 500


def _edge_key(u, v):
    """Order-independent key for an undirected edge (string node ids)."""
    a, b = str(u), str(v)
    return (a, b) if a <= b else (b, a)


def forman_ricci_unweighted(UG: nx.Graph, u, v) -> int:
    """Weber-Jost-Saucan Forman-Ricci with unit weights: ``4 - deg(u) - deg(v)``."""
    return 4 - UG.degree(u) - UG.degree(v)


def compute_graph_metrics(
    G: nx.Graph,
    *,
    view_label: str = "graph",
    expensive: bool = False,
    betweenness_k: int | None = None,
) -> dict[str, Any]:
    """Compute displayed-graph metrics.

    Parameters
    ----------
    G : networkx graph
        The displayed graph (may be directed for incidence/bipartite views).
    view_label : str
        Human-readable view name (e.g. ``"adjacency"``) for UI captions.
    expensive : bool
        When True, also compute advanced per-node/edge centralities.
    betweenness_k : int, optional
        Sample size for approximate betweenness; defaults to an internal policy.

    Returns
    -------
    dict
        ``{"graph": {...}, "graph_scope": {...}, "nodes": {...}, "edges": {...},
        "flags": {...}}``.
    """
    notes: list[str] = []
    directed = G.is_directed()
    UG = G.to_undirected() if directed else G
    if directed:
        notes.append(f"Computed on an undirected view of the {view_label} graph.")

    n = UG.number_of_nodes()
    m = UG.number_of_edges()
    degrees = dict(UG.degree())
    deg_values = list(degrees.values()) if degrees else [0]

    graph: dict[str, Any] = {
        "nodes": n,
        "edges": m,
        "density": nx.density(UG) if n > 1 else 0.0,
        "avg_degree": (sum(deg_values) / n) if n else 0.0,
        "max_degree": max(deg_values),
        "min_degree": min(deg_values),
    }

    components = list(nx.connected_components(UG))
    graph["components"] = len(components)
    largest_cc_nodes = max(components, key=len) if components else set()
    graph["largest_cc_size"] = len(largest_cc_nodes)

    try:
        graph["avg_clustering"] = nx.average_clustering(UG) if n else 0.0
    except Exception:
        graph["avg_clustering"] = None
    try:
        graph["transitivity"] = nx.transitivity(UG) if n else 0.0
    except Exception:
        graph["transitivity"] = None

    graph_scope: dict[str, str] = {}

    # Default tier: distance measures on the largest connected component.
    graph["diameter"] = None
    graph["radius"] = None
    distance_skipped = n > DISTANCE_NODE_LIMIT or m > DISTANCE_EDGE_LIMIT
    if distance_skipped:
        notes.append(
            "Diameter/radius skipped (graph too large); enable advanced metrics "
            "or reduce sampling."
        )
    elif largest_cc_nodes:
        cc = UG.subgraph(largest_cc_nodes)
        try:
            ecc = nx.eccentricity(cc)
            graph["diameter"] = max(ecc.values()) if ecc else None
            graph["radius"] = min(ecc.values()) if ecc else None
            if graph["components"] > 1:
                graph_scope["diameter"] = "largest CC"
                graph_scope["radius"] = "largest CC"
        except Exception:
            notes.append("Diameter/radius unavailable for this graph.")

    # Per-node cheap metrics.
    nodes: dict[str, dict[str, Any]] = {}
    deg_centrality = nx.degree_centrality(UG) if n else {}
    try:
        clustering = nx.clustering(UG)
    except Exception:
        clustering = {}
    for node in UG.nodes():
        key = str(node)
        nodes[key] = {
            "degree": degrees.get(node, 0),
            "degree_centrality": deg_centrality.get(node),
            "clustering": clustering.get(node),
        }

    # Per-edge cheap metrics (Forman-Ricci).
    edges: dict[tuple, dict[str, Any]] = {}
    for u, v in UG.edges():
        edges[_edge_key(u, v)] = {"forman_ricci": forman_ricci_unweighted(UG, u, v)}

    flags: dict[str, Any] = {
        "view_label": view_label,
        "expensive": bool(expensive),
        "betweenness_approx": False,
        "distance_skipped": distance_skipped,
        "notes": notes,
    }

    if expensive and n:
        _add_advanced_metrics(
            UG, nodes, edges, graph, flags, betweenness_k=betweenness_k
        )

    return {
        "graph": graph,
        "graph_scope": graph_scope,
        "nodes": nodes,
        "edges": edges,
        "flags": flags,
    }


def _add_advanced_metrics(UG, nodes, edges, graph, flags, *, betweenness_k=None):
    """Augment node/edge dicts with expensive centralities (in place)."""
    n = UG.number_of_nodes()

    # Betweenness: exact for small graphs, k-sampled otherwise.
    if betweenness_k is None:
        betweenness_k = (
            None if n <= BETWEENNESS_EXACT_LIMIT else min(n, BETWEENNESS_SAMPLE_K)
        )
    try:
        if betweenness_k is not None and betweenness_k < n:
            betw = nx.betweenness_centrality(UG, k=betweenness_k, seed=0)
            flags["betweenness_approx"] = True
            flags["notes"].append(
                f"Betweenness approximated with k={betweenness_k} samples."
            )
        else:
            betw = nx.betweenness_centrality(UG)
    except Exception:
        betw = {}

    try:
        ebetw = nx.edge_betweenness_centrality(UG)
    except Exception:
        ebetw = {}

    try:
        closeness = nx.closeness_centrality(UG)
    except Exception:
        closeness = {}

    # Eccentricity per node on the largest CC (finite only there).
    eccentricity: dict[Any, Any] = {}
    try:
        components = list(nx.connected_components(UG))
        if components:
            cc_nodes = max(components, key=len)
            eccentricity = nx.eccentricity(UG.subgraph(cc_nodes))
    except Exception:
        eccentricity = {}

    try:
        eigenvector = nx.eigenvector_centrality(UG, max_iter=500, tol=1e-04)
    except Exception:
        eigenvector = {}
        flags["notes"].append("Eigenvector centrality did not converge.")

    try:
        pagerank = nx.pagerank(UG)
    except Exception:
        pagerank = {}

    for node in UG.nodes():
        key = str(node)
        slot = nodes.setdefault(key, {})
        slot["betweenness"] = betw.get(node)
        slot["closeness"] = closeness.get(node)
        slot["eccentricity"] = eccentricity.get(node)
        slot["eigenvector"] = eigenvector.get(node)
        slot["pagerank"] = pagerank.get(node)

    for (u, v), val in ebetw.items():
        edges.setdefault(_edge_key(u, v), {})["edge_betweenness"] = val

    try:
        graph["assortativity"] = nx.degree_assortativity_coefficient(UG)
    except Exception:
        graph["assortativity"] = None


# ---------------------------------------------------------------------------
# Formatting helpers (kept here so the app stays lean and the format is shared)
# ---------------------------------------------------------------------------

def fmt_value(v: Any) -> str:
    """Compact human-readable formatting for metric values."""
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) >= 1000 or abs(v) < 0.001:
            return f"{v:.2e}"
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


# Labels for whole-graph metrics, in display order.
GRAPH_METRIC_LABELS = [
    ("nodes", "Nodes"),
    ("edges", "Edges"),
    ("density", "Density"),
    ("avg_degree", "Avg degree"),
    ("max_degree", "Max degree"),
    ("min_degree", "Min degree"),
    ("components", "Components"),
    ("largest_cc_size", "Largest CC size"),
    ("avg_clustering", "Avg clustering"),
    ("transitivity", "Transitivity"),
    ("diameter", "Diameter"),
    ("radius", "Radius"),
    ("assortativity", "Degree assortativity"),
]

# Compact subset for the floating HUD overlay.
HUD_METRIC_KEYS = ["nodes", "edges", "density", "avg_degree", "components", "diameter"]

NODE_TOOLTIP_LABELS = [
    ("degree_centrality", "deg cent"),
    ("clustering", "clustering"),
    ("betweenness", "betw"),
    ("closeness", "closeness"),
    ("eccentricity", "ecc"),
]

EDGE_TOOLTIP_LABELS = [
    ("forman_ricci", "Forman"),
    ("edge_betweenness", "edge betw"),
]


def build_hud_rows(metrics: dict) -> list[dict]:
    """Compact ``[{label, value}]`` rows for the canvas HUD overlay."""
    graph = metrics.get("graph", {})
    scope = metrics.get("graph_scope", {})
    label_map = dict(GRAPH_METRIC_LABELS)
    rows = []
    for key in HUD_METRIC_KEYS:
        if key not in graph or graph.get(key) is None:
            continue
        label = label_map.get(key, key)
        if key in scope:
            label = f"{label} ({scope[key]})"
        rows.append({"label": label, "value": fmt_value(graph[key])})
    return rows


def node_payload_metrics(metrics: dict, node_id: str) -> dict:
    """Formatted per-node metric strings for a D3 node tooltip."""
    nd = (metrics.get("nodes") or {}).get(str(node_id))
    if not nd:
        return {}
    out = {}
    for key, label in NODE_TOOLTIP_LABELS:
        if key in nd and nd[key] is not None:
            out[label] = fmt_value(nd[key])
    return out


def edge_payload_metrics(metrics: dict, u: str, v: str) -> dict:
    """Formatted per-edge metric strings for a D3 edge tooltip."""
    ed = (metrics.get("edges") or {}).get(_edge_key(u, v))
    if not ed:
        return {}
    out = {}
    for key, label in EDGE_TOOLTIP_LABELS:
        if key in ed and ed[key] is not None:
            out[label] = fmt_value(ed[key])
    return out


def summarize_metrics(metrics: dict, centrality: str = "degree_centrality",
                      top_k: int = 5) -> dict:
    """Top-k nodes by a centrality and Forman-Ricci extremes for the Explore tab."""
    nodes = metrics.get("nodes") or {}
    edges = metrics.get("edges") or {}

    ranked = [
        (nid, vals.get(centrality))
        for nid, vals in nodes.items()
        if vals.get(centrality) is not None
    ]
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    top_nodes = ranked[:top_k]

    forman = [
        (key, vals["forman_ricci"])
        for key, vals in edges.items()
        if vals.get("forman_ricci") is not None
    ]
    forman.sort(key=lambda kv: kv[1])
    most_negative = forman[:top_k]
    most_positive = list(reversed(forman[-top_k:])) if forman else []

    return {
        "centrality": centrality,
        "top_nodes": top_nodes,
        "forman_most_negative": most_negative,
        "forman_most_positive": most_positive,
    }
