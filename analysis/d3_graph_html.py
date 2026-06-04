"""Self-contained HTML pages with D3 v7 force layout for TopoBench graph payloads."""

from __future__ import annotations

import html as html_module
import json
from typing import Any


def build_standalone_d3_html(
    payload: dict[str, Any],
    *,
    embed: bool = False,
    chart_min_height: int = 480,
    cache_marker: str | None = None,
) -> str:
    """
    Build a full HTML document (D3 from jsDelivr).

    Parameters
    ----------
    payload : dict
        Graph payload (see keys below).
    embed : bool, optional
        When True, size the chart for a fixed-height iframe (Streamlit
        ``components.html``) instead of the full viewport. Default ``False``.
    chart_min_height : int, optional
        Minimum chart height in pixels when ``embed=True``. Default ``480``.
    cache_marker : str, optional
        If set, emitted as an HTML comment in ``<head>`` so successive embeds
        differ (helps Streamlit / browser iframe cache busting).

    Payload keys
    ------------
    graphType : 'adjacency' | 'bipartite' | 'layered' | 'layered3d'
    title, subtitle : str
    nodes : list of dicts with id, label, degree, color, layer (0|1 for bipartite),
            optional stroke (CSS color for ring highlight)
    links : list of {source, target} string ids; optional color (stroke) and kind
    """
    # Do not html.escape the JSON: browsers often expose raw ``&quot;`` in
    # script textContent, which breaks ``JSON.parse``. Embed raw JSON and
    # break accidental ``</script>`` sequences in the serialized text.
    json_text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    embedded_json = json_text.replace("</", "\\u003c/")
    title_esc = html_module.escape(payload.get("title") or "Graph")

    if embed:
        chart_size_css = (
            f"width: 100%; min-height: {int(chart_min_height)}px; "
            "height: calc(100% - 72px);"
        )
        body_extra = "html, body { height: 100%; }"
    else:
        chart_size_css = "width: 100vw; min-height: 480px; height: calc(100vh - 88px);"
        body_extra = ""

    cache_comment = ""
    if cache_marker:
        cache_comment = (
            f"  <!-- cache_marker: {html_module.escape(str(cache_marker), quote=True)} -->\n"
        )

    use_3d = payload.get("graphType") == "layered3d"
    force_graph_3d_script = (
        '  <script src="https://cdn.jsdelivr.net/npm/3d-force-graph"></script>\n'
        if use_3d
        else ""
    )

    # Split HTML: f-string cannot embed arbitrary JSON (``{`` / ``}`` in payload).
    _head = f"""<!DOCTYPE html>
<html lang="en">
<head>
{cache_comment}  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title_esc}</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
{force_graph_3d_script}  <style>
    * {{ box-sizing: border-box; }}
    {body_extra}
    body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 0; background: #f0f0f0; }}
    header {{ padding: 12px 16px; background: #fff; border-bottom: 1px solid #ccc; }}
    header h1 {{ margin: 0; font-size: 1.1rem; }}
    header p {{ margin: 4px 0 0; color: #555; font-size: 0.85rem; }}
    header p:empty {{ display: none; margin: 0; }}
    #chart-wrap {{ position: relative; {chart_size_css} }}
    #chart {{ width: 100%; height: 100%; min-height: inherit; }}
    #err {{ color: #b00; padding: 16px; white-space: pre-wrap; font-family: monospace; }}
    svg {{ display: block; background: #fafafa; width: 100%; height: 100%; }}
    .node circle {{ stroke-width: 2px; cursor: grab; }}
    .node text {{ font-size: 9px; pointer-events: none; fill: #111; }}
    line.link {{ stroke-opacity: 0.85; }}
    #legend {{
      display: none; position: absolute; top: 10px; right: 10px; z-index: 5;
      background: rgba(255, 255, 255, 0.92); border: 1px solid #d0d0d0;
      border-radius: 6px; padding: 8px 10px; font-size: 12px; line-height: 1.35;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08); pointer-events: none;
      max-width: min(220px, 60%);
    }}
    #legend .legend-title {{ display: block; font-weight: 600; font-size: 11px;
      color: #555; letter-spacing: 0.02em; text-transform: uppercase;
      margin-bottom: 6px; }}
    #legend .legend-row {{ display: flex; align-items: center; gap: 8px;
      margin: 3px 0; }}
    #legend .legend-dot {{ display: inline-block; width: 12px; height: 12px;
      border-radius: 50%; border: 1px solid rgba(0,0,0,0.18); flex-shrink: 0; }}
    #legend .legend-label {{ color: #222; }}
  </style>
</head>
<body>
  <header>
    <h1 id="hdr-title"></h1>
    <p id="hdr-sub"></p>
  </header>
  <div id="err"></div>
  <div id="chart-wrap">
    <div id="legend" aria-label="Rank legend"></div>
    <div id="chart"></div>
  </div>
  <script type="application/json" id="graph-payload">"""

    # Plain string (not f-string): JavaScript uses single `{` / `}`.
    _tail = """
</script>
  <script>
  (function() {
    function showError(msg) {
      document.getElementById("err").textContent = msg;
    }
    try {
      const raw = document.getElementById("graph-payload").textContent;
      const payload = JSON.parse(raw);
      document.getElementById("hdr-title").textContent = payload.title || "Graph";
      document.getElementById("hdr-sub").textContent = payload.subtitle || "";

      (function renderLegend() {
        var box = document.getElementById("legend");
        if (!box) return;
        var entries = (payload.legend || []).filter(function(e) { return e && e.color; });
        if (entries.length === 0) {
          box.innerHTML = "";
          box.style.display = "none";
          return;
        }
        var html = '<span class="legend-title">Legend</span>';
        entries.forEach(function(e) {
          var label = e.label || ("Rank " + e.rank);
          html += '<div class="legend-row">'
            + '<span class="legend-dot" style="background:' + e.color + '"></span>'
            + '<span class="legend-label">' + label + '</span>'
            + '</div>';
        });
        box.innerHTML = html;
        box.style.display = "block";
      })();

      const nodes = (payload.nodes || []).map(function(d) {
        const o = Object.assign({}, d);
        o.id = String(o.id);
        return o;
      });
      if (nodes.length === 0) {
        showError("No nodes in graph payload.");
        return;
      }

      if (payload.graphType === "layered3d") {
        if (typeof ForceGraph3D === "undefined") {
          showError("3d-force-graph failed to load from CDN. Connect to the internet or use a recent browser.");
          return;
        }
        const chart = document.getElementById("chart");
        const width = Math.max(420, chart.clientWidth || window.innerWidth || 800);
        const height = Math.max(420, chart.clientHeight || (window.innerHeight - 100) || 600);
        const layersSorted = (payload.layers || []).slice().sort(function(a, b) { return a - b; });
        const levelDist = (payload.dagLevelDistance !== undefined) ? Number(payload.dagLevelDistance) : 140;
        const center = (layersSorted.length - 1) / 2;
        const yByRank = {};
        layersSorted.forEach(function(rank, i) {
          yByRank[String(rank)] = (i - center) * levelDist;
        });
        const nodes3d = nodes.map(function(n) {
          const rank = (n.layer === undefined ? 0 : n.layer);
          const y = yByRank[String(rank)];
          return Object.assign({}, n, {
            fy: (y === undefined ? 0 : y)
          });
        });
        const links3d = (payload.links || []).map(function(l) {
          return {
            source: String(l.source),
            target: String(l.target),
            color: l.color
          };
        });
        const graph3d = ForceGraph3D()(chart)
          .width(width)
          .height(height)
          .backgroundColor("#fafafa")
          .nodeRelSize(4)
          .nodeColor(function(n) { return n.color || "#666"; })
          .nodeLabel(function(n) { return (n.label || n.id) + " — degree " + (n.degree || 0); })
          .nodeVal(function(n) { return 1 + Math.log1p(n.degree || 1); })
          .linkColor(function(l) { return l.color || "#888"; })
          .linkOpacity(0.6)
          .linkWidth(0.6)
          .graphData({ nodes: nodes3d, links: links3d });
        var chargeForce = graph3d.d3Force("charge");
        if (chargeForce) chargeForce.strength(-40);
        var linkForce = graph3d.d3Force("link");
        if (linkForce) linkForce.distance(40).strength(0.7);

        (function() {
          var ro = new ResizeObserver(function() {
            if (!chart) return;
            var w = Math.max(320, chart.clientWidth || 800);
            var h = Math.max(320, chart.clientHeight || 600);
            graph3d.width(w).height(h);
          });
          ro.observe(chart);
        })();
        return;
      }

      if (typeof d3 === "undefined") {
        showError("D3 failed to load from CDN. Connect to the internet or open this file in Edge/Chrome.");
        return;
      }

      const nodeById = new Map(nodes.map(function(d) { return [String(d.id), d]; }));
      const links = (payload.links || []).map(function(l) {
        const s = nodeById.get(String(l.source));
        const t = nodeById.get(String(l.target));
        return {
          source: s,
          target: t,
          color: l.color
        };
      }).filter(function(l) { return l.source && l.target; });

      const chart = document.getElementById("chart");
      const width = Math.max(420, chart.clientWidth || window.innerWidth || 800);
      const height = Math.max(420, chart.clientHeight || (window.innerHeight - 100) || 600);

      const svg = d3.select("#chart").append("svg")
        .attr("width", width).attr("height", height)
        .attr("viewBox", [0, 0, width, height]);

      const g = svg.append("g");
      const zoom = d3.zoom()
        .scaleExtent([0.12, 6])
        .on("zoom", function(ev) { g.attr("transform", ev.transform); });
      svg.call(zoom);

      const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(function(d) { return d.id; }).distance(48).strength(0.62))
        .force("charge", d3.forceManyBody().strength(function() {
          return payload.graphType === "bipartite" ? -100 : -180;
        }))
        .force("collision", d3.forceCollide().radius(function(d) {
          return 6 + Math.min(18, Math.sqrt((d.degree || 1) + 1) * 2.4);
        }));

      if (payload.graphType === "bipartite") {
        simulation
          .force("x", d3.forceX(width / 2).strength(0.08))
          .force("y", d3.forceY(function(d) {
            return d.layer === 0 ? height * 0.24 : height * 0.76;
          }).strength(0.9));
      } else if (payload.graphType === "layered") {
        var layers = (payload.layers || []).slice().sort(function(a, b) { return a - b; });
        if (layers.length === 0) {
          var ls = new Set();
          nodes.forEach(function(n) { ls.add(n.layer || 0); });
          layers = Array.from(ls).sort(function(a, b) { return a - b; });
        }
        var topPad = 60;
        var bottomPad = 40;
        var usable = Math.max(120, height - topPad - bottomPad);
        var step = layers.length > 1 ? usable / (layers.length - 1) : 0;
        var layerY = {};
        layers.forEach(function(rank, i) {
          // Lowest rank at the bottom, highest rank at the top.
          var idxFromTop = layers.length - 1 - i;
          layerY[String(rank)] = topPad + idxFromTop * step;
        });
        simulation
          .force("x", d3.forceX(width / 2).strength(0.05))
          .force("y", d3.forceY(function(d) {
            var y = layerY[String(d.layer)];
            return (y === undefined) ? height / 2 : y;
          }).strength(1.1));

        var bandG = g.append("g").attr("class", "layer-bands");
        var labelDict = payload.layerLabels || {};
        layers.forEach(function(rank) {
          var y = layerY[String(rank)];
          if (y === undefined) return;
          bandG.append("line")
            .attr("x1", 0).attr("x2", width)
            .attr("y1", y).attr("y2", y)
            .attr("stroke", "#ccc")
            .attr("stroke-dasharray", "4 6")
            .attr("stroke-width", 1);
          bandG.append("text")
            .attr("x", width - 8).attr("y", y - 6)
            .attr("text-anchor", "end")
            .attr("fill", "#666")
            .attr("font-size", "11px")
            .text(labelDict[String(rank)] || ("Rank " + rank));
        });
      } else {
        simulation.force("center", d3.forceCenter(width / 2, height / 2));
      }

      const link = g.append("g").selectAll("line")
        .data(links)
        .join("line")
        .attr("class", "link")
        .attr("stroke", function(d) { return d.color || "#888"; })
        .attr("stroke-width", 1.2);

      const node = g.append("g").selectAll("g")
        .data(nodes)
        .join("g")
        .attr("class", "node")
        .call(d3.drag()
          .on("start", function(ev, d) {
            if (!ev.active) simulation.alphaTarget(0.35).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on("drag", function(ev, d) {
            d.fx = ev.x; d.fy = ev.y;
          })
          .on("end", function(ev, d) {
            if (!ev.active) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
          }));

      node.append("circle")
        .attr("r", function(d) {
          return 5 + Math.min(16, Math.log1p(d.degree || 1) * 3.2);
        })
        .attr("fill", function(d) { return d.color || "#666"; })
        .attr("stroke", function(d) { return d.stroke || "#fff"; });

      node.append("title").text(function(d) {
        return (d.label || d.id) + " — degree " + (d.degree || 0);
      });

      node.filter(function(d) { return String(d.id).length <= 12; })
        .append("text")
        .attr("dx", 10)
        .attr("dy", 4)
        .text(function(d) { return d.id; });

      simulation.on("tick", function() {
        link
          .attr("x1", function(d) { return d.source.x; })
          .attr("y1", function(d) { return d.source.y; })
          .attr("x2", function(d) { return d.target.x; })
          .attr("y2", function(d) { return d.target.y; });
        node.attr("transform", function(d) {
          return "translate(" + d.x + "," + d.y + ")";
        });
      });
    } catch (e) {
      showError((e && e.stack) ? e.stack : String(e));
    }
  })();
  </script>
</body>
</html>
"""

    return _head + embedded_json + _tail