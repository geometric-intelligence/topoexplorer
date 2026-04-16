"""Self-contained HTML pages with D3 v7 force layout for TopoBench graph payloads."""

from __future__ import annotations

import html as html_module
import json
from typing import Any


def build_standalone_d3_html(payload: dict[str, Any]) -> str:
    """
    Build a full HTML document (D3 from jsDelivr).

    Payload keys
    ------------
    graphType : 'adjacency' | 'bipartite'
    title, subtitle : str
    nodes : list of dicts with id, label, degree, color, layer (0|1 for bipartite),
            optional stroke (CSS color for ring highlight)
    links : list of {source, target} string ids
    """
    # Do not html.escape the JSON: browsers often expose raw ``&quot;`` in
    # script textContent, which breaks ``JSON.parse``. Embed raw JSON and
    # break accidental ``</script>`` sequences in the serialized text.
    json_text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    embedded_json = json_text.replace("</", "\\u003c/")
    title_esc = html_module.escape(payload.get("title") or "Graph")

    # Split HTML: f-string cannot embed arbitrary JSON (``{`` / ``}`` in payload).
    _head = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title_esc}</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 0; background: #f0f0f0; }}
    header {{ padding: 12px 16px; background: #fff; border-bottom: 1px solid #ccc; }}
    header h1 {{ margin: 0; font-size: 1.1rem; }}
    header p {{ margin: 4px 0 0; color: #555; font-size: 0.85rem; }}
    #chart {{ width: 100vw; min-height: 480px; height: calc(100vh - 88px); }}
    #err {{ color: #b00; padding: 16px; white-space: pre-wrap; font-family: monospace; }}
    svg {{ display: block; background: #fafafa; }}
    .node circle {{ stroke-width: 2px; cursor: grab; }}
    .node text {{ font-size: 9px; pointer-events: none; fill: #111; }}
    line.link {{ stroke: #888; stroke-opacity: 0.75; }}
  </style>
</head>
<body>
  <header>
    <h1 id="hdr-title"></h1>
    <p id="hdr-sub"></p>
  </header>
  <div id="err"></div>
  <div id="chart"></div>
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
      if (typeof d3 === "undefined") {
        showError("D3 failed to load from CDN. Connect to the internet or open this file in Edge/Chrome.");
        return;
      }
      const raw = document.getElementById("graph-payload").textContent;
      const payload = JSON.parse(raw);
      document.getElementById("hdr-title").textContent = payload.title || "Graph";
      document.getElementById("hdr-sub").textContent = payload.subtitle || "";

      const nodes = (payload.nodes || []).map(function(d) {
        const o = Object.assign({}, d);
        o.id = String(o.id);
        return o;
      });
      if (nodes.length === 0) {
        showError("No nodes in graph payload.");
        return;
      }

      const nodeById = new Map(nodes.map(function(d) { return [String(d.id), d]; }));
      const links = (payload.links || []).map(function(l) {
        const s = nodeById.get(String(l.source));
        const t = nodeById.get(String(l.target));
        return { source: s, target: t };
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
      } else {
        simulation.force("center", d3.forceCenter(width / 2, height / 2));
      }

      const link = g.append("g").selectAll("line")
        .data(links)
        .join("line")
        .attr("class", "link")
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