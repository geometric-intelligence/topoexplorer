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
    # ``3d-force-graph`` is ESM-only as of 1.75+ and requires
    # ``three >= 0.179``, which itself dropped UMD ``build/three.min.js``
    # somewhere around r150. Loading them via plain ``<script src>`` tags
    # therefore no longer works. Instead we use jsdelivr's ``/+esm`` URLs,
    # which auto-bundle each package as a single ESM module with all
    # transitive dependencies inlined. We expose both as globals (via a
    # promise on ``window``) so the rest of our inline non-module script
    # can pick them up once the modules finish loading.
    force_graph_3d_script = (
        '  <script type="module">\n'
        '    window.__force3dPromise = (async () => {\n'
        '      const [fgMod, threeMod] = await Promise.all([\n'
        '        import("https://cdn.jsdelivr.net/npm/3d-force-graph@1.80.0/+esm"),\n'
        '        import("https://cdn.jsdelivr.net/npm/three@0.179.1/+esm")\n'
        '      ]);\n'
        '      const FG = fgMod.default || fgMod.ForceGraph3D || fgMod;\n'
        '      window.ForceGraph3D = FG;\n'
        '      window.THREE = threeMod;\n'
        '      return { ForceGraph3D: FG, THREE: threeMod };\n'
        '    })();\n'
        '    window.__force3dPromise.catch(function(err) {\n'
        '      window.__force3dError = (err && err.message) ? err.message : String(err);\n'
        '    });\n'
        '  </script>\n'
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
      max-width: min(280px, 65%);
    }}
    #legend .legend-title {{ display: block; font-weight: 600; font-size: 11px;
      color: #555; letter-spacing: 0.02em; text-transform: uppercase;
      margin-bottom: 6px; }}
    #legend .legend-section + .legend-section {{
      margin-top: 8px; padding-top: 8px;
      border-top: 1px dashed rgba(0,0,0,0.12);
    }}
    #legend .legend-row {{ display: flex; align-items: center; gap: 8px;
      margin: 3px 0; }}
    #legend .legend-dot {{ display: inline-block; width: 12px; height: 12px;
      border-radius: 50%; border: 1px solid rgba(0,0,0,0.18); flex-shrink: 0; }}
    #legend .legend-label {{ color: #222; }}
    #legend .legend-glyph {{ flex-shrink: 0; display: inline-block;
      line-height: 0; }}
    #legend .legend-glyph svg {{ display: block; }}
    #metrics-hud {{
      display: none; position: absolute; top: 10px; left: 10px; z-index: 5;
      background: rgba(255, 255, 255, 0.92); border: 1px solid #d0d0d0;
      border-radius: 6px; padding: 8px 10px; font-size: 12px; line-height: 1.4;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08); pointer-events: none;
      max-width: min(240px, 60%);
    }}
    #metrics-hud .hud-title {{ display: block; font-weight: 600; font-size: 11px;
      color: #555; letter-spacing: 0.02em; text-transform: uppercase;
      margin-bottom: 6px; }}
    #metrics-hud .hud-row {{ display: flex; justify-content: space-between;
      gap: 12px; margin: 2px 0; }}
    #metrics-hud .hud-label {{ color: #555; }}
    #metrics-hud .hud-value {{ color: #111; font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body>
  <header>
    <h1 id="hdr-title"></h1>
    <p id="hdr-sub"></p>
  </header>
  <div id="err"></div>
  <div id="chart-wrap">
    <div id="metrics-hud" aria-label="Displayed graph metrics"></div>
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
    function metricLines(obj) {
      if (!obj) return "";
      var parts = [];
      Object.keys(obj).forEach(function(k) { parts.push(k + ": " + obj[k]); });
      return parts.length ? ("\\n" + parts.join("\\n")) : "";
    }
    function nodeTooltip(d) {
      return (d.label || d.id) + " — degree " + (d.degree || 0) + metricLines(d.metrics);
    }
    function edgeTooltip(l) {
      var s = (l.source && l.source.id) ? l.source.id : l.source;
      var t = (l.target && l.target.id) ? l.target.id : l.target;
      return s + " — " + t + metricLines(l.metrics);
    }
    try {
      const raw = document.getElementById("graph-payload").textContent;
      const payload = JSON.parse(raw);
      document.getElementById("hdr-title").textContent = payload.title || "Graph";
      document.getElementById("hdr-sub").textContent = payload.subtitle || "";

      (function renderLegend() {
        var box = document.getElementById("legend");
        if (!box) return;
        var rankEntries = (payload.legend || []).filter(function(e) { return e && e.color; });
        var relEntries = (payload.relationsLegend || []).filter(function(e) { return !!e; });
        if (rankEntries.length === 0 && relEntries.length === 0) {
          box.innerHTML = "";
          box.style.display = "none";
          return;
        }

        function escAttr(s) {
          return String(s == null ? "" : s).replace(/"/g, "&quot;");
        }
        function escText(s) {
          return String(s == null ? "" : s)
            .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }

        // Build a compact two-node + edge glyph that mirrors the actual
        // plot: gradient + directional arrow for incidence relations,
        // solid colour line for adjacency relations. The source endpoint
        // always sits on the left and the target on the right, so the
        // arrow naturally points right -- "up" incidences run lower ->
        // higher rank, "down" incidences run higher -> lower.
        function relationGlyphHtml(rel, idx) {
          var W = 110, H = 18, R = 5, padX = 6;
          var x1 = padX + R;
          var x2 = W - padX - R;
          var midY = H / 2;
          var defsParts = [];
          var lineExtra = '';
          var lineX2 = x2;
          var solidMode = (payload.neighborhoodColorMode === "unique_solid")
            || !(rel.colorStart && rel.colorEnd);
          if (rel.kind === "incidence" && !solidMode) {
            var gradId = "lglegrad-" + idx;
            var arrowId = "lglegarrow-" + idx;
            defsParts.push(
              '<linearGradient id="' + gradId + '"'
              + ' x1="' + x1 + '" y1="' + midY
              + '" x2="' + x2 + '" y2="' + midY
              + '" gradientUnits="userSpaceOnUse">'
              + '<stop offset="0%" stop-color="' + escAttr(rel.colorStart) + '"/>'
              + '<stop offset="100%" stop-color="' + escAttr(rel.colorEnd) + '"/>'
              + '</linearGradient>'
            );
            defsParts.push(
              '<marker id="' + arrowId + '" viewBox="0 -5 10 10" refX="9"'
              + ' refY="0" markerWidth="7" markerHeight="7" orient="auto">'
              + '<path d="M0,-4L8,0L0,4Z" fill="' + escAttr(rel.colorEnd)
              + '" opacity="0.92"/></marker>'
            );
            lineExtra = ' stroke="url(#' + gradId + ')"'
              + ' marker-end="url(#' + arrowId + ')"';
            // Pull the line back slightly so the arrow tip lands just at
            // the target circle border, not buried inside it.
            lineX2 = x2 - 1;
          } else {
            var strokeColor = rel.color || "#888888";
            lineExtra = ' stroke="' + escAttr(strokeColor) + '"';
            if (rel.kind === "incidence") {
              var arrowIdSolid = "lglegarrow-solid-" + idx;
              defsParts.push(
                '<marker id="' + arrowIdSolid + '" viewBox="0 -5 10 10" refX="9"'
                + ' refY="0" markerWidth="7" markerHeight="7" orient="auto">'
                + '<path d="M0,-4L8,0L0,4Z" fill="' + escAttr(strokeColor)
                + '" opacity="0.92"/></marker>'
              );
              lineExtra += ' marker-end="url(#' + arrowIdSolid + ')"';
              lineX2 = x2 - 1;
            }
          }
          return '<svg width="' + W + '" height="' + H
            + '" viewBox="0 0 ' + W + ' ' + H + '">'
            + (defsParts.length ? '<defs>' + defsParts.join('') + '</defs>' : '')
            + '<line x1="' + x1 + '" y1="' + midY
            + '" x2="' + lineX2 + '" y2="' + midY
            + '" stroke-width="1.6"' + lineExtra + '/>'
            + '<circle cx="' + x1 + '" cy="' + midY + '" r="' + (R - 0.5)
            + '" fill="' + escAttr(rel.srcColor) + '" stroke="#fff"'
            + ' stroke-width="1"/>'
            + '<circle cx="' + x2 + '" cy="' + midY + '" r="' + (R - 0.5)
            + '" fill="' + escAttr(rel.tgtColor) + '" stroke="#fff"'
            + ' stroke-width="1"/>'
            + '</svg>';
        }

        var html = '';
        if (rankEntries.length) {
          html += '<div class="legend-section">'
            + '<span class="legend-title">Ranks</span>';
          rankEntries.forEach(function(e) {
            var label = e.label || ("Rank " + e.rank);
            html += '<div class="legend-row">'
              + '<span class="legend-dot" style="background:'
              + escAttr(e.color) + '"></span>'
              + '<span class="legend-label">' + escText(label) + '</span>'
              + '</div>';
          });
          html += '</div>';
        }
        if (relEntries.length) {
          html += '<div class="legend-section">'
            + '<span class="legend-title">Neighborhoods</span>';
          relEntries.forEach(function(rel, i) {
            html += '<div class="legend-row">'
              + '<span class="legend-glyph">'
              + relationGlyphHtml(rel, i)
              + '</span>'
              + '<span class="legend-label">' + escText(rel.label || "")
              + '</span>'
              + '</div>';
          });
          html += '</div>';
        }
        box.innerHTML = html;
        box.style.display = "block";
      })();

      (function renderMetricsHud() {
        var box = document.getElementById("metrics-hud");
        if (!box) return;
        var rows = payload.graphMetrics || [];
        if (!payload.showMetricsHud || rows.length === 0) {
          box.innerHTML = "";
          box.style.display = "none";
          return;
        }
        var html = '<span class="hud-title">Displayed graph</span>';
        rows.forEach(function(r) {
          html += '<div class="hud-row">'
            + '<span class="hud-label">' + r.label + '</span>'
            + '<span class="hud-value">' + r.value + '</span>'
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

      // Multiple adjacencies between the same node pair (e.g.
      // ``1-up_adjacency-0`` and ``2-up_adjacency-0`` both connecting
      // the same two nodes) used to be merged into a single edge with
      // a blended colour. Python now emits one payload link per
      // via-rank; here we tag duplicate-endpoint links with a
      // symmetric "slot" so both 2D and 3D pipelines can offset them
      // perpendicular to the line and render them as visible
      // parallel edges. Done once on ``payload.links`` so both the
      // 2D and 3D map() copies below pick up the same slot values.
      (function assignParallelSlots() {
        const groups = new Map();
        (payload.links || []).forEach(function(l) {
          const a = String(l.source);
          const b = String(l.target);
          const key = (a < b) ? (a + "\u0001" + b) : (b + "\u0001" + a);
          let arr = groups.get(key);
          if (!arr) { arr = []; groups.set(key, arr); }
          arr.push(l);
        });
        groups.forEach(function(arr) {
          const n = arr.length;
          arr.forEach(function(l, i) {
            l._parallelCount = n;
            l._parallelSlot = (n > 1) ? (i - (n - 1) / 2) : 0;
          });
        });
      })();
      // Per-slot perpendicular gap, sized so parallel edges sit flush
      // against each other (reading like one "double-thick" multicoloured
      // stroke) rather than as visually separated rails. The gap equals
      // the on-screen stroke width: 1.8 px in 2D (matches the ``<line>``
      // stroke-width) and ``2 * LINK_RADIUS`` = 1.0 world unit in 3D
      // (matches the cylinder diameter).
      const PARALLEL_GAP_2D = 1.8;
      const PARALLEL_GAP_3D = 1.0;

      if (payload.graphType === "layered3d") {
        // ``3d-force-graph`` is now ESM-only, so it's loaded async via an
        // import() in a separate module script that resolves
        // ``window.__force3dPromise`` to ``{ ForceGraph3D, THREE }``. We
        // defer 3D scene construction until that promise resolves.
        function build3DScene() {
          if (typeof ForceGraph3D === "undefined") {
            showError("3d-force-graph failed to load from CDN. Connect to the internet or use a recent browser.");
            return;
          }
          _build3D();
        }
        if (window.__force3dPromise) {
          window.__force3dPromise.then(build3DScene).catch(function(err) {
            showError(
              "3D libraries failed to load from CDN: "
              + (err && err.message ? err.message : String(err))
              + ". Connect to the internet or open this file in a recent browser."
            );
          });
        } else if (window.__force3dError) {
          showError(
            "3D libraries failed to load from CDN: "
            + window.__force3dError
            + ". Connect to the internet or open this file in a recent browser."
          );
        } else {
          // Fallback: the module loader hasn't installed the promise yet.
          // Poll briefly, then surface the global if it appeared, or report.
          var attempts = 0;
          var poll = setInterval(function() {
            attempts += 1;
            if (window.__force3dPromise) {
              clearInterval(poll);
              window.__force3dPromise.then(build3DScene).catch(function(err) {
                showError(
                  "3D libraries failed to load from CDN: "
                  + (err && err.message ? err.message : String(err))
                );
              });
            } else if (attempts > 200) {  // ~10s
              clearInterval(poll);
              showError("3d-force-graph failed to load from CDN. Connect to the internet or open this file in a recent browser.");
            }
          }, 50);
        }
        return;
      }

      function _build3D() {
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
            color: l.color,
            colorStart: l.colorStart,
            colorEnd: l.colorEnd,
            kind: l.kind,
            metrics: l.metrics,
            _parallelSlot: l._parallelSlot || 0,
            _parallelCount: l._parallelCount || 1
          };
        });

        // ``THREE`` is loaded from CDN ahead of ``3d-force-graph``. When
        // available we render each link as a custom ``THREE.Line`` with a
        // two-stop vertex-color gradient (start at source, end at target),
        // mirroring the 2D SVG ``<linearGradient>`` look. Otherwise we fall
        // back to a flat per-link color so the view still renders.
        const hasTHREE = (typeof THREE !== "undefined");
        const graph3d = ForceGraph3D()(chart)
          .width(width)
          .height(height)
          .backgroundColor("#fafafa")
          .nodeRelSize(4)
          .nodeColor(function(n) { return n.color || "#666"; })
          .nodeLabel(function(n) { return nodeTooltip(n); })
          .nodeVal(function(n) { return 1 + Math.log1p(n.degree || 1); })
          .linkLabel(function(l) { return edgeTooltip(l); })
          // Directional arrows on incidence (cross-rank) links only.
          // Adjacency edges live within a single rank and remain undirected.
          // ``3d-force-graph`` builds these as ``THREE.Mesh`` cones whose
          // *base* width scales with ``linkDirectionalArrowLength``, so a
          // larger length produces a chunkier head as well as a longer one.
          // ``linkDirectionalArrowRelPos`` is the parametric position along
          // the segment (0 = at source, 1 = at target); ``0.97`` puts the
          // cone tip just shy of the destination node surface.
          .linkDirectionalArrowLength(function(l) {
            return l.kind === "incidence" ? 6.0 : 0;
          })
          .linkDirectionalArrowRelPos(0.99)
          .linkDirectionalArrowColor(function(l) {
            return l.colorEnd || l.color || "#888888";
          });

        if (hasTHREE) {
          // ``THREE.Line`` / ``LineBasicMaterial`` is hard-capped to a 1-pixel
          // stroke on virtually every WebGL platform, so it can never match
          // the 2D SVG line thickness. Instead we represent each link as a
          // thin cylinder ``THREE.Mesh`` with vertex-coloured gradients along
          // its length, then orient and stretch it from source to target
          // every frame. Cylinder mesh is built as a unit-length tube along
          // local +Y; the base sits at the origin and the cap at (0, 1, 0).
          const LINK_RADIUS = 0.5;       // tube radius (similar to 2D ~1.2 px stroke)
          const RADIAL_SEGMENTS = 8;     // smooth enough at typical view scales
          graph3d
            .linkThreeObject(function(l) {
              var startStr = l.colorStart || l.color || "#888888";
              var endStr = l.colorEnd || l.color || startStr;
              var cStart = new THREE.Color(startStr);
              var cEnd = new THREE.Color(endStr);
              // Open-ended cylinder of unit height along +Y, then translate
              // upward by 0.5 so its base sits at (0, 0, 0).
              var geom = new THREE.CylinderGeometry(
                LINK_RADIUS, LINK_RADIUS, 1, RADIAL_SEGMENTS, 1, true
              );
              geom.translate(0, 0.5, 0);
              var pos = geom.attributes.position;
              var colors = new Float32Array(pos.count * 3);
              var tmp = new THREE.Color();
              for (var i = 0; i < pos.count; i++) {
                // y in [0, 1] post-translate; lerp from start colour at
                // the base to end colour at the cap.
                var t = Math.min(1, Math.max(0, pos.getY(i)));
                tmp.copy(cStart).lerp(cEnd, t);
                colors[i * 3 + 0] = tmp.r;
                colors[i * 3 + 1] = tmp.g;
                colors[i * 3 + 2] = tmp.b;
              }
              geom.setAttribute(
                "color",
                new THREE.BufferAttribute(colors, 3)
              );
              var mat = new THREE.MeshBasicMaterial({
                vertexColors: true,
                transparent: true,
                opacity: 0.85
              });
              var mesh = new THREE.Mesh(geom, mat);
              // Defensive: skip the default ``Object3D`` raycasting (not
              // needed and would otherwise consume tooltip events).
              mesh.raycast = function() {};
              // Stash the parallel-slot info on the mesh itself so the
              // per-frame ``linkPositionUpdate`` can apply the
              // perpendicular offset without depending on the 3rd
              // argument signature of every 3d-force-graph build.
              mesh.userData._parallelSlot = (l && l._parallelSlot) || 0;
              mesh.userData._parallelCount = (l && l._parallelCount) || 1;
              return mesh;
            })
            .linkPositionUpdate(function(mesh, posObj, link) {
              if (!mesh) return false;
              var sx = posObj.start.x, sy = posObj.start.y, sz = posObj.start.z;
              var ex = posObj.end.x,   ey = posObj.end.y,   ez = posObj.end.z;
              var dx = ex - sx, dy = ey - sy, dz = ez - sz;
              var len = Math.sqrt(dx * dx + dy * dy + dz * dz);
              // Apply the same perpendicular-slot offset as 2D so that
              // duplicate-endpoint links (parallel adjacencies) render as
              // visibly separate cylinders instead of overlapping cores.
              var slot = (mesh.userData && mesh.userData._parallelSlot) || 0;
              if (slot === 0 && link && link._parallelSlot) {
                slot = link._parallelSlot;
              }
              var ox = 0, oy = 0, oz = 0;
              if (slot !== 0 && len > 1e-6) {
                // Build a unit vector perpendicular to the edge. Cross with
                // world-up first; if the edge is nearly vertical, fall back
                // to world-right so the basis is always well-defined.
                var dir = new THREE.Vector3(dx / len, dy / len, dz / len);
                var up = new THREE.Vector3(0, 1, 0);
                if (Math.abs(dir.dot(up)) > 0.95) {
                  up = new THREE.Vector3(1, 0, 0);
                }
                var perp = new THREE.Vector3().crossVectors(dir, up).normalize();
                var off = slot * PARALLEL_GAP_3D;
                ox = perp.x * off; oy = perp.y * off; oz = perp.z * off;
              }
              mesh.position.set(sx + ox, sy + oy, sz + oz);
              // Stretch along local +Y to match the actual edge length.
              mesh.scale.set(1, Math.max(len, 1e-6), 1);
              if (len > 1e-6) {
                var yAxis = new THREE.Vector3(0, 1, 0);
                var dir2 = new THREE.Vector3(dx / len, dy / len, dz / len);
                mesh.quaternion.setFromUnitVectors(yAxis, dir2);
              }
              // Returning ``true`` tells 3d-force-graph that we've handled
              // the per-frame position update for this link.
              return true;
            });
        } else {
          graph3d
            .linkColor(function(l) { return l.color || "#888888"; })
            .linkOpacity(0.85)
            .linkWidth(1.4);
        }

        graph3d.graphData({ nodes: nodes3d, links: links3d });
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
      const solidNeighborhoodColors = (payload.neighborhoodColorMode === "unique_solid");
      const links = (payload.links || []).map(function(l, idx) {
        const s = nodeById.get(String(l.source));
        const t = nodeById.get(String(l.target));
        return {
          source: s,
          target: t,
          color: l.color,
          colorStart: l.colorStart,
          colorEnd: l.colorEnd,
          kind: l.kind,
          metrics: l.metrics,
          _parallelSlot: l._parallelSlot || 0,
          _parallelCount: l._parallelCount || 1,
          _gradId: (l.colorStart && l.colorEnd && !solidNeighborhoodColors)
            ? ("grad-link-" + idx) : null
        };
      }).filter(function(l) { return l.source && l.target; });

      // Arrowheads for incidence (cross-rank) links only. Adjacency edges
      // live within a single rank and stay undirected. Each unique
      // end-color gets its own ``<marker>`` so the arrow tip naturally
      // matches whatever the link's ``colorEnd`` is, even when the line
      // itself is drawn with a gradient stroke.
      function arrowColorIdSuffix(c) {
        return String(c).replace(/[^a-zA-Z0-9]/g, "");
      }
      function linkArrowColor(l) {
        return l.colorEnd || l.color || "#444";
      }

      const chart = document.getElementById("chart");
      const width = Math.max(420, chart.clientWidth || window.innerWidth || 800);
      const height = Math.max(420, chart.clientHeight || (window.innerHeight - 100) || 600);

      const svg = d3.select("#chart").append("svg")
        .attr("width", width).attr("height", height)
        .attr("viewBox", [0, 0, width, height]);

      // SVG <defs>: per-link color gradients (rank-to-rank edges) and one
      // reusable arrowhead marker per distinct incidence color, referenced by
      // links via marker-end.
      const defs = svg.append("defs");
      const gradientLinks = links.filter(function(l) { return !!l._gradId; });
      const gradients = defs.selectAll("linearGradient")
        .data(gradientLinks)
        .join("linearGradient")
        .attr("id", function(d) { return d._gradId; })
        .attr("gradientUnits", "userSpaceOnUse");
      gradients.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", function(d) { return d.colorStart || d.color || "#888"; });
      gradients.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", function(d) { return d.colorEnd || d.color || "#888"; });

      const arrowColorSet = new Set();
      links.forEach(function(l) {
        if (l.kind === "incidence") arrowColorSet.add(linkArrowColor(l));
      });
      const arrowColors = Array.from(arrowColorSet);
      defs.selectAll("marker.arrow")
        .data(arrowColors)
        .join("marker")
        .attr("class", "arrow")
        .attr("id", function(c) { return "arrow-" + arrowColorIdSuffix(c); })
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 9)
        .attr("refY", 0)
        .attr("markerUnits", "userSpaceOnUse")
        .attr("markerWidth", 8)
        .attr("markerHeight", 8)
        .attr("orient", "auto")
        .append("path")
          .attr("d", "M0,-4L8,0L0,4Z")
          .attr("fill", function(c) { return c; })
          .attr("opacity", 0.92);

      const g = svg.append("g");
      const zoom = d3.zoom()
        .scaleExtent([0.12, 6])
        .on("zoom", function(ev) { g.attr("transform", ev.transform); });
      svg.call(zoom);

      // Force-directed layout. Links pull adjacent cells to a ~48px rest
      // distance; charge pushes all nodes apart (weaker for bipartite views to
      // keep the two layers compact); collision scales the exclusion radius
      // with node degree so busy hubs do not overlap.
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
            .attr("stroke-width", 1.8);
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
        .attr("stroke", function(d) {
          return d._gradId ? ("url(#" + d._gradId + ")") : (d.color || "#888");
        })
        .attr("stroke-width", 1.8)
        .attr("marker-end", function(d) {
          if (d.kind !== "incidence") return null;
          return "url(#arrow-" + arrowColorIdSuffix(linkArrowColor(d)) + ")";
        });

      link.append("title").text(function(d) { return edgeTooltip(d); });

      function nodeRadius(d) {
        return 5 + Math.min(16, Math.log1p((d && d.degree) || 1) * 3.2);
      }
      function updateLinkGeometry(d) {
        var sx = d.source.x, sy = d.source.y;
        var tx = d.target.x, ty = d.target.y;
        var dx = tx - sx, dy = ty - sy;
        var len = Math.sqrt(dx * dx + dy * dy);
        // Perpendicular offset so duplicate-endpoint links (mostly
        // adjacencies sharing a node pair across different via-ranks)
        // render as parallel lines instead of pixel-perfect overlaps.
        var ox = 0, oy = 0;
        var slot = d._parallelSlot || 0;
        if (slot !== 0 && isFinite(len) && len > 1e-3) {
          var nx = -dy / len, ny = dx / len;
          var off = slot * PARALLEL_GAP_2D;
          ox = nx * off;
          oy = ny * off;
        }
        d._x1 = sx + ox;
        d._y1 = sy + oy;
        if (d.kind === "incidence" && isFinite(len) && len > 1e-3) {
          // Pull the visible endpoint back to the node border plus a small
          // clearance so the marker tip is fully visible outside the circle.
          var r = nodeRadius(d.target) + 3.5;
          var k = Math.max(0, (len - r) / len);
          d._x2 = sx + dx * k + ox;
          d._y2 = sy + dy * k + oy;
        } else {
          d._x2 = tx + ox;
          d._y2 = ty + oy;
        }
      }

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

      node.append("title").text(function(d) { return nodeTooltip(d); });

      node.filter(function(d) { return String(d.id).length <= 12; })
        .append("text")
        .attr("dx", 10)
        .attr("dy", 4)
        .text(function(d) { return d.id; });

      simulation.on("tick", function() {
        link
          .each(updateLinkGeometry)
          .attr("x1", function(d) { return d._x1; })
          .attr("y1", function(d) { return d._y1; })
          .attr("x2", function(d) { return d._x2; })
          .attr("y2", function(d) { return d._y2; });
        gradients
          .attr("x1", function(d) { return d._x1; })
          .attr("y1", function(d) { return d._y1; })
          .attr("x2", function(d) { return d._x2; })
          .attr("y2", function(d) { return d._y2; });
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