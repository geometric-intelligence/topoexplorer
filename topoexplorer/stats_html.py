"""Standalone HTML export of TopoExplorer graph statistics.

The metrics-side counterpart to :mod:`d3_graph_html`: given the metric tables
already computed for the current view, produce a single self-contained HTML page
(no external scripts or assets) suitable for reports, slides and paper
appendices. It reuses the shared TopoBench branding so it matches the app and the
visualization export.
"""

import html as html_module
from typing import Any

import branding


def _esc(value: Any) -> str:
    """HTML-escape a value, rendering ``None`` as an empty string."""
    return html_module.escape("" if value is None else str(value))


def _table_html(columns: list, rows: list[list]) -> str:
    """Render a list of rows as an HTML ``<table>`` with a header row."""
    head = "".join(f"<th>{_esc(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def build_stats_html(sections: list[dict], *, title: str, subtitle: str = "") -> str:
    """Build a self-contained, branded HTML page of metric tables.

    Args:
        sections: Ordered list of section dicts. Each has ``heading`` (str),
            ``columns`` (list of column labels), ``rows`` (list of row lists),
            and an optional ``note`` (str) shown under the heading. Sections with
            no rows are skipped.
        title: Page/header title (typically the current view label).
        subtitle: Optional secondary line shown under the title.

    Returns:
        str: A complete HTML document string that renders fully offline.
    """
    b = branding.BRAND
    parts: list[str] = []
    for sec in sections:
        rows = sec.get("rows") or []
        if not rows:
            continue
        parts.append(f'<section><h2>{_esc(sec.get("heading", ""))}</h2>')
        if sec.get("note"):
            parts.append(f'<p class="note">{_esc(sec["note"])}</p>')
        parts.append(_table_html(sec.get("columns") or [], rows))
        parts.append("</section>")
    body_sections = "\n".join(parts) or '<p class="empty">No metrics available.</p>'

    header = branding.header_html(_esc(title), _esc(subtitle))
    footer = branding.footer_html()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{_esc(title)} &middot; metrics</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: {b["bg_page"]}; color: {b["ink"]};
      font-family: system-ui, Segoe UI, sans-serif; }}
    main {{ max-width: 900px; margin: 0 auto; padding: 24px 20px 40px; }}
    section {{ margin: 22px 0; }}
    h2 {{ font-size: 1rem; margin: 0 0 8px; color: {b["primary_deep"]};
      border-left: 4px solid {b["primary"]}; padding-left: 10px; }}
    p.note {{ margin: 0 0 8px; color: {b["muted"]}; font-size: 0.82rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th, td {{ text-align: left; padding: 7px 12px;
      border-bottom: 1px solid {b["border"]}; }}
    thead th {{ background: {b["bg_soft"]}; color: {b["ink"]}; font-weight: 600;
      border-bottom: 2px solid {b["primary"]}; }}
    tbody tr:nth-child(even) {{ background: #faf7fb; }}
    td {{ font-variant-numeric: tabular-nums; }}
    p.empty {{ color: {b["muted"]}; padding: 24px 20px; }}
  </style>
</head>
<body>
{header}
<main>
{body_sections}
</main>
{footer}
</body>
</html>"""
