"""Shared TopoBench branding for TopoExplorer.

Centralises the TopoBench brand palette and small HTML header/footer snippets so
the Streamlit app, the standalone visualization export and the standalone
statistics export share a single, consistent visual identity. Colours are taken
from the TopoBench logo (the magenta "Bench" wordmark) and its documentation
theme. No image asset is used; branding is colour + text only.
"""

# TopoBench brand palette. ``primary`` is the magenta of the "Bench" wordmark;
# the rest are supporting neutrals/tints chosen to read as the same identity.
BRAND = {
    "primary": "#c13ba8",       # TopoBench magenta (wordmark accent)
    "primary_deep": "#8e2a80",  # darker magenta for borders / hover
    "accent": "#e85fd0",        # lighter magenta/pink for subtle highlights
    "ink": "#2c2a3a",           # near-black body text
    "muted": "#8a8f98",         # grey (the "Topo" wordmark tone)
    "border": "#e6d7e8",        # light magenta-tinted border
    "bg_soft": "#fbf6fb",       # very light magenta tint for panel backgrounds
    "bg_page": "#ffffff",       # clean white page background (paper-ready)
}


def header_html(title: str, subtitle: str = "") -> str:
    """Build a branded, text-only HTML header (title + "built on TopoBench").

    Uses inline styles only, so it renders correctly when dropped into any
    self-contained export without relying on external CSS.

    Args:
        title: Main heading text (already HTML-escaped by the caller).
        subtitle: Optional secondary line under the title.

    Returns:
        str: An HTML ``<header>`` fragment.
    """
    sub = (
        f'<p style="margin:4px 0 0;color:{BRAND["muted"]};font-size:0.85rem;">'
        f"{subtitle}</p>"
        if subtitle
        else ""
    )
    return (
        f'<header style="display:flex;align-items:center;gap:16px;'
        f"padding:14px 20px;background:#fff;"
        f'border-bottom:3px solid {BRAND["primary"]};">'
        f'<div style="flex:1 1 auto;min-width:0;">'
        f'<h1 style="margin:0;font-size:1.15rem;color:{BRAND["ink"]};'
        f'font-family:Georgia,\'Times New Roman\',serif;">{title}</h1>'
        f"{sub}"
        f"</div>"
        f'<div style="flex:0 0 auto;color:{BRAND["muted"]};font-size:0.75rem;'
        f'text-align:right;">built on '
        f'<span style="color:{BRAND["primary"]};font-weight:600;">TopoBench</span>'
        f"</div>"
        f"</header>"
    )


def footer_html() -> str:
    """Build a small branded footer credit for exported HTML files.

    Returns:
        str: An HTML ``<footer>`` fragment with a "built on TopoBench" credit.
    """
    return (
        f'<footer style="padding:8px 20px;text-align:right;'
        f'color:{BRAND["muted"]};font-size:0.72rem;'
        f'border-top:1px solid {BRAND["border"]};background:{BRAND["bg_soft"]};">'
        f"Generated with TopoExplorer &middot; built on "
        f'<span style="color:{BRAND["primary"]};font-weight:600;">TopoBench</span>'
        f"</footer>"
    )
