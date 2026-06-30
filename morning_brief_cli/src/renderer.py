"""Render BriefData to Markdown and self-contained HTML.

Output format is controlled by the caller; this module is purely functional —
no file I/O, no side effects. The ``write_brief`` helper handles persistence.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from src.brief import BriefData, Source

# ──────────────────────────────────────────────── shared helpers ──────────────


def _company_heading(ticker: str, name: str) -> str:
    if ticker:
        return f"{ticker} — {name}"
    return name


def _format_citations_html(text: str) -> str:
    """Replace [N] markers in already-escaped text with clickable superscript links."""
    return re.sub(
        r"\[(\d+)\]",
        r'<sup><a href="#src-\1" class="cite-link">[\1]</a></sup>',
        text,
    )


# ──────────────────────────────────────────────── markdown ────────────────────


def render_markdown(brief: BriefData) -> str:
    """Return the full Markdown text for the brief."""
    lines: list[str] = []

    lines.append(f"# Morning Brief — {brief.brief_date}")
    lines.append(
        f"_Portfolio: {len(brief.companies)} companies"
        f" | Topics: {len(brief.topics)}"
        f" | Generated: {brief.generated_at}_"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    for cb in brief.companies:
        company = cb.company
        heading = _company_heading(company.ticker, company.name)
        lines.append(f"## {heading}")
        lines.append("")

        has_content = False
        for topic in brief.topics:
            section = cb.sections.get(topic.id)
            if section and section.summary.strip():
                lines.append(f"**{topic.label}:** {section.summary}")
                has_content = True

        if not has_content:
            lines.append("_No material developments identified for this period._")

        lines.append("")
        lines.append("---")
        lines.append("")

    if brief.sources:
        lines.append("## Sources")
        lines.append("")
        for source in brief.sources:
            headline = source.headline or source.url
            date_str = f" ({source.timestamp[:10]})" if source.timestamp else ""
            url_str = f" — {source.url}" if source.url else ""
            lines.append(f"[{source.index}] {headline}{date_str}{url_str}")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────── HTML ───────────────────────


def _css() -> str:
    return """
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;
         background:#f0f2f5;color:#1a1a2e;line-height:1.65;font-size:15px}
    /* ── header ── */
    .mb-header{background:#1a1a2e;color:#fff;padding:22px 36px 18px}
    .mb-header__title{font-size:24px;font-weight:700;letter-spacing:-0.3px}
    .mb-header__meta{font-size:13px;color:#8fa8c0;margin-top:5px}
    /* ── nav bar ── */
    .mb-nav{background:#fff;border-bottom:1px solid #dde3ea;padding:10px 36px;
            position:sticky;top:0;z-index:200;overflow-x:auto;white-space:nowrap;
            box-shadow:0 1px 4px rgba(0,0,0,.06)}
    .mb-nav__chip{display:inline-block;background:#f0f4f8;border:1px solid #dde3ea;
                  border-radius:6px;padding:5px 12px;margin-right:8px;font-size:13px;
                  font-weight:600;color:#1a1a2e;text-decoration:none;transition:background .15s}
    .mb-nav__chip:hover{background:#dde3ea}
    /* ── main layout ── */
    .mb-main{max-width:900px;margin:28px auto;padding:0 16px 40px}
    /* ── company card ── */
    .mb-card{background:#fff;border:1px solid #dde3ea;border-radius:12px;
             margin-bottom:20px;overflow:hidden;
             box-shadow:0 1px 3px rgba(0,0,0,.04)}
    .mb-card__header{background:#f7f9fb;border-bottom:1px solid #dde3ea;
                     padding:14px 24px;display:flex;align-items:baseline;gap:10px;
                     cursor:pointer;user-select:none}
    .mb-card__ticker{font-size:19px;font-weight:700;color:#1a1a2e}
    .mb-card__name{font-size:15px;color:#5e7087;font-weight:400}
    .mb-card__toggle{margin-left:auto;font-size:18px;color:#8fa8c0;transition:transform .2s}
    .mb-card__toggle.collapsed{transform:rotate(-90deg)}
    .mb-card__body{padding:20px 24px}
    /* ── topic sections ── */
    .mb-topic{margin-bottom:14px}
    .mb-topic:last-child{margin-bottom:0}
    .mb-topic__label{font-size:11px;font-weight:700;color:#5e7087;text-transform:uppercase;
                      letter-spacing:.7px;margin-bottom:5px}
    .mb-topic__text{font-size:14px;color:#2c3a4a;line-height:1.7}
    .mb-no-content{font-style:italic;color:#8fa8c0;font-size:14px}
    /* ── citations ── */
    .cite-link{color:#0066cc;text-decoration:none;font-size:10px;
               vertical-align:super;line-height:0}
    .cite-link:hover{text-decoration:underline}
    /* ── sources section ── */
    .mb-sources{max-width:900px;margin:0 auto 50px;padding:0 16px}
    .mb-sources__title{font-size:17px;font-weight:700;color:#1a1a2e;
                        padding:18px 0 12px;border-top:2px solid #dde3ea;margin-bottom:4px}
    .mb-sources__list{list-style:none}
    .mb-sources__item{font-size:13px;color:#2c3a4a;padding:7px 0;
                       border-bottom:1px solid #f0f2f5;display:flex;gap:8px}
    .mb-sources__item:last-child{border-bottom:none}
    .mb-sources__num{font-weight:700;color:#1a1a2e;white-space:nowrap;min-width:28px}
    .mb-sources__link{color:#0066cc;text-decoration:none;word-break:break-all}
    .mb-sources__link:hover{text-decoration:underline}
    .mb-sources__meta{color:#8fa8c0;margin-left:4px;white-space:nowrap}
    /* ── responsive ── */
    @media(max-width:640px){
      .mb-header{padding:16px 16px 14px}
      .mb-nav{padding:10px 16px}
      .mb-card__body{padding:16px}
      .mb-main{padding:0 8px 30px}
    }
"""


def _js() -> str:
    return """
(function(){
  // Smooth-scroll for all in-page anchor links
  document.querySelectorAll('a[href^="#"]').forEach(function(a){
    a.addEventListener('click',function(e){
      var t=document.querySelector(this.getAttribute('href'));
      if(t){e.preventDefault();t.scrollIntoView({behavior:'smooth',block:'start'});}
    });
  });

  // Collapse/expand company cards
  document.querySelectorAll('.mb-card__header').forEach(function(h){
    h.addEventListener('click',function(){
      var body=this.nextElementSibling;
      var icon=this.querySelector('.mb-card__toggle');
      var open=body.style.display!=='none';
      body.style.display=open?'none':'block';
      if(icon) icon.classList.toggle('collapsed',open);
    });
  });
})();
"""


def _source_item_html(source: Source, e: Any = html.escape) -> str:
    headline = source.headline or source.url or f"Source {source.index}"
    ts = source.timestamp[:10] if source.timestamp else ""
    meta_parts = [p for p in [source.source_name, ts] if p]
    meta_str = (
        f' <span class="mb-sources__meta">({", ".join(meta_parts)})</span>'
        if meta_parts
        else ""
    )
    if source.url:
        content = f'<a href="{e(source.url)}" class="mb-sources__link">{e(headline)}</a>{meta_str}'
    else:
        content = f'<span class="mb-sources__link">{e(headline)}</span>{meta_str}'

    return (
        f'<li id="src-{source.index}" class="mb-sources__item">'
        f'<span class="mb-sources__num">[{source.index}]</span>'
        f"{content}</li>"
    )


# re-export html.escape type for the helper above
from typing import Any  # noqa: E402


def render_html(brief: BriefData) -> str:
    """Return a self-contained HTML document for the brief."""
    e = html.escape
    parts: list[str] = []

    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="UTF-8">')
    parts.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    parts.append(f"<title>Morning Brief — {e(brief.brief_date)}</title>")
    parts.append(f"<style>{_css()}</style>")
    parts.append("</head>")
    parts.append("<body>")

    # ── header ──────────────────────────────────────────────────────────────
    parts.append('<header class="mb-header">')
    parts.append(f'<div class="mb-header__title">Morning Brief — {e(brief.brief_date)}</div>')
    parts.append(
        f'<div class="mb-header__meta">'
        f'Portfolio: {len(brief.companies)} companies'
        f" &nbsp;|&nbsp; Topics: {len(brief.topics)}"
        f" &nbsp;|&nbsp; Generated: {e(brief.generated_at)}"
        f"</div>"
    )
    parts.append("</header>")

    # ── navigation ──────────────────────────────────────────────────────────
    parts.append('<nav class="mb-nav">')
    for cb in brief.companies:
        company = cb.company
        display = company.ticker or company.name
        anchor = company.ticker or company.company_id
        parts.append(
            f'<a href="#company-{e(anchor)}" class="mb-nav__chip">{e(display)}</a>'
        )
    parts.append("</nav>")

    # ── main content ────────────────────────────────────────────────────────
    parts.append('<main class="mb-main">')
    for cb in brief.companies:
        company = cb.company
        anchor = company.ticker or company.company_id

        parts.append(f'<div id="company-{e(anchor)}" class="mb-card">')
        parts.append('<div class="mb-card__header">')
        if company.ticker:
            parts.append(f'<span class="mb-card__ticker">{e(company.ticker)}</span>')
        parts.append(f'<span class="mb-card__name">{e(company.name)}</span>')
        parts.append('<span class="mb-card__toggle">&#x25BE;</span>')
        parts.append("</div>")  # header

        parts.append('<div class="mb-card__body">')
        has_content = False
        for topic in brief.topics:
            section = cb.sections.get(topic.id)
            if section and section.summary.strip():
                has_content = True
                parts.append('<div class="mb-topic">')
                parts.append(f'<div class="mb-topic__label">{e(topic.label)}</div>')
                parts.append(
                    f'<div class="mb-topic__text">'
                    f'{_format_citations_html(e(section.summary))}'
                    f"</div>"
                )
                parts.append("</div>")  # mb-topic

        if not has_content:
            parts.append(
                '<p class="mb-no-content">No material developments identified for this period.</p>'
            )

        parts.append("</div>")  # card body
        parts.append("</div>")  # mb-card

    parts.append("</main>")

    # ── sources ─────────────────────────────────────────────────────────────
    if brief.sources:
        parts.append('<section class="mb-sources">')
        parts.append('<h2 class="mb-sources__title">Sources</h2>')
        parts.append('<ul class="mb-sources__list">')
        for source in brief.sources:
            parts.append(_source_item_html(source))
        parts.append("</ul>")
        parts.append("</section>")

    parts.append(f"<script>{_js()}</script>")
    parts.append("</body>")
    parts.append("</html>")

    return "\n".join(parts)


# ──────────────────────────────────────────────── persistence ─────────────────


def write_brief(
    brief: BriefData,
    briefs_dir: Path,
    formats: list[str],
) -> dict[str, Path]:
    """Render and persist the brief in the requested formats.

    ``formats`` may contain ``"md"``, ``"html"``, or ``"both"``
    (``"both"`` expands to both formats).
    """
    briefs_dir.mkdir(parents=True, exist_ok=True)
    normalized: set[str] = set()
    for fmt in formats:
        if fmt == "both":
            normalized.update({"md", "html"})
        else:
            normalized.add(fmt)

    date_slug = brief.brief_date.replace("-", "")
    output_paths: dict[str, Path] = {}

    if "md" in normalized:
        md_path = briefs_dir / f"morning_brief_{date_slug}.md"
        md_path.write_text(render_markdown(brief), encoding="utf-8")
        output_paths["md"] = md_path

    if "html" in normalized:
        html_path = briefs_dir / f"morning_brief_{date_slug}.html"
        html_path.write_text(render_html(brief), encoding="utf-8")
        output_paths["html"] = html_path

    return output_paths
