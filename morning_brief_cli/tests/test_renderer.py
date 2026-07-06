"""Tests for src.renderer — no API calls required."""

from __future__ import annotations

from src.brief import BriefData, CompanyBrief, CompanySection, PortfolioCompany, Source
from src.renderer import render_html, render_markdown, write_brief
from src.topics import TOPICS

# ──────────────────────────────────────────────── fixtures ────────────────────

def _make_company(ticker: str, name: str, company_id: str = "") -> PortfolioCompany:
    return PortfolioCompany(company_id=company_id or ticker, name=name, ticker=ticker)


def _make_brief(
    with_sources: bool = True,
    with_sections: bool = True,
) -> BriefData:
    aapl = _make_company("AAPL", "Apple Inc.", "D8442A")
    msft = _make_company("MSFT", "Microsoft Corp.", "228D42")

    sources = (
        [
            Source(1, "https://example.com/1", "Apple Q2 Earnings", "Quartr", "2026-06-01"),
            Source(2, "https://example.com/2", "Apple Supply Chain", "Reuters", "2026-06-15"),
        ]
        if with_sources
        else []
    )

    aapl_sections = (
        {
            "earnings": CompanySection(
                topic_id="earnings",
                topic_label="Earnings & Guidance",
                summary="Revenue grew 8% year-on-year to $94 billion, beating consensus by 2%. [1]",
                cited_indices=[1],
            ),
            "supply_chain": CompanySection(
                topic_id="supply_chain",
                topic_label="Supply Chain & Ops",
                summary="Management flagged ongoing component shortages in legacy nodes. [2]",
                cited_indices=[2],
            ),
        }
        if with_sections
        else {}
    )

    return BriefData(
        brief_date="2026-06-29",
        generated_at="2026-06-29 09:00 UTC",
        companies=[
            CompanyBrief(company=aapl, sections=aapl_sections),
            CompanyBrief(company=msft, sections={}),
        ],
        sources=sources,
        topics=TOPICS,
    )


# ──────────────────────────────────────────────── markdown tests ──────────────


class TestRenderMarkdown:
    def test_header_contains_date(self) -> None:
        brief = _make_brief()
        md = render_markdown(brief)
        assert "# Morning Brief — 2026-06-29" in md

    def test_meta_line_has_counts(self) -> None:
        brief = _make_brief()
        md = render_markdown(brief)
        assert "2 companies" in md
        assert f"Topics: {len(TOPICS)}" in md
        assert "2026-06-29 09:00 UTC" in md

    def test_company_headings(self) -> None:
        brief = _make_brief()
        md = render_markdown(brief)
        assert "## AAPL — Apple Inc." in md
        assert "## MSFT — Microsoft Corp." in md

    def test_topic_sections_rendered(self) -> None:
        brief = _make_brief()
        md = render_markdown(brief)
        assert "**Earnings & Guidance:**" in md
        assert "Revenue grew 8%" in md
        assert "[1]" in md

    def test_no_content_placeholder(self) -> None:
        brief = _make_brief()
        md = render_markdown(brief)
        assert "_No material developments identified" in md

    def test_sources_appendix(self) -> None:
        brief = _make_brief()
        md = render_markdown(brief)
        assert "## Sources" in md
        assert "[1] Apple Q2 Earnings" in md
        assert "https://example.com/1" in md

    def test_no_sources_when_empty(self) -> None:
        brief = _make_brief(with_sources=False)
        md = render_markdown(brief)
        assert "## Sources" not in md

    def test_no_ticker_falls_back_to_name(self) -> None:
        # Build a brief where one company has no ticker
        no_ticker = PortfolioCompany(company_id="D8442A", name="Apple Inc.", ticker="")
        brief = _make_brief()
        brief.companies[0].company = no_ticker
        md = render_markdown(brief)
        assert "Apple Inc." in md


# ──────────────────────────────────────────────── HTML tests ──────────────────


class TestRenderHtml:
    def test_doctype_and_charset(self) -> None:
        brief = _make_brief()
        doc = render_html(brief)
        assert "<!DOCTYPE html>" in doc
        assert 'charset="UTF-8"' in doc

    def test_title_contains_date(self) -> None:
        brief = _make_brief()
        doc = render_html(brief)
        assert "<title>Morning Brief — 2026-06-29</title>" in doc

    def test_no_external_resources(self) -> None:
        brief = _make_brief()
        doc = render_html(brief)
        # Must not load any external resource
        for bad in ("https://cdn", "http://cdn", "fonts.googleapis", "cdnjs", "jsdelivr"):
            assert bad not in doc, f"External resource found: {bad}"

    def test_company_anchors_in_nav(self) -> None:
        brief = _make_brief()
        doc = render_html(brief)
        assert 'href="#company-AAPL"' in doc
        assert 'href="#company-MSFT"' in doc

    def test_company_card_ids(self) -> None:
        brief = _make_brief()
        doc = render_html(brief)
        assert 'id="company-AAPL"' in doc
        assert 'id="company-MSFT"' in doc

    def test_citation_links_rendered(self) -> None:
        brief = _make_brief()
        doc = render_html(brief)
        assert 'href="#src-1"' in doc
        assert 'id="src-1"' in doc

    def test_source_urls_are_hyperlinked(self) -> None:
        brief = _make_brief()
        doc = render_html(brief)
        assert 'href="https://example.com/1"' in doc

    def test_no_content_placeholder_for_msft(self) -> None:
        brief = _make_brief()
        doc = render_html(brief)
        assert "No material developments identified" in doc

    def test_inline_css_only(self) -> None:
        brief = _make_brief()
        doc = render_html(brief)
        assert "<style>" in doc
        assert '<link rel="stylesheet"' not in doc

    def test_html_escaping(self) -> None:
        # Company name with HTML special chars
        brief = _make_brief()
        brief.companies[0].company = PortfolioCompany(
            company_id="X", ticker="X", name="AT&T <Corp>"
        )
        doc = render_html(brief)
        assert "AT&amp;T" in doc or "AT&T" not in doc  # must be escaped or absent

    def test_javascript_included(self) -> None:
        brief = _make_brief()
        doc = render_html(brief)
        assert "<script>" in doc
        assert "scrollIntoView" in doc


# ──────────────────────────────────────────────── write_brief tests ───────────


class TestWriteBrief:
    def test_writes_md_file(self, tmp_path) -> None:
        brief = _make_brief()
        paths = write_brief(brief, tmp_path, formats=["md"])
        assert "md" in paths
        assert paths["md"].exists()
        content = paths["md"].read_text(encoding="utf-8")
        assert "# Morning Brief" in content

    def test_writes_html_file(self, tmp_path) -> None:
        brief = _make_brief()
        paths = write_brief(brief, tmp_path, formats=["html"])
        assert "html" in paths
        assert paths["html"].exists()
        content = paths["html"].read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_both_format_writes_two_files(self, tmp_path) -> None:
        brief = _make_brief()
        paths = write_brief(brief, tmp_path, formats=["both"])
        assert "md" in paths
        assert "html" in paths
        assert paths["md"].exists()
        assert paths["html"].exists()

    def test_filename_contains_date(self, tmp_path) -> None:
        brief = _make_brief()
        paths = write_brief(brief, tmp_path, formats=["both"])
        assert "20260629" in paths["md"].name
        assert "20260629" in paths["html"].name
