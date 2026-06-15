"""Visual QA gate contract tests for Politicians page."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "src" / "web" / "frontend" / "scripts" / "politicians-visual-qa.mjs"
PAGE = REPO_ROOT / "src" / "web" / "frontend" / "src" / "pages" / "PoliticiansPage.tsx"


def test_visual_qa_script_captures_required_viewports_and_states():
    text = SCRIPT.read_text(encoding="utf-8")

    for viewport in ("desktop", "tablet", "mobile"):
        assert f"name: '{viewport}'" in text
    for state in ("healthy", "degraded-low-confidence", "empty", "disabled", "loading"):
        assert f"name: '{state}'" in text
    assert "politicians-${scenario.name}-${viewport.name}.png" in text
    assert "politicians-${viewport.name}.png" in text


def test_visual_qa_script_checks_layout_breakage():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "assertNoBrokenLayout" in text
    assert "document.documentElement.scrollWidth - window.innerWidth" in text
    assert "broken table rows" in text


def test_politicians_page_uses_dashboard_surface_not_marketing_landing():
    text = PAGE.read_text(encoding="utf-8")

    assert "glass-card" in text
    assert "PoliticianInsightBar" in text
    assert "TradeFeedTable" in text
    assert "SourceHealthStrip" in text
    assert "hero" not in text.lower()
    assert "landing" not in text.lower()


def test_visual_qa_uses_semantic_dashboard_colors_and_spacing():
    text = "\n".join([
        PAGE.read_text(encoding="utf-8"),
        (REPO_ROOT / "src" / "web" / "frontend" / "src" / "features" / "politicians" / "components" / "SourceHealthStrip.tsx").read_text(encoding="utf-8"),
        (REPO_ROOT / "src" / "web" / "frontend" / "src" / "features" / "politicians" / "components" / "TradeFeedTable.tsx").read_text(encoding="utf-8"),
    ])

    assert "var(--accent-emerald)" in text
    assert "var(--accent-amber)" in text
    assert "var(--accent-rose)" in text
    assert "var(--violet-8)" in text
    assert "rounded-[8px]" in text
