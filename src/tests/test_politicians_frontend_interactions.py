"""Frontend interaction contract tests for the Politicians page."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "src" / "web" / "frontend"


def test_page_wires_summary_table_filters_and_source_health_from_api():
    page = _read("src/pages/PoliticiansPage.tsx")
    table = _read("src/features/politicians/components/TradeFeedTable.tsx")
    source_health = _read("src/features/politicians/components/SourceHealthStrip.tsx")

    assert "api.politiciansSummary" in page
    assert "<PoliticianInsightBar" in page
    assert "<TradeFeedTable" in page
    assert "<SourceHealthStrip" in page
    assert "top_traders_only: topTradersOnly" in table
    assert "stock_linked_only: stockLinkedOnly" in table
    assert "transaction_side: quickTransactionSide || undefined" in table
    assert "filer: selectedTrader?.filer_name" in table
    assert "FilterInput" in table and "FilterSelect" in table and "Toggle" in table
    assert "api.politiciansSourceHealth" in source_health


def test_count_chips_apply_filters_to_trade_feed():
    insight_bar = _read("src/features/politicians/components/PoliticianInsightBar.tsx")
    table = _read("src/features/politicians/components/TradeFeedTable.tsx")
    page = _read("src/pages/PoliticiansPage.tsx")

    assert "onClick={() => onFilterChange(metric.filter)}" in insight_bar
    assert "activeFilter={activeFilter}" in page
    assert "insightFilter={activeFilter}" in page
    assert "insightFilter === 'tracked' && !row.is_tracked_asset" in table
    assert "insightFilter === 'watchlist' && !row.is_watchlist_asset" in table
    assert "insightFilter === 'late' && !rowFlags(row).includes('late_disclosure')" in table


def test_sorting_updates_table_order():
    table = _read("src/features/politicians/components/TradeFeedTable.tsx")

    assert "setSortDir((dir) => (dir === 'asc' ? 'desc' : 'asc'))" in table
    assert "setSortKey(next)" in table
    assert "compareRows(a, b, sortKey, sortDir)" in table
    assert "SORT_OPTIONS" in table
    assert "{ key: 'disclosure_date', label: 'Disclosure' }" in table
    assert "{ key: 'amount_mid_usd', label: 'Amount' }" in table
    assert "<SortChip" in table


def test_successful_trader_filter_is_available_and_backed_by_api_payload():
    table = _read("src/features/politicians/components/TradeFeedTable.tsx")
    api = _read("src/api.ts")

    assert "top_traders_only: topTradersOnly" in table
    assert "successful_trader_rank" in table
    assert "Top trader filter" in table
    assert "Top 10 congressional traders" in table
    assert "handleTopTradersToggle" in table
    assert "handleSelectTrader" in table
    assert "selectedTrader?.filer_key" in table
    assert "stockLinkedOnly" in table
    assert "quickTransactionSide" in table
    assert "handleQuickTransactionSide" in table
    assert "matchesTransactionSide" in table
    assert 'label="Purchase"' in table
    assert 'label="Sale"' in table
    assert "isStockLinkedRow" in table
    assert 'label="Stocks"' in table
    assert "onSelectTrader={handleSelectTrader}" in table
    assert "All top traders" in table
    assert "aria-pressed={selected}" in table
    assert "formatScore(trader.success_score)" in table
    assert "sortByNewestDisclosure" in table
    assert "setSortKey('disclosure_date')" in table
    assert "setSortDir('desc')" in table
    assert "tradesQ.data?.successful_traders?.leaderboard" in table
    assert "successful_traders?: PoliticiansSuccessfulTraders" in api


def test_filer_drawer_opens_and_closes_by_click_and_escape():
    page = _read("src/pages/PoliticiansPage.tsx")
    drawer = _read("src/features/politicians/components/FilerDrawer.tsx")

    assert "onSelectFiler={setSelectedFiler}" in page
    assert "filerId={selectedFiler}" in page
    assert "onClose={() => setSelectedFiler(null)}" in page
    assert "event.key === 'Escape'" in drawer
    assert "onMouseDown={onClose}" in drawer
    assert "onClick={onClose}" in drawer


def test_source_links_use_blank_target_and_safe_rel_attributes():
    for relative in (
        "src/features/politicians/components/TradeFeedTable.tsx",
        "src/features/politicians/components/FilerDrawer.tsx",
        "src/features/politicians/components/FilingAuditMetadata.tsx",
    ):
        text = _read(relative)
        assert 'target="_blank"' in text
        assert 'rel="noopener noreferrer"' in text


def test_disabled_state_renders_when_backend_marks_feature_unavailable():
    page = _read("src/pages/PoliticiansPage.tsx")

    assert "notice.status === 'disabled'" in page
    assert "notice.enabled === false" in page
    assert "<DisabledPanel" in page
    assert "Politician Monitoring Disabled" in page


def _read(relative: str) -> str:
    return (FRONTEND / relative).read_text(encoding="utf-8")
