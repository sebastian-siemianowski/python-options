"""
Ticker alias / canonicalization map.

Users frequently enter the "native" ticker they'd use on an exchange website
(e.g. ``BAYN`` for Bayer on Xetra, ``HO`` for Thales on Euronext Paris), but
Yahoo Finance — which drives our data pipeline — requires exchange-suffixed
symbols (``BAYN.DE``, ``HO.PA``). This module owns the single source of truth
for that translation so that both POST and DELETE endpoints resolve a given
user input to the same stored symbol.

Design notes
------------
- Keys are the raw UPPERCASE input; values are the Yahoo-compatible symbol.
- Aliases are bidirectional by design: ``canonicalize("BAYN")`` and
  ``canonicalize("BAYN.DE")`` both return ``"BAYN.DE"``.
- Unknown inputs pass through unchanged so the watchlist still accepts new
  symbols without code changes.
- The alias table is intentionally conservative — we only include tickers
  that are unambiguous or that have already been requested by users. When a
  ticker is also a valid US symbol (e.g. ``BA`` could be Boeing or BAE
  Systems), we follow the user's established preference from their existing
  watchlist rather than guessing.
"""

from __future__ import annotations

# ── Alias map ───────────────────────────────────────────────────────────────
# Format: "USER_INPUT": "YAHOO_CANONICAL"
TICKER_ALIASES: dict[str, str] = {
    # ── Xetra / German exchange (suffix .DE) ────────────────────────────────
    "BAYN": "BAYN.DE",     # Bayer
    "BMW": "BMW.DE",       # BMW common
    "BMW3": "BMW3.DE",     # BMW preferred
    "VOW": "VOW.DE",       # Volkswagen common
    "VOW3": "VOW3.DE",     # Volkswagen preferred
    "TKA": "TKA.DE",       # ThyssenKrupp
    "RHM": "RHM.DE",       # Rheinmetall
    "HAG": "HAG.DE",       # Hensoldt
    "R3NK": "R3NK.DE",     # Renk Group
    "RENK": "R3NK.DE",     # Renk Group (alt spelling)
    "SAP": "SAP.DE",       # SAP SE
    "SIE": "SIE.DE",       # Siemens
    "ALV": "ALV.DE",       # Allianz
    "DTE": "DTE.DE",       # Deutsche Telekom
    "IFX": "IFX.DE",       # Infineon
    "MBG": "MBG.DE",       # Mercedes-Benz Group
    "DB1": "DB1.DE",       # Deutsche Boerse
    "DBK": "DBK.DE",       # Deutsche Bank

    # ── Euronext Paris (suffix .PA) ─────────────────────────────────────────
    "HO": "HO.PA",         # Thales
    "AIR": "AIR.PA",       # Airbus
    "SAF": "SAF.PA",       # Safran
    "MC": "MC.PA",         # LVMH
    "OR": "OR.PA",         # L'Oreal
    "BNP": "BNP.PA",       # BNP Paribas
    "SU": "SU.PA",         # Schneider Electric
    "CAP": "CAP.PA",       # Capgemini
    "DG": "DG.PA",         # Vinci
    "EXA": "EXA.PA",       # Exail Technologies
    "AM": "AM.PA",         # Dassault Aviation

    # ── Warsaw GPW (suffix .WA) ─────────────────────────────────────────────
    "ACP": "ACP.WA",       # Asseco Poland SA
    "SNT": "SNT.WA",       # Synektik SA

    # ── Oslo Boers (suffix .OL) ─────────────────────────────────────────────
    "KOG": "KOG.OL",       # Kongsberg Gruppen
    "KOZ1": "KOG.OL",      # Kongsberg Gruppen (Xetra cross-listing shortcut)

    # ── LSE ETFs (suffix .L) ────────────────────────────────────────────────
    "MAGD": "MAGD.L",      # Amundi Magnificent 7 UCITS ETF (LSE)

    # ── London Stock Exchange (suffix .L) ───────────────────────────────────
    # Note: BA is ambiguous — on US it's Boeing, on LSE it's BAE Systems.
    # The user's existing watchlist uses BA.L (BAE), so we route here.
    "BA": "BA.L",          # BAE Systems
    "BA.": "BA.L",         # trailing-dot shortcut → BAE Systems

    # ── Toronto Stock Exchange (suffix .TO) ─────────────────────────────────
    "MDA": "MDA.TO",       # MDA Space

    # ── Vienna (suffix .VI) ────────────────────────────────────────────────
    "FACC": "FACC.VI",     # FACC AG

    # ── Euronext Amsterdam (suffix .AS) ────────────────────────────────────
    "THEON": "THEON.AS",   # Theon International

    # ── Australian Securities Exchange (suffix .AX) ─────────────────────────
    "BABY": "MVF.AX",      # Monash IVF Group (ticker is MVF on ASX)
    "BABY.AX": "MVF.AX",   # heal legacy watchlist entries
    "MVF": "MVF.AX",

    # ── FX pairs (Yahoo suffix =X) ─────────────────────────────────────────
    "XAGUSD": "XAGUSD=X",  # Silver / USD
    "XAUUSD": "XAUUSD=X",  # Gold   / USD
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",
    "USDCAD": "CAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "PLNJPY": "PLNJPY=X",
    "EURPLN": "EURPLN=X",
    "USDPLN": "PLN=X",

    # ── Common typos ────────────────────────────────────────────────────────
    "GOOO": "GOOG",        # Alphabet (class C)
    "QQQQ": "QQQ",         # Nasdaq 100 ETF
    "IKBR": "IBKR",        # Interactive Brokers
    "MSFI": "MSFT",        # Microsoft
    "METI": "META",        # Meta Platforms
    "TSLD": "TSLA",        # Tesla
    "AVGI": "AVGO",        # Broadcom
    "AMDI": "AMD",         # AMD
    "AMZD": "AMZN",        # Amazon

    # ── ETF aliases (delisted / alternative symbols) ────────────────────────
    "XLKS": "XLK",         # Technology Select Sector SPDR (likely typo for XLK)
    "GLDW": "GLD",         # WisdomTree Gold Enhanced (delisted) -> SPDR Gold Shares
}


def canonicalize(symbol: str) -> str:
    """Resolve a user-entered symbol to its Yahoo-compatible form.

    The lookup is case-insensitive and trims surrounding whitespace. If the
    symbol is not in the alias table, the normalized upper-case input is
    returned unchanged so unknown tickers still flow through to storage.

    The same function is used on both add and remove so that ``BAYN`` and
    ``BAYN.DE`` resolve to the same stored entry regardless of which form
    the user types.
    """
    if not symbol:
        return ""
    key = symbol.strip().upper()
    return TICKER_ALIASES.get(key, key)
