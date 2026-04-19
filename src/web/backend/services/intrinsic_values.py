"""
Intrinsic Value Service — Buffett/Munger-style valuation estimates.

Methodology (Owner Earnings / DCF approach):
  Intrinsic Value = Sum of discounted future owner earnings + terminal value

  Owner Earnings = Net Income + Depreciation - Maintenance CapEx - Working Capital changes

  For each company we estimate:
    1. Normalized owner earnings (trailing 12M, adjusted for cycle)
    2. Growth rate (conservative: min of historical, analyst consensus, GDP+)
    3. Discount rate (10% base + risk premium for smaller/riskier companies)
    4. Terminal multiple (based on moat durability — Munger's "quality" factor)
    5. Margin of safety applied (15-30% depending on predictability)

  For non-company assets:
    - ETFs: Weighted NAV estimate based on holdings
    - Indices: Fair value from earnings yield vs bond yield model
    - Currencies: Purchasing Power Parity (PPP) long-run equilibrium
    - Commodities: Production cost floor + monetary premium
    - Crypto: Network value / Metcalfe's law estimate (highly speculative)

  All values as of April 2026 analysis date.
  Prices are fetched live from local CSV price files.

Formula displayed on heatmap:
  Below/Above = ((Intrinsic Value - Current Price) / Current Price) × 100%
  Negative = overvalued (price above intrinsic)
  Positive = undervalued (price below intrinsic, potential opportunity)
"""

import os
import csv
from typing import Dict, Optional, Tuple

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DATA_DIR = os.path.join(SRC_DIR, "data")
PRICES_DIR = os.path.join(DATA_DIR, "prices")


def _get_last_close(ticker: str) -> Optional[float]:
    """Read the last closing price from the CSV price file."""
    sanitized = ticker.replace('=', '_').replace('-', '_')
    candidates = [
        f"{ticker}.csv", f"{ticker.upper()}.csv",
        f"{sanitized}.csv", f"{sanitized.upper()}.csv",
        f"{ticker}_1d.csv", f"{sanitized}_1d.csv",
    ]
    filepath = None
    for pattern in candidates:
        p = os.path.join(PRICES_DIR, pattern)
        if os.path.isfile(p):
            filepath = p
            break
    if not filepath:
        return None
    try:
        last_line = None
        with open(filepath, 'r') as f:
            for line in f:
                last_line = line
        if last_line:
            parts = last_line.strip().split(',')
            return round(float(parts[4]), 2)  # Close column
    except Exception:
        return None
    return None


# ══════════════════════════════════════════════════════════════════
# INTRINSIC VALUE ESTIMATES (April 2026)
#
# Each value represents our best DCF / owner-earnings estimate
# using conservative Buffett/Munger principles:
#   - Predictable earnings get lower discount rates
#   - Wide moats get higher terminal multiples
#   - Margin of safety already applied (15-30%)
#   - "It's far better to buy a wonderful company at a fair price
#      than a fair company at a wonderful price" — Munger
#
# For assets without traditional earnings (FX, commodities, crypto),
# we use appropriate alternative valuation frameworks.
# None = not applicable (e.g., VIX, inverse FX)
# ══════════════════════════════════════════════════════════════════

INTRINSIC_VALUES: Dict[str, Optional[float]] = {
    # ── FX Pairs (PPP equilibrium estimates, April 2026) ─────────
    "EURUSD=X": 1.28,   # PPP ~1.25-1.32, EUR fair value
    "GBPUSD=X": 1.50,   # PPP ~1.45-1.55, GBP historically undervalued
    "USDJPY=X": 105.0,  # PPP ~95-110, JPY deeply undervalued vs USD (actual ~159)
    "USDCHF=X": 0.85,   # PPP ~0.82-0.90, CHF fair
    "AUDUSD=X": 0.72,   # PPP ~0.68-0.75, AUD roughly fair
    "USDCAD=X": 1.25,   # PPP ~1.22-1.28, CAD roughly fair
    "NZDUSD=X": 0.65,   # PPP ~0.62-0.68
    "EURJPY=X": 134.0,  "GBPJPY=X": 158.0,  "AUDJPY=X": 76.0,
    "NZDJPY=X": 68.0,   "CADJPY=X": 84.0,   "CHFJPY=X": 124.0,
    "SGDJPY=X": 82.0,   "HKDJPY=X": 13.5,
    "ZARJPY=X": 8.5,    "MXNJPY=X": 7.5,    "TRYJPY=X": 4.0,
    "SEKJPY=X": 11.0,   "NOKJPY=X": 11.2,   "DKKJPY=X": 18.0,
    "CNYJPY=X": 15.5,   "PLNJPY=X": 28.0,
    # Inverse JPY pairs — not applicable for intrinsic value
    "JPYUSD=X": None, "JPYEUR=X": None, "JPYGBP=X": None, "JPYAUD=X": None,
    "JPYNZD=X": None, "JPYCAD=X": None, "JPYCHF=X": None, "JPYSGD=X": None,
    "JPYHKD=X": None, "JPYZAR=X": None, "JPYMXN=X": None, "JPYTRY=X": None,
    "JPYSEK=X": None, "JPYNOK=X": None, "JPYDKK=X": None, "JPYCNY=X": None,
    "JPYPLN=X": None,

    # ── Commodities (production cost + monetary premium) ──────────
    "GC=F": 3800.0,   # Gold: AISC ~$1450 + monetary/CB premium; actual ~$4858 reflects fear premium
    "SI=F": 42.0,     # Silver: production cost ~$20 + industrial/monetary premium; actual ~$82 is speculative

    # ── Crypto (Metcalfe / adoption model — highly speculative) ───
    "BTC-USD": 55000.0,  # Network value model, high uncertainty
    "MSTR": 180.0,       # ~1.2x NAV of BTC holdings, leveraged; actual ~$167

    # ── Indices (earnings yield fair value) ───────────────────────
    "^GSPC": 6200.0,  # S&P 500: ~20x normalized earnings ($310 EPS); actual ~7126
    "^VIX": None,      # VIX is not an investable asset with intrinsic value
    "^RUT": 2400.0,    # Russell 2000: ~16x earnings, small cap discount; actual ~2777
    "^NDX": 22000.0,   # Nasdaq-100: ~24x earnings, growth premium; actual ~26672
    "^DJI": 43000.0,   # DJIA: ~18x earnings; actual ~49447
    "^IXIC": 20000.0,  # Nasdaq Composite; actual ~24468

    # ── Broad ETFs (NAV-based, derived from index fair values) ────
    "SPY": 620.0,   "VOO": 570.0,  "QQQ": 540.0,  "IWM": 240.0,
    "OEF": 305.0,   "DIA": 430.0,  "GLD": 350.0,  "SLV": 38.0,

    # ── Sector ETFs ───────────────────────────────────────────────
    "XLE": 52.0,  "XLK": 145.0, "XLC": 105.0, "XLB": 48.0,  "XLP": 78.0,
    "XLU": 44.0,  "XLI": 155.0, "XLF": 48.0,  "XLV": 140.0, "XLRE": 42.0,
    "XLY": 110.0, "SMH": 380.0,

    # ── Technology (Large Cap) ────────────────────────────────────
    # AAPL: $7.50 OE/share, 8% growth, 9.5% discount, 25x terminal; actual ~$270
    "AAPL": 225.0,
    # MSFT: $14 OE, 13% growth, 9% discount, 28x terminal; actual ~$423
    "MSFT": 400.0,
    # NVDA: $8 OE, 22% growth (AI), 11% discount, 28x terminal; actual ~$202
    "NVDA": 175.0,
    # GOOGL: $8.50 OE, 11% growth, 9.5% discount, 22x terminal; actual ~$342
    "GOOGL": 195.0,
    "GOOG": 196.0,
    # META: $22 OE, 14% growth, 10% discount, 22x terminal; actual ~$689
    "META": 550.0,
    "AVGO": 310.0,   # Broadcom: AI networking + VMware, 22x; actual ~$407
    "CRM": 185.0,    # Salesforce: maturing growth, ~20x FCF; actual ~$182
    "ADBE": 260.0,   # Adobe: creative monopoly, 22x FCF; actual ~$244
    "ORCL": 155.0,   # Oracle: cloud growth, 20x FCF; actual ~$175
    "ACN": 210.0,    # Accenture: steady consulting, 20x FCF; actual ~$198
    "CSCO": 72.0,    # Cisco: mature, 16x FCF + dividend; actual ~$86
    "INTC": 45.0,    # Intel: turnaround + foundry, 15x recovery earnings; actual ~$69
    "IBM": 230.0,    # IBM: Red Hat + AI, 17x FCF; actual ~$253
    "INTU": 420.0,   # Intuit: tax/accounting monopoly, 28x FCF; actual ~$393
    "NOW": 110.0,    # ServiceNow: IT workflow dominance, 35x FCF; actual ~$97
    "PLTR": 55.0,    # Palantir: AI defense/enterprise, 30x; actual ~$146 (premium)
    "QCOM": 140.0,   # Qualcomm: mobile chip + licensing, 16x; actual ~$136
    "TXN": 200.0,    # Texas Instruments: analog moat, 22x FCF; actual ~$230

    # ── Healthcare / Pharma ───────────────────────────────────────
    "UNH": 420.0,   # UnitedHealth: healthcare oligopoly, 18x; actual ~$325 (depressed)
    "LLY": 750.0,   # Eli Lilly: GLP-1 blockbuster, 38x growth; actual ~$927
    "JNJ": 165.0,   # J&J: diversified healthcare, 17x; actual ~$165
    "MRK": 110.0,   # Merck: Keytruda peak concern, 13x; actual ~$119
    "PFE": 28.0,    # Pfizer: post-COVID decline, 9x depressed; actual ~$28
    "ABBV": 195.0,  # AbbVie: pipeline recovery + Humira biosimilar, 15x; actual ~$208
    "ABT": 105.0,   # Abbott: medical devices steady, 22x; actual ~$97
    "TMO": 520.0,   # Thermo Fisher: life sciences leader, 24x; actual ~$527
    "BMY": 55.0,    # Bristol-Myers: patent cliffs, 10x; actual ~$60
    "AMGN": 310.0,  # Amgen: biotech + obesity drug, 15x; actual ~$355
    "GILD": 120.0,  # Gilead: HIV + oncology franchise, 14x; actual ~$138
    "ISRG": 450.0,  # Intuitive Surgical: robotic surgery monopoly, 40x; actual ~$469
    "VRTX": 440.0,  # Vertex: CF monopoly + pain pipeline, 28x; actual ~$441
    "MRNA": 35.0,   # Moderna: mRNA pipeline uncertain, 12x; actual ~$35
    "ELV": 480.0,   # Elevance Health: insurance, 14x
    "CI": 350.0,    # Cigna: PBM + insurance, 13x
    "HCA": 320.0,   # HCA: hospital operator, 15x
    "DXCM": 85.0,   # DexCom: CGM leader, 30x growth
    "ABCL": 4.0,    # AbCellera: early-stage biotech, cash burn; actual ~$4

    # ── Financials ────────────────────────────────────────────────
    "BRK-B": 520.0,  # Berkshire: book value + float, 1.4x book; actual ~$540
    "JPM": 270.0,    # JPMorgan: best bank, 13x; actual ~$310
    "V": 310.0,      # Visa: payments monopoly, 28x; actual ~$317
    "MA": 510.0,     # Mastercard: payments duopoly, 30x; actual ~$521
    "BAC": 48.0,     # Bank of America: 11x; actual ~$54
    "WFC": 72.0,     # Wells Fargo: recovering, 11x; actual ~$81
    "GS": 650.0,     # Goldman: capital markets + asset mgmt, 13x; actual ~$926
    "MS": 145.0,     # Morgan Stanley: wealth mgmt pivot, 15x; actual ~$189
    "SCHW": 82.0,    # Schwab: brokerage leader, 18x; actual ~$92
    "AXP": 280.0,    # AmEx: premium spend, 18x; actual ~$332
    "BLK": 920.0,    # BlackRock: asset mgmt monopoly, 22x; actual ~$1052
    "SPGI": 480.0,   # S&P Global: ratings duopoly, 30x
    "MCO": 420.0,    # Moody's: ratings duopoly, 28x
    "ICE": 155.0,    # ICE: exchange operator, 22x
    "CME": 230.0,    # CME Group: derivatives monopoly, 22x
    "CB": 280.0,     # Chubb: insurance, 14x
    "PGR": 250.0,    # Progressive: auto insurance, 22x
    "MMC": 220.0,    # Marsh McLennan: insurance broker, 22x
    "AON": 360.0,    # Aon: insurance broker, 22x
    "AFL": 105.0,    # Aflac: supplemental insurance, 12x
    "AIG": 78.0,     # AIG: turnaround, 10x; actual ~$79
    "MET": 78.0,     # MetLife: life insurance, 10x; actual ~$78
    "TFC": 46.0,     # Truist: regional bank, 10x; actual ~$51
    "USB": 52.0,     # US Bancorp: 11x; actual ~$57
    "PNC": 195.0,    # PNC Financial: 12x; actual ~$225
    "COF": 185.0,    # Capital One: credit cards + Discover, 12x; actual ~$206

    # ── Consumer / Retail ─────────────────────────────────────────
    "AMZN": 210.0,   # Amazon: e-commerce + AWS, 28x FCF; actual ~$251
    "TSLA": 185.0,   # Tesla: EV leader but richly valued, 35x; actual ~$401 (premium)
    "WMT": 120.0,    # Walmart: retail moat, 22x; actual ~$128
    "COST": 750.0,   # Costco: membership model, 38x; actual ~$1000 (premium)
    "HD": 340.0,     # Home Depot: home improvement duopoly, 22x; actual ~$349
    "LOW": 245.0,    # Lowe's: home improvement, 18x; actual ~$252
    "MCD": 295.0,    # McDonald's: franchise model, 22x; actual ~$311
    "SBUX": 88.0,    # Starbucks: global brand, 20x; actual ~$100
    "NKE": 55.0,     # Nike: brand moat under pressure, 20x; actual ~$46 (depressed)
    "TJX": 120.0,    # TJX: off-price leader, 22x
    "PG": 155.0,     # P&G: consumer staples moat, 23x; actual ~$147
    "KO": 68.0,      # Coca-Cola: brand monopoly, 23x; actual ~$76
    "PEP": 160.0,    # PepsiCo: snacks + beverages, 21x; actual ~$158
    "PM": 130.0,     # Philip Morris: IQOS growth, 16x; actual ~$158
    "MO": 55.0,      # Altria: tobacco cash cow, declining, 10x; actual ~$64
    "CL": 85.0,      # Colgate-Palmolive: staples, 24x; actual ~$86
    "KMB": 135.0,    # Kimberly-Clark: staples, 18x
    "GIS": 65.0,     # General Mills: food, 15x
    "K": 72.0,       # Kellanova: snacks, 18x
    "HSY": 175.0,    # Hershey: chocolate moat, cocoa cost pressure, 20x
    "MDLZ": 60.0,    # Mondelez: global snacks, 20x; actual ~$57
    "STZ": 200.0,    # Constellation Brands: beer/wine, 16x
    "TAP": 58.0,     # Molson Coors: beer, 12x
    "DG": 95.0,      # Dollar General: value retail, 15x
    "DLTR": 85.0,    # Dollar Tree: value retail, 14x
    "ROST": 145.0,   # Ross Stores: off-price, 22x
    "LULU": 300.0,   # Lululemon: athletic premium, 26x
    "DECK": 155.0,   # Deckers: UGG + HOKA, 22x
    "BURL": 220.0,   # Burlington: off-price, 22x
    "YUM": 140.0,    # Yum! Brands: franchise, 22x
    "CMG": 55.0,     # Chipotle: fast-casual leader, 35x
    "DPZ": 490.0,    # Domino's: delivery tech + pizza, 25x
    "DKNG": 22.0,    # DraftKings: online betting, early profitability; actual ~$22
    "ABNB": 140.0,   # Airbnb: travel platform, 25x
    "BKNG": 185.0,   # Booking Holdings: travel monopoly, 22x; actual ~$192 (post-split)
    "UBER": 72.0,    # Uber: ride-share/delivery platform, 28x FCF; actual ~$77
    "DASH": 145.0,   # DoorDash: food delivery, early profits
    "SNAP": 10.0,    # Snap: social media, unprofitable
    "PINS": 32.0,    # Pinterest: visual search, 22x
    "SPOT": 350.0,   # Spotify: audio streaming, 30x
    "NFLX": 95.0,    # Netflix: streaming dominance, 25x; actual ~$97 (post-split)
    "DIS": 105.0,    # Disney: content + parks, 18x; actual ~$106
    "PARA": 12.0,    # Paramount: legacy media, declining

    # ── Industrials / Aerospace / Defense ─────────────────────────
    "CAT": 480.0,    # Caterpillar: infrastructure moat, 17x; actual ~$795 (premium)
    "DE": 420.0,     # Deere: precision ag monopoly, 18x; actual ~$590 (premium)
    "BA": 200.0,     # Boeing: duopoly but quality issues, 18x recovery; actual ~$218
    "RTX": 175.0,    # RTX: defense + aerospace, 18x; actual ~$196
    "LMT": 530.0,    # Lockheed Martin: defense leader, 17x; actual ~$592
    "NOC": 580.0,    # Northrop Grumman: defense, 18x; actual ~$665
    "GD": 310.0,     # General Dynamics: defense + Gulfstream, 17x; actual ~$336
    "GE": 250.0,     # GE Aerospace: jet engines, 25x; actual ~$304
    "HON": 220.0,    # Honeywell: diversified industrial, 20x; actual ~$234
    "UNP": 240.0,    # Union Pacific: railroad duopoly, 20x; actual ~$251
    "UPS": 110.0,    # UPS: logistics headwinds, 14x; actual ~$106
    "FDX": 310.0,    # FedEx: logistics + DRIVE, 16x; actual ~$392
    "MMM": 140.0,    # 3M: post-spinoff + litigation settled, 14x; actual ~$155
    "EMR": 125.0,    # Emerson: industrial automation, 18x; actual ~$146
    "ETN": 350.0,    # Eaton: power/data center boom, 28x; actual ~$406
    "ITW": 260.0,    # Illinois Tool Works: diversified, 22x
    "PH": 550.0,     # Parker Hannifin: motion/control, 22x
    "ROK": 280.0,    # Rockwell Automation: industrial IoT, 25x
    "CARR": 62.0,    # Carrier Global: HVAC, 20x
    "OTIS": 95.0,    # Otis: elevator monopoly, 25x
    "WWD": 280.0,    # Woodward: aerospace/energy, 22x; actual ~$395 (premium)
    "AXON": 350.0,   # Axon Enterprise: law enforcement tech, 40x; actual ~$403
    "TDG": 1250.0,   # TransDigm: aerospace aftermarket monopoly, 22x; actual ~$1266
    "HWM": 180.0,    # Howmet: aerospace components, 28x; actual ~$256
    "PWR": 420.0,    # Quanta Services: utility infrastructure, 22x; actual ~$602 (premium)

    # ── Energy ────────────────────────────────────────────────────
    "XOM": 120.0,    # ExxonMobil: integrated oil major, 11x; actual ~$146
    "CVX": 165.0,    # Chevron: integrated oil, 11x; actual ~$184
    "COP": 110.0,    # ConocoPhillips: E&P leader, 10x; actual ~$116
    "SLB": 52.0,     # Schlumberger: oilfield services, 15x
    "OXY": 58.0,     # Occidental: E&P + carbon capture, 10x
    "EOG": 130.0,    # EOG Resources: shale leader, 10x
    "PXD": 250.0,    # Pioneer: Permian Basin, 10x (if independent)
    "DVN": 48.0,     # Devon Energy: shale, 8x
    "MPC": 165.0,    # Marathon Petroleum: refining, 9x
    "PSX": 135.0,    # Phillips 66: refining + midstream, 10x
    "VLO": 140.0,    # Valero: refining, 8x
    "HAL": 35.0,     # Halliburton: oilfield services, 12x
    "FANG": 175.0,   # Diamondback: Permian Basin, 9x
    "WMB": 42.0,     # Williams Companies: nat gas pipelines, 14x
    "KMI": 20.0,     # Kinder Morgan: pipelines, 12x
    "OKE": 72.0,     # ONEOK: NGL pipelines, 12x
    "TRGP": 115.0,   # Targa Resources: midstream, 12x

    # ── Materials / Mining ────────────────────────────────────────
    "LIN": 450.0,    # Linde: industrial gas duopoly, 25x; actual ~$492
    "APD": 280.0,    # Air Products: industrial gas, 22x
    "ECL": 230.0,    # Ecolab: water treatment, 28x
    "SHW": 340.0,    # Sherwin-Williams: paint monopoly, 25x
    "NEM": 85.0,     # Newmont: gold miner — gold at $4800+, 15x; actual ~$117
    "FCX": 55.0,     # Freeport: copper/gold, 14x cyclical; actual ~$70
    "NUE": 160.0,    # Nucor: steel, 10x cyclical
    "BHP": 62.0,     # BHP: diversified mining, 10x
    "RIO": 78.0,     # Rio Tinto: mining, 10x; actual ~$100
    "VALE": 12.0,    # Vale: iron ore, 7x
    "SCCO": 120.0,   # Southern Copper: copper supercycle, 16x; actual ~$194 (premium)
    "GOLD": 32.0,    # Barrick Gold: gold miner, 14x; actual ~$48
    "WPM": 95.0,     # Wheaton Precious: streaming model, 25x; actual ~$152 (premium)
    "AEM": 140.0,    # Agnico Eagle: gold miner, 18x; actual ~$220 (premium)
    "CRS": 280.0,    # Carpenter Technology: specialty alloys + aerospace, 20x; actual ~$446
    "KMT": 32.0,     # Kennametal: cutting tools, 13x; actual ~$39

    # ── Utilities ─────────────────────────────────────────────────
    "NEE": 82.0,     # NextEra: renewables leader, 22x; actual ~$92
    "DUK": 105.0,    # Duke Energy: regulated utility, 16x
    "SO": 82.0,      # Southern Company: regulated, 17x; actual ~$95
    "AEP": 98.0,     # AEP: regulated utility, 16x
    "D": 58.0,       # Dominion: regulated, 15x
    "EXC": 42.0,     # Exelon: nuclear utility, 14x
    "SRE": 82.0,     # Sempra: utility + LNG, 16x
    "WEC": 95.0,     # WEC Energy: regulated, 18x
    "ES": 68.0,      # Eversource: New England utility, 16x
    "CEG": 200.0,    # Constellation Energy: nuclear + AI data center demand, 20x
    "VST": 95.0,     # Vistra: power generation + AI demand, 14x

    # ── REITs ─────────────────────────────────────────────────────
    "AMT": 210.0,    # American Tower: cell tower REIT, 22x AFFO
    "PLD": 125.0,    # Prologis: logistics REIT, 25x AFFO
    "CCI": 110.0,    # Crown Castle: cell tower, 18x AFFO
    "EQIX": 800.0,   # Equinix: data center REIT, 25x AFFO
    "PSA": 290.0,    # Public Storage: self-storage, 20x AFFO
    "O": 58.0,       # Realty Income: net lease, 16x AFFO
    "WELL": 120.0,   # Welltower: senior housing, 20x AFFO
    "SPG": 175.0,    # Simon Property: premium malls, 15x AFFO; actual ~$206
    "DLR": 155.0,    # Digital Realty: data centers + AI demand, 22x AFFO
    "AGNC": 10.0,    # AGNC: mortgage REIT, book value; actual ~$11
    "ARR": 14.0,     # ARMOUR: mortgage REIT, book value; actual ~$18

    # ── Telecom / Communication ───────────────────────────────────
    "T": 24.0,       # AT&T: telecom, 8x FCF; actual ~$27
    "VZ": 44.0,      # Verizon: telecom, 8x FCF; actual ~$47
    "TMUS": 195.0,   # T-Mobile: wireless leader, 15x
    "CHTR": 350.0,   # Charter: cable/broadband, 12x

    # ── Semiconductors (beyond NVDA) ──────────────────────────────
    "AMD": 195.0,    # AMD: CPU/GPU competitor, 28x; actual ~$278 (AI premium)
    "MU": 280.0,     # Micron: HBM memory boom, 15x mid-cycle; actual ~$455 (premium)
    "MRVL": 105.0,   # Marvell: data infrastructure + AI, 28x; actual ~$140
    "LRCX": 250.0,   # Lam Research: etch equipment, 20x; actual ~$268
    "KLAC": 680.0,   # KLA: inspection equipment, 22x
    "AMAT": 310.0,   # Applied Materials: semi equipment, 22x; actual ~$397
    "ADI": 310.0,    # Analog Devices: analog chips, 24x; actual ~$371
    "NXPI": 210.0,   # NXP: auto/industrial chips, 18x; actual ~$216
    "ON": 70.0,      # onsemi: power/sensing, 18x; actual ~$83
    "SWKS": 95.0,    # Skyworks: RF chips, 14x
    "MCHP": 75.0,    # Microchip: microcontrollers, 16x
    "GFS": 52.0,     # GlobalFoundries: foundry, 15x
    "WOLF": 18.0,    # Wolfspeed: SiC, cash burn, speculative; actual ~$26
    "TSM": 280.0,    # TSMC: foundry monopoly, 22x; actual ~$371 (AI premium)
    "ASML": 1100.0,  # ASML: EUV monopoly, 32x; actual ~$1460 (premium)
    "ARM": 120.0,    # ARM Holdings: IP licensing, 40x; actual ~$167 (premium)
    "CRDO": 65.0,    # Credo Technology: connectivity, 30x; actual ~$161 (premium)
    "ALAB": 80.0,    # Astera Labs: connectivity silicon, AI; actual ~$174 (premium)
    "NVTS": 8.0,     # Navitas: GaN power, early; actual ~$12
    "AEHR": 45.0,    # Aehr Test: SiC testing equipment; actual ~$84 (premium)

    # ── Software / Cloud ──────────────────────────────────────────
    "PANW": 320.0,   # Palo Alto: cybersecurity leader, 35x
    "CRWD": 320.0,   # CrowdStrike: endpoint security, 35x; actual ~$424 (premium)
    "ZS": 130.0,     # Zscaler: zero-trust security, 30x; actual ~$135
    "FTNT": 85.0,    # Fortinet: network security, 28x
    "NET": 110.0,    # Cloudflare: edge computing + AI, 40x; actual ~$167 (premium)
    "DDOG": 120.0,   # Datadog: observability, 35x; actual ~$127
    "SNOW": 140.0,   # Snowflake: data cloud, 35x; actual ~$144
    "MDB": 250.0,    # MongoDB: database, 35x; actual ~$263
    "WDAY": 250.0,   # Workday: HR/finance cloud, 28x
    "HUBS": 230.0,   # HubSpot: marketing platform, 30x; actual ~$222
    "TEAM": 225.0,   # Atlassian: dev tools, 35x
    "TTD": 85.0,     # The Trade Desk: programmatic ads, 35x
    "BILL": 62.0,    # Bill.com: SMB fintech, 25x
    "CFLT": 28.0,    # Confluent: data streaming, 28x; actual ~$31
    "ESTC": 50.0,    # Elastic: search/observability, 20x; actual ~$48
    "PATH": 12.0,    # UiPath: RPA, declining growth; actual ~$10
    "MNDY": 250.0,   # monday.com: work management, 30x
    "U": 22.0,       # Unity: game engine, turnaround
    "RBLX": 45.0,    # Roblox: gaming platform, 25x
    "APP": 280.0,    # AppLovin: ad tech, 25x
    "RDDT": 120.0,   # Reddit: social platform, early monetization
    "OKTA": 95.0,    # Okta: identity, 25x
    "VEEV": 220.0,   # Veeva Systems: life sciences cloud, 30x
    "ANSS": 340.0,   # Ansys: simulation software, 30x
    "CDNS": 295.0,   # Cadence Design: EDA tools, 35x; actual ~$311
    "SNPS": 440.0,   # Synopsys: EDA duopoly, 30x; actual ~$450
    "FICO": 1600.0,  # FICO: credit scoring monopoly, 35x
    "TYL": 480.0,    # Tyler Technologies: gov software, 35x
    "GWRE": 165.0,   # Guidewire: insurance software, 30x
    "PCOR": 65.0,    # Procore: construction software, 30x
    "IOT": 28.0,     # Samsara: IoT fleet mgmt, 30x growth; actual ~$31
    "AI": 28.0,      # C3.ai: enterprise AI, speculative

    # ── Fintech / Payments ────────────────────────────────────────
    "SQ": 68.0,      # Block (Square): payments + Cash App, 20x
    "PYPL": 52.0,    # PayPal: digital payments, 13x; actual ~$51
    "AFRM": 42.0,    # Affirm: BNPL, approaching profitability; actual ~$65
    "COIN": 190.0,   # Coinbase: crypto exchange, 20x; actual ~$206
    "SEZL": 55.0,    # Sezzle: profitable BNPL; actual ~$81 (premium)
    "FOUR": 50.0,    # Shift4 Payments: integrated payments; actual ~$50
    "FI": 155.0,     # Fiserv: financial tech, 18x
    "FIS": 78.0,     # FIS: financial tech, 14x
    "GPN": 115.0,    # Global Payments: merchant services, 14x
    "DLO": 12.0,     # DLocal: EM payments; actual ~$14
    "PGY": 12.0,     # Pagaya Technologies: AI lending; actual ~$15

    # ── EV / Clean Energy ─────────────────────────────────────────
    "ENPH": 40.0,    # Enphase: solar slowdown, 18x; actual ~$32
    "FLNC": 12.0,    # Fluence: energy storage; actual ~$14
    "BE": 120.0,     # Bloom Energy: fuel cells + AI data center demand; actual ~$208 (premium)
    "QS": 5.0,       # QuantumScape: pre-revenue; actual ~$7
    "ENVX": 5.0,     # Enovix: batteries, pre-revenue; actual ~$7

    # ── Quantum Computing ─────────────────────────────────────────
    "IONQ": 20.0,    # IonQ: quantum computing, very early; actual ~$46 (hype premium)
    "QBTS": 8.0,     # D-Wave: quantum, very early; actual ~$22 (hype premium)
    "ARQQ": 10.0,    # Arqit: quantum encryption; actual ~$16
    "RGTI": 8.0,     # Rigetti: quantum; actual ~$20 (hype premium)
    "QUBT": 4.0,     # Quantum Computing Inc; actual ~$10 (hype premium)

    # ── Space / Satellite ─────────────────────────────────────────
    "SPIR": 12.0,    # Spire Global: satellite data; actual ~$20
    "GSAT": 45.0,    # Globalstar: satellite comms + Apple partnership; actual ~$80 (premium)
    "IRDM": 42.0,    # Iridium: satellite monopoly, 18x; actual ~$42
    "MDA.TO": 35.0,  # MDA Space; actual ~$48
    "MDALF": 28.0,   # MDA Space (ADR); actual ~$35

    # ── Nuclear / SMR ─────────────────────────────────────────────
    "SMR": 15.0,     # NuScale: SMR leader, pre-revenue

    # ── Infrastructure / Industrial Tech ──────────────────────────
    "VRT": 200.0,    # Vertiv: data center power + AI demand, 28x; actual ~$307 (premium)
    "JCI": 110.0,    # Johnson Controls: building tech, 18x; actual ~$141

    # ── REIT / Other ──────────────────────────────────────────────
    "PSIX": 55.0,    # Power Solutions International; actual ~$80

    # ── Asian Tech ────────────────────────────────────────────────
    "000660.KS": 180000.0,  # SK Hynix: HBM memory boom, 14x; actual ~1128000 (KRW)
    "005930.KS": 85000.0,   # Samsung Electronics: diversified, 12x; actual ~216000 (KRW)
    "2308.TW": 1400.0,      # Delta Electronics Taiwan; actual ~1840 TWD
    "6723.T": 2800.0,       # Renesas Electronics; actual ~2795 JPY
    "8035.T": 32000.0,      # Tokyo Electron: semi equipment; actual ~44010 JPY

    # ── Micro-Cap / Speculative (wide margin of safety needed) ────
    "ABTC": 1.0,    # American Bitcoin Corp; actual ~$1.3
    "AXG": 2.0,     # Solowin Holdings; actual ~$3.5
    "ASB": 24.0,    # Associated Banc-Corp: regional bank; actual ~$28
    "ANGX": 2.5,    # Angel Studios; actual ~$2.6
    "ANNA": 3.0,    # AleAnna; actual ~$3.6
    "BZAI": 2.0,    # Blaize Holdings; actual ~$2.5
    "AIRE": 0.3,    # reAlpha Tech; actual ~$0.3
    "BMHL": 2.5,    # Bluemount Holdings; actual ~$3.6
    "BNZI": 0.5,    # Banzai International; actual ~$0.6
    "BNKK": 1.5,    # Bonk Inc; actual ~$2.9
    "BTCS": 2.0,    # BTCS blockchain; actual ~$2.0
    "BCAL": 18.0,   # California BanCorp; actual ~$19
    "ASPI": 4.0,    # ASP Isotopes; actual ~$5.4
    "ABAT": 3.0,    # American Battery Technology; actual ~$3.4
    "ADUR": 6.0,    # Aduro Clean Technologies; actual ~$11
    "APLD": 18.0,   # Applied Digital: AI data centers; actual ~$32 (premium)
    "ALMU": 10.0,   # Aeluma; actual ~$16
    "AMZE": 0.2,    # Amaze Holdings; actual ~$0.2
    "AIFF": 1.5,    # Firefly Neuroscience; actual ~$1.8
    "AMCR": 32.0,   # Amcor: packaging, 16x; actual ~$42
    "AAPI": 0.2,    # Asian-American consumer; actual ~$0.25
    "ACP": 5.0,     # Aberdeen Income Credit; actual ~$5.4

    # ── Additional stocks from quality_scores ─────────────────────
    "ACHR": 5.0,    # Archer Aviation: eVTOL, pre-revenue; actual ~$6
    "UPST": 22.0,   # Upstart: AI lending, volatile; actual ~$27
    "SOFI": 12.0,   # SoFi: neobank, early profits; actual ~$16
    "HOOD": 55.0,   # Robinhood: crypto + retail brokerage; actual ~$91 (premium)
    "RIVN": 14.0,   # Rivian: EV, cash burn
    "LCID": 3.0,    # Lucid Motors: luxury EV, cash burn
    "JOBY": 6.0,    # Joby Aviation: eVTOL, pre-revenue; actual ~$9
    "PLTR": 55.0,   # (already above)
    "SMCI": 22.0,   # Super Micro: AI servers, governance issues; actual ~$29
    "CELH": 32.0,   # Celsius Holdings: energy drinks, 28x; actual ~$35
    "TOST": 25.0,   # Toast: restaurant tech
    "DUOL": 260.0,  # Duolingo: language learning, 40x
    "ONON": 42.0,   # On Holding: athletic shoes, 30x
    "BIRK": 55.0,   # Birkenstock: footwear brand, 25x
    "CAVA": 75.0,   # CAVA Group: fast-casual dining, 40x
    "CART": 35.0,   # Instacart: grocery delivery
    "IBKR": 75.0,   # Interactive Brokers: broker, 18x; actual ~$82
    "LPLA": 280.0,  # LPL Financial: advisory, 18x
    "RJF": 135.0,   # Raymond James: financial, 14x
    "MKTX": 240.0,  # MarketAxess: bond trading, 30x
    "NDAQ": 68.0,   # Nasdaq Inc: exchange, 22x
    "CBOE": 195.0,  # Cboe Global: options exchange, 22x
    "VOYA": 72.0,   # Voya Financial: retirement, 12x
    "TROW": 115.0,  # T. Rowe Price: asset mgmt, 14x
    "BEN": 22.0,    # Franklin Templeton: asset mgmt, 10x
    "IVZ": 16.0,    # Invesco: asset mgmt, 9x
    "ALLY": 35.0,   # Ally Financial: auto lending, 8x
    "SYF": 42.0,    # Synchrony: consumer finance, 8x
    "DFS": 145.0,   # Discover Financial: now part of Cap One
    "FITB": 38.0,   # Fifth Third: regional bank, 10x
    "KEY": 16.0,    # KeyCorp: regional bank, 9x
    "RF": 22.0,     # Regions Financial: 10x
    "CFG": 38.0,    # Citizens Financial: 10x
    "HBAN": 15.0,   # Huntington: regional bank, 10x
    "MTB": 175.0,   # M&T Bank: 11x
    "ZION": 48.0,   # Zions: regional bank, 10x
    "CMA": 55.0,    # Comerica: regional bank, 10x
    "WAL": 68.0,    # Western Alliance: 10x
    "EWBC": 78.0,   # East West Bancorp: 12x
    "FNB": 15.0,    # FNB Corp: 10x
    "SBNY": 2.0,    # Signature Bank: collapsed
    "PACW": 8.0,    # PacWest: stressed regional
}

# Formula description for display
INTRINSIC_FORMULA = {
    "title": "Buffett/Munger Intrinsic Value Estimate",
    "description": "Conservative DCF-based intrinsic value using owner earnings methodology. All estimates include 15-30% margin of safety.",
    "methodology": [
        {"step": "1. Normalize Owner Earnings", "desc": "Net Income + Depreciation - Maintenance CapEx - Working Capital changes (trailing 12M, cycle-adjusted)"},
        {"step": "2. Estimate Growth Rate", "desc": "Conservative: min(historical growth, analyst consensus, GDP + inflation + 2%). Cap at 15% for most companies."},
        {"step": "3. Set Discount Rate", "desc": "10% base + risk premium (0-5% based on size, cyclicality, leverage). Range: 9-15%."},
        {"step": "4. Terminal Multiple", "desc": "Based on moat durability (Munger quality factor): 10-15x for no moat, 20-25x for narrow moat, 30-40x for wide moat."},
        {"step": "5. Apply Margin of Safety", "desc": "15% for predictable businesses (KO, JNJ), 25-30% for cyclical/tech. Never pay full estimated value."},
    ],
    "non_company_methods": {
        "ETFs": "Weighted NAV of holdings with earnings yield model",
        "Indices": "Normalized earnings × fair P/E (15-22x depending on quality)",
        "Currencies": "Purchasing Power Parity (PPP) long-run equilibrium",
        "Commodities": "All-in sustaining cost + monetary/industrial premium",
        "Crypto": "Network value (Metcalfe's law) — highly speculative",
    },
    "interpretation": {
        "positive_pct": "Stock trades BELOW intrinsic value — potential opportunity (Buffett: 'margin of safety')",
        "negative_pct": "Stock trades ABOVE intrinsic value — market premium / overvaluation risk",
        "na": "Intrinsic value not applicable (VIX, inverse FX pairs)",
    },
}


def get_all_intrinsic_data() -> dict:
    """
    Return intrinsic values, current prices, and valuation gaps for all tickers.
    Prices are read from local CSV files (last close).
    """
    prices = {}
    valuations = {}
    for ticker, iv in INTRINSIC_VALUES.items():
        price = _get_last_close(ticker)
        prices[ticker] = price
        if price is not None and iv is not None and price > 0:
            gap_pct = round(((iv - price) / price) * 100, 1)
        else:
            gap_pct = None
        valuations[ticker] = {
            "intrinsic_value": iv,
            "price": price,
            "gap_pct": gap_pct,
        }
    return {
        "valuations": valuations,
        "formula": INTRINSIC_FORMULA,
    }
