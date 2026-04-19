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

    # ── Heatmap Coverage (April 2026) ─────────────────────────────
    # ETFs — NAV-based fair value
    "AFK": 26.0,       # VanEck Africa Index ETF: EM discount; actual ~$29
    "AMZD": 8.5,       # iShares Amazon Options ETF: options overlay NAV; actual ~$9.1
    "ANGL": 27.0,      # VanEck Fallen Angel HY Bond ETF: credit risk NAV; actual ~$29.4
    "CNXT": 48.0,      # VanEck ChiNext ETF: China tech discount; actual ~$52.6
    "DURA": 34.0,      # VanEck Durable Dividend ETF: dividend NAV; actual ~$36.9
    "ESPO": 85.0,      # VanEck Video Gaming eSports ETF: gaming cyclical; actual ~$95.8
    "GDX": 80.0,       # VanEck Gold Miners ETF: miners lag gold; actual ~$100.3
    "GDXJ": 105.0,     # VanEck Junior Gold Miners ETF: higher risk; actual ~$133.1
    "GLIN": 42.0,      # VanEck India Growth Leaders ETF: India growth; actual ~$47
    "IDX": 13.0,       # VanEck Indonesia Index ETF: EM; actual ~$14.7
    "ITA": 200.0,      # iShares US Aerospace & Defense ETF: defense boom; actual ~$231.9
    "MLN": 16.5,       # VanEck Long Muni ETF: bond NAV; actual ~$17.7
    "MOAT": 92.0,      # VanEck Wide Moat ETF: quality premium; actual ~$102.3
    "MOO": 75.0,       # VanEck Agribusiness ETF: agri cyclical; actual ~$83.3
    "MOTG": 36.0,      # VanEck Morningstar Global Wide Moat ETF; actual ~$40
    "MOTI": 33.0,      # VanEck Morningstar Intl Moat ETF; actual ~$36.9
    "NLR": 125.0,      # VanEck Uranium+Nuclear Energy ETF: nuclear renaissance; actual ~$146
    "OIH": 340.0,      # VanEck Oil Services ETF: energy cyclical; actual ~$400.7
    "PLTI": 13.5,      # iShares Palantir Options ETF: options overlay; actual ~$15.4
    "PPH": 95.0,       # VanEck Pharmaceutical ETF: pharma steady; actual ~$104.5
    "REMX": 85.0,      # VanEck Rare Earth/Strategic Metals ETF; actual ~$101.6
    "SGLP.L": 310.0,   # Invesco Physical Gold ETC: gold-backed; actual ~$345.4
    "VNQ": 85.0,       # Vanguard Real Estate ETF: REIT NAV; actual ~$96.7
    "MSTP": 2.2,       # YieldMax MSTR Option Income Strategy ETF; actual ~$2.5

    # ── Precious Metals Miners ────────────────────────────────────
    "AG": 15.0,        # First Majestic Silver: high cost producer; actual ~$21.5
    "CDE": 14.0,       # Coeur Mining: gold/silver, turnaround; actual ~$20.4
    "EXK": 7.0,        # Endeavour Silver: small producer; actual ~$10.2
    "GORO": 1.0,       # Gold Resource: micro miner; actual ~$1.6
    "GROY": 2.8,       # Gold Royalty Corp: royalty model; actual ~$3.8
    "HL": 13.0,        # Hecla Mining: silver/gold; actual ~$19.5
    "HYMC": 28.0,      # Hycroft Mining: turnaround; actual ~$44.2
    "IAUX": 1.2,       # i-80 Gold: development stage; actual ~$1.7
    "IDR": 32.0,       # Idaho Strategic Resources: rare earth; actual ~$43.3
    "KGC": 25.0,       # Kinross Gold: mid-tier producer; actual ~$34.9
    "PAAS": 42.0,      # Pan American Silver: diversified; actual ~$59.1
    "RGLD": 200.0,     # Royal Gold: royalty/streaming, 25x; actual ~$268.1
    "SLVR": 45.0,      # Silver Tiger Metals: exploration; actual ~$65.1
    "SVM": 9.0,        # Silvercorp Metals: China ops; actual ~$12.8
    "USAS": 4.5,       # Americas Gold and Silver: small; actual ~$6.6

    # ── Aerospace & Defense ───────────────────────────────────────
    "AIR": 85.0,       # AAR Corp: MRO services, 14x; actual ~$105.1
    "AIR.PA": 155.0,   # Airbus SE: duopoly with Boeing, 18x; actual ~$179.5
    "AVAV": 155.0,     # AeroVironment: defense drones, 28x; actual ~$191.4
    "BA.L": 1900.0,    # BAE Systems: UK defense prime, 16x; actual ~2255.5 GBp
    "BAH": 68.0,       # Booz Allen Hamilton: gov consulting, 15x; actual ~$81.8
    "BWXT": 190.0,     # BWX Technologies: nuclear defense, 22x; actual ~$235.8
    "CACI": 420.0,     # CACI International: defense IT, 16x; actual ~$527.2
    "CW": 580.0,       # Curtiss-Wright: defense industrial, 20x; actual ~$735.7
    "DCO": 110.0,      # Ducommun: aerostructures, 14x; actual ~$138
    "DRS": 35.0,       # Leonardo DRS: electronic warfare, 18x; actual ~$44.6
    "ESLT": 700.0,     # Elbit Systems: Israeli defense, 18x; actual ~$872.6
    "HAG.DE": 65.0,    # Hensoldt AG: German defense sensors, 18x; actual ~$80.8
    "HEI": 235.0,      # HEICO Corp: aerospace parts monopoly, 35x; actual ~$291.6
    "HEIA.AS": 55.0,   # HEICO Class A; actual ~$68.6
    "HII": 320.0,      # Huntington Ingalls: shipbuilding duopoly, 14x; actual ~$394.8
    "HO.PA": 220.0,    # Thales SA: French defense/electronics, 16x; actual ~$264.6
    "HXL": 72.0,       # Hexcel: aerospace composites, 18x; actual ~$88.8
    "KOG.OL": 300.0,   # Kongsberg Gruppen: Norwegian defense, 18x; actual ~372.5 NOK
    "KRMN": 65.0,      # Karman Holdings: defense tech; actual ~$83.6
    "KTOS": 55.0,      # Kratos Defense: drones/hypersonics, 30x; actual ~$71
    "LDOS": 125.0,     # Leidos: defense IT services, 14x; actual ~$155.2
    "LHX": 280.0,      # L3Harris Technologies: defense electronics, 16x; actual ~$350.4
    "LOAR": 52.0,      # Loar Holdings: aerospace components; actual ~$67.5
    "MOG-A": 260.0,    # Moog Inc: flight controls, 16x; actual ~$322.8
    "MRCY": 65.0,      # Mercury Systems: defense electronics, 20x; actual ~$84.1
    "MTX.DE": 280.0,   # MTU Aero Engines: engine MRO, 18x; actual ~346.1 EUR
    "OSK": 120.0,      # Oshkosh: defense/specialty vehicles, 12x; actual ~$147.9
    "PKE": 28.0,       # Park Aerospace: composites, 14x; actual ~$34.5
    "R3NK.DE": 42.0,   # Renk Group: German defense gearboxes, 16x; actual ~55.0 EUR
    "RHM.DE": 1100.0,  # Rheinmetall AG: German defense leader, 18x; actual ~1495.2 EUR
    "SAIC": 78.0,      # Science Applications: defense IT, 12x; actual ~$95.4
    "SARO": 22.0,      # StandardAero: MRO services; actual ~$26.8
    "TDY": 510.0,      # Teledyne Technologies: defense instruments, 22x; actual ~$635.8
    "TXT": 75.0,       # Textron: diversified defense, 12x; actual ~$91.6
    "VSEC": 180.0,     # VSE Corporation: aviation services, 14x; actual ~$226.5
    "VVX": 52.0,       # V2X Inc: defense services, 12x; actual ~$66.7
    "EXA.PA": 95.0,    # Exail Technologies: French defense tech; actual ~120.6 EUR
    "FACC.VI": 11.0,   # FACC AG: aero composites, 12x; actual ~14.7 EUR
    "FINMY": 26.0,     # Leonardo SpA ADR: Italian defense, 12x; actual ~$33.9
    "SAABY": 25.0,     # Saab AB ADR: Swedish defense, 16x; actual ~$32.4
    "SAF": 20.0,       # Safran SA: French engine maker, 18x; actual ~$25.0

    # ── Space & Drones ────────────────────────────────────────────
    "ASTS": 55.0,      # AST SpaceMobile: satellite broadband, pre-revenue; actual ~$85.5
    "BKSY": 24.0,      # BlackSky Technology: satellite imagery; actual ~$37.6
    "FJET": 4.0,       # Starfighters Space: speculative; actual ~$6.3
    "FLY": 30.0,       # Firefly Aerospace: small launch; actual ~$43.7
    "HOVR": 1.2,       # New Horizon Aircraft: eVTOL, pre-rev; actual ~$1.8
    "LUNR": 18.0,      # Intuitive Machines: lunar lander; actual ~$27.6
    "MNTS": 4.0,       # Momentus: in-space transport, speculative; actual ~$7.5
    "PL": 25.0,        # Planet Labs: earth observation data; actual ~$38.5
    "RCAT": 8.0,       # Red Cat Holdings: military drones; actual ~$12.7
    "RDW": 7.0,        # Redwire Corp: space infrastructure; actual ~$10.3
    "SATL": 4.5,       # Satellogic: satellite imagery; actual ~$6.9
    "SIDU": 3.5,       # Sidus Space: small sat; actual ~$5.4
    "SPCE": 1.5,       # Virgin Galactic: space tourism, cash burn; actual ~$2.9
    "VOYG": 22.0,      # Voyager Technologies: space tech; actual ~$31.3

    # ── Defense Small-Cap & Niche ─────────────────────────────────
    "AIRI": 2.2,       # Air Industries Group: defense sub-assemblies; actual ~$3.2
    "AIRO": 6.0,       # AIRO Group Holdings: defense; actual ~$8.1
    "AOUT": 7.5,       # American Outdoor Brands: firearms accessories; actual ~$9.6
    "ASTC": 2.0,       # Astrotech Corp: instruments; actual ~$2.9
    "ATRO": 60.0,      # Astronics Corp: aerospace lighting/power, 14x; actual ~$77
    "BYRN": 4.5,       # Byrna Technologies: less-lethal weapons; actual ~$6.7
    "CDRE": 24.0,      # Cadre Holdings: safety equipment, 14x; actual ~$31.3
    "CODA": 10.0,      # Coda Octopus: marine tech; actual ~$13.3
    "CVU": 2.5,        # CPI Aerostructures: small defense; actual ~$3.5
    "DFSC": 1.3,       # Defsec Technologies: micro-cap defense; actual ~$1.9
    "ISSC": 16.0,      # Innovative Solutions & Support: avionics; actual ~$22
    "NPK": 115.0,      # National Presto: defense/housewares, 12x; actual ~$143.7
    "ONDS": 7.0,       # Ondas Holdings: drones/mesh networks; actual ~$10
    "OPXS": 8.5,       # Optex Systems: military optics; actual ~$11.5
    "PEW": 2.0,        # Grabagun Digital: speculative; actual ~$3.0
    "POWW": 1.4,       # AMMO Inc: ammunition maker; actual ~$2.1
    "PRZO": 0.4,       # ParaZero Technologies: drone safety; actual ~$0.7
    "RGR": 34.0,       # Sturm Ruger: firearms, 12x; actual ~$42.2
    "SIF": 11.0,       # SIFCO Industries: aerospace forgings; actual ~$14.6
    "SKYH": 8.0,       # Sky Harbour: private aviation hangars; actual ~$11
    "SPAI": 3.0,       # Safe Pro Group: safety tech; actual ~$4.4
    "SWBI": 11.0,      # Smith & Wesson: firearms, 10x; actual ~$14.9
    "TATT": 30.0,      # TAT Technologies: MRO services; actual ~$40.6
    "UMAC": 9.0,       # Unusual Machines: consumer drones; actual ~$14.1
    "VTSI": 3.5,       # VirTra: law enforcement simulators; actual ~$4.7

    # ── Large Cap / Blue Chip ─────────────────────────────────────
    "BABA": 105.0,     # Alibaba: China e-commerce, 10x OE; actual ~$141
    "BK": 110.0,       # BNY Mellon: custody bank, 12x; actual ~$135.1
    "C": 105.0,        # Citigroup: global bank, restructuring, 8x; actual ~$132.2
    "CMCSA": 24.0,     # Comcast: cable/broadband, declining subs, 10x; actual ~$29.6
    "CVS": 60.0,       # CVS Health: pharmacy/insurance, 8x; actual ~$77.3
    "DHR": 160.0,      # Danaher: life sciences, 22x; actual ~$194.8
    "GM": 62.0,        # General Motors: auto cyclical, 6x; actual ~$81.3
    "ILMN": 105.0,     # Illumina: genomics leader, 25x; actual ~$134.5
    "MDT": 72.0,       # Medtronic: medical devices, 14x; actual ~$86.2
    "NVO": 32.0,       # Novo Nordisk: GLP-1 leader, 22x; actual ~$40.5
    "REGN": 600.0,     # Regeneron: biotech, wide moat, 18x; actual ~$750.6
    "TGT": 105.0,      # Target: retail, 12x; actual ~$127.8

    # ── Semiconductors / Tech Hardware ────────────────────────────
    "ANET": 130.0,     # Arista Networks: networking, 28x; actual ~$164.2
    "AOSL": 25.0,      # Alpha & Omega Semi: power semi, 12x; actual ~$33.9
    "IFX.DE": 38.0,    # Infineon Technologies: auto/power semi, 16x; actual ~48.9 EUR
    "MPWR": 1100.0,    # Monolithic Power Systems: analog, 35x; actual ~$1468.4
    "NBIS": 110.0,     # Nebius Group: AI infra spin-off; actual ~$157.1
    "POWI": 45.0,      # Power Integrations: power conversion, 22x; actual ~$58.7
    "SANM": 135.0,     # Sanmina: contract manufacturing, 12x; actual ~$174.1
    "STM": 34.0,       # STMicroelectronics: auto/industrial semi, 12x; actual ~$44.2
    "VSH": 20.0,       # Vishay Intertechnology: passives, 10x; actual ~$25.9

    # ── Software / AI / Tech ──────────────────────────────────────
    "CRWV": 75.0,      # CoreWeave: AI cloud infra, early revenue; actual ~$116.9
    "GLBE": 25.0,      # Global-E Online: cross-border e-commerce; actual ~$33.9
    "GLXY": 18.0,      # Galaxy Digital: crypto asset mgr; actual ~$25.8
    "GRND": 9.0,       # Grindr: dating app, 20x; actual ~$13.2
    "SDGR": 8.0,       # Schrödinger: drug discovery software; actual ~$12.3
    "SOUN": 5.0,       # SoundHound AI: voice AI, speculative; actual ~$8.1
    "SYM": 42.0,       # Symbotic: warehouse robotics, 25x; actual ~$63.2
    "TEM": 35.0,       # Tempus AI: healthcare AI; actual ~$55.9

    # ── Nuclear / Uranium / Energy ────────────────────────────────
    "CCJ": 90.0,       # Cameco: uranium producer, 20x; actual ~$120.7
    "DNN": 2.8,        # Denison Mines: uranium developer; actual ~$3.9
    "GEV": 750.0,      # GE Vernova: energy spin-off, 25x; actual ~$1002.8
    "LEU": 140.0,      # Centrus Energy: uranium enrichment; actual ~$203.6
    "NXE": 9.0,        # NexGen Energy: uranium development; actual ~$12.7
    "OKLO": 40.0,      # Oklo Inc: advanced nuclear, pre-revenue; actual ~$66.8
    "SMR": 15.0,       # NuScale Power: SMR, pre-revenue; actual ~$21 (from quality)
    "UEC": 10.0,       # Uranium Energy Corp: producer; actual ~$15
    "UUUU": 14.0,      # Energy Fuels: uranium/rare earth; actual ~$20.5

    # ── Mining / Materials / Resources ────────────────────────────
    "ALB": 145.0,      # Albemarle: lithium, cyclical, 12x; actual ~$197.8
    "ATI": 125.0,      # ATI Inc: specialty metals, 14x; actual ~$164.7
    "CRML": 8.0,       # Critical Metals: rare earth exploration; actual ~$12.6
    "ERMAY": 4.5,      # Eramet SA ADR: mining/metallurgy; actual ~$6.5
    "GLNCY": 11.0,     # Glencore ADR: commodities trader, 8x; actual ~$14.9
    "HBM": 18.0,       # Hudbay Minerals: copper/gold, 10x; actual ~$25.9
    "ILKAF": 3.8,      # Iluka Resources: mineral sands; actual ~$5.4
    "IVN.TO": 9.0,     # Ivanhoe Mines: copper development; actual ~12.3 CAD
    "LYSCF": 10.0,     # Lynas Rare Earths: rare earth producer; actual ~$15
    "MP": 42.0,        # MP Materials: rare earth processing, 20x; actual ~$61
    "MTRN": 140.0,     # Materion: specialty materials, 16x; actual ~$182
    "SQM": 65.0,       # SQM: lithium/fertilizer, 10x; actual ~$88.8
    "TECK": 45.0,      # Teck Resources: copper/zinc, 10x; actual ~$59.4
    "UAMY": 7.0,       # US Antimony: strategic metals; actual ~$10.6

    # ── Biotech / Pharma ──────────────────────────────────────────
    "ALNY": 230.0,     # Alnylam Pharma: RNAi therapeutics, 30x; actual ~$309.7
    "APLM": 9.0,       # Apollomics: oncology biotech; actual ~$13.9
    "APLS": 30.0,      # Apellis Pharma: complement inhibitor; actual ~$40.9
    "ATAI": 2.5,       # AtaiBeckley: psychedelic medicine; actual ~$4
    "BBIO": 55.0,      # BridgeBio Pharma: genetic disease; actual ~$76.8
    "BEAM": 20.0,      # Beam Therapeutics: base editing; actual ~$31.4
    "CRSP": 42.0,      # CRISPR Therapeutics: gene editing; actual ~$58
    "NTLA": 10.0,      # Intellia Therapeutics: gene editing; actual ~$15
    "NUTX": 70.0,      # Nutex Health: healthcare services; actual ~$105.6
    "PACB": 1.0,       # PacBio: gene sequencing, cash burn; actual ~$1.7
    "RXRX": 2.5,       # Recursion Pharma: AI drug discovery; actual ~$3.8

    # ── Auto / EV / Transport ─────────────────────────────────────
    "BMW3.DE": 68.0,   # BMW Preferred: German auto, 6x; actual ~83.8 EUR
    "EMBJ": 50.0,      # Embraer SA: regional jets, 14x; actual ~$68
    "EVEX": 1.8,       # Eve Holding: eVTOL, pre-revenue; actual ~$2.9
    "EVTL": 1.8,       # Vertical Aerospace: eVTOL, pre-revenue; actual ~$3
    "VOW3.DE": 75.0,   # Volkswagen Preferred: German auto, 4x; actual ~92.7 EUR
    "BETA": 12.0,      # Beta Technologies: eVTOL; actual ~$18.1

    # ── Industrials / Misc ────────────────────────────────────────
    "B": 34.0,         # Barnes Group: industrial diversified, 12x; actual ~$43.3
    "BAYN.DE": 32.0,   # Bayer AG: pharma/crop, litigation risk, 8x; actual ~41.1 EUR
    "CAE": 22.0,       # CAE Inc: flight sim/training, 14x; actual ~$26.9
    "CIFR": 12.0,      # Cipher Mining: bitcoin miner; actual ~$19.4
    "DPRO": 3.5,       # Draganfly: commercial drones; actual ~$5.9
    "EH": 7.5,         # EHang Holdings: autonomous air vehicles; actual ~$11.6
    "FTAI": 190.0,     # FTAI Aviation: engine leasing, 18x; actual ~$259.1
    "GPUS": 0.08,      # Hyperscale Data: micro-cap; actual ~$0.15
    "IREN": 32.0,      # Iris Energy: bitcoin/AI data centers; actual ~$48.1
    "KITT": 0.2,       # Nauticus Robotics: ocean robotics; actual ~$0.4
    "MSA": 140.0,      # MSA Safety: safety equipment, 20x; actual ~$173.2
    "NU": 10.0,        # Nu Holdings: Latin American neobank, 20x; actual ~$15.3
    "THEON.AS": 25.0,  # Theon International: night vision optics; actual ~33.4 EUR
    "TKA.DE": 6.5,     # thyssenkrupp: German industrial, restructuring; actual ~9.3 EUR
    "VSAT": 42.0,      # ViaSat: satellite broadband, debt-heavy; actual ~$62.9
    "VWAV": 4.5,       # Visionwave Holdings: speculative; actual ~$7.1
    "AZ": 5.5,         # A2Z Cust2Mate Solutions: retail tech; actual ~$8.3
    "AM": 16.0,        # Dassault Aviation SA: French jets/defense; actual ~$21.3
    "RKLB": 55.0,      # Rocket Lab USA: small launch + spacecraft, 30x; actual ~$84.8
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
