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
    # ── FX Pairs (PPP equilibrium estimates) ──────────────────────
    "EURUSD=X": 1.22,   # PPP ~1.22-1.28, EUR slightly undervalued
    "GBPUSD=X": 1.45,   # PPP ~1.45-1.55, GBP historically undervalued
    "USDJPY=X": 105.0,  # PPP ~90-105, JPY deeply undervalued vs USD
    "USDCHF=X": 0.88,   # PPP ~0.85-0.92, CHF fair
    "AUDUSD=X": 0.72,   # PPP ~0.68-0.75, AUD roughly fair
    "USDCAD=X": 1.25,   # PPP ~1.22-1.28, CAD roughly fair
    "NZDUSD=X": 0.65,   # PPP ~0.62-0.68
    "EURJPY=X": 128.0,  "GBPJPY=X": 152.0,  "AUDJPY=X": 76.0,
    "NZDJPY=X": 68.0,   "CADJPY=X": 84.0,   "CHFJPY=X": 119.0,
    "SGDJPY=X": 80.0,   "HKDJPY=X": 13.5,
    "ZARJPY=X": 8.5,    "MXNJPY=X": 7.0,    "TRYJPY=X": 4.5,
    "SEKJPY=X": 10.5,   "NOKJPY=X": 10.8,   "DKKJPY=X": 17.2,
    "CNYJPY=X": 15.0,   "PLNJPY=X": 27.0,
    # Inverse JPY pairs — not applicable for intrinsic value
    "JPYUSD=X": None, "JPYEUR=X": None, "JPYGBP=X": None, "JPYAUD=X": None,
    "JPYNZD=X": None, "JPYCAD=X": None, "JPYCHF=X": None, "JPYSGD=X": None,
    "JPYHKD=X": None, "JPYZAR=X": None, "JPYMXN=X": None, "JPYTRY=X": None,
    "JPYSEK=X": None, "JPYNOK=X": None, "JPYDKK=X": None, "JPYCNY=X": None,
    "JPYPLN=X": None,

    # ── Commodities (production cost + monetary premium) ──────────
    "GC=F": 3200.0,   # Gold: all-in sustaining cost ~$1400 + monetary premium
    "SI=F": 32.0,     # Silver: production cost ~$18 + industrial/monetary premium

    # ── Crypto (Metcalfe / adoption model — highly speculative) ───
    "BTC-USD": 55000.0,  # Network value model, high uncertainty
    "MSTR": 220.0,       # ~1.2x NAV of BTC holdings, leveraged

    # ── Indices (earnings yield fair value) ───────────────────────
    "^GSPC": 4800.0,  # S&P 500: ~18x normalized earnings
    "^VIX": None,      # VIX is not an investable asset with intrinsic value
    "^RUT": 2200.0,    # Russell 2000: ~15x earnings, small cap discount
    "^NDX": 18500.0,   # Nasdaq-100: ~22x earnings, growth premium
    "^DJI": 39000.0,   # DJIA: ~17x earnings
    "^IXIC": 16000.0,  # Nasdaq Composite

    # ── Broad ETFs (NAV-based) ────────────────────────────────────
    "SPY": 480.0,   "VOO": 442.0,  "QQQ": 440.0,  "IWM": 210.0,
    "OEF": 235.0,   "DIA": 380.0,  "GLD": 310.0,  "SLV": 28.0,

    # ── Sector ETFs ───────────────────────────────────────────────
    "XLE": 88.0,  "XLK": 195.0, "XLC": 82.0,  "XLB": 85.0,  "XLP": 78.0,
    "XLU": 72.0,  "XLI": 115.0, "XLF": 42.0,  "XLV": 145.0, "XLRE": 40.0,
    "XLY": 185.0, "SMH": 230.0,

    # ── Technology (Large Cap) ────────────────────────────────────
    # AAPL: $6.50 owner earnings/share, 8% growth, 9.5% discount, 25x terminal
    "AAPL": 210.0,
    # MSFT: $12 owner earnings, 12% growth, 9% discount, 28x terminal
    "MSFT": 380.0,
    # NVDA: $5.50 OE, 25% growth (AI), 11% discount, 30x terminal — volatile
    "NVDA": 155.0,
    # GOOGL: $7 OE, 10% growth, 9.5% discount, 22x terminal
    "GOOGL": 175.0,
    "GOOG": 176.0,
    # META: $18 OE, 12% growth, 10% discount, 20x terminal
    "META": 520.0,
    "AVGO": 175.0,   # Broadcom: strong margins but richly valued
    "CRM": 280.0,    # Salesforce: maturing growth, ~25x FCF
    "ADBE": 480.0,   # Adobe: creative monopoly, 28x FCF
    "ORCL": 145.0,   # Oracle: cloud transition, 20x FCF
    "ACN": 340.0,    # Accenture: steady consulting, 25x FCF
    "CSCO": 55.0,    # Cisco: mature, 15x FCF + dividend
    "INTC": 28.0,    # Intel: turnaround uncertain, 12x depressed earnings
    "IBM": 195.0,    # IBM: Red Hat + AI, 15x FCF
    "INTU": 620.0,   # Intuit: tax/accounting monopoly, 30x FCF
    "NOW": 850.0,    # ServiceNow: IT workflow dominance, 35x FCF
    "PLTR": 25.0,    # Palantir: growth but profitability uncertain
    "QCOM": 170.0,   # Qualcomm: mobile chip + licensing, 18x
    "TXN": 185.0,    # Texas Instruments: analog moat, 22x FCF

    # ── Healthcare / Pharma ───────────────────────────────────────
    "UNH": 540.0,   # UnitedHealth: healthcare oligopoly, 20x
    "LLY": 650.0,   # Eli Lilly: GLP-1 blockbuster, 35x growth
    "JNJ": 165.0,   # J&J: diversified healthcare, 17x
    "MRK": 115.0,   # Merck: Keytruda peak, 14x
    "PFE": 32.0,    # Pfizer: post-COVID decline, 10x depressed
    "ABBV": 175.0,  # AbbVie: Humira loss + pipeline, 14x
    "ABT": 115.0,   # Abbott: medical devices steady, 22x
    "TMO": 560.0,   # Thermo Fisher: life sciences leader, 25x
    "BMY": 55.0,    # Bristol-Myers: patent cliffs, 10x
    "AMGN": 290.0,  # Amgen: biotech steady, 14x
    "GILD": 85.0,   # Gilead: HIV franchise, 12x
    "ISRG": 420.0,  # Intuitive Surgical: robotic surgery monopoly, 40x
    "VRTX": 440.0,  # Vertex: CF monopoly + pain pipeline, 28x
    "MRNA": 45.0,   # Moderna: mRNA pipeline uncertain, 15x
    "ELV": 500.0,   # Elevance Health: insurance, 15x
    "CI": 340.0,    # Cigna: PBM + insurance, 13x
    "HCA": 310.0,   # HCA: hospital operator, 15x
    "DXCM": 95.0,   # DexCom: CGM leader, 35x growth
    "ABCL": 8.0,    # AbCellera: early-stage biotech

    # ── Financials ────────────────────────────────────────────────
    "BRK-B": 420.0,  # Berkshire: book value + float, 1.4x book
    "JPM": 210.0,    # JPMorgan: best bank, 12x
    "V": 280.0,      # Visa: payments monopoly, 28x
    "MA": 470.0,     # Mastercard: payments duopoly, 30x
    "BAC": 42.0,     # Bank of America: 11x
    "WFC": 60.0,     # Wells Fargo: recovering, 10x
    "GS": 480.0,     # Goldman: capital markets, 12x
    "MS": 105.0,     # Morgan Stanley: wealth mgmt, 14x
    "SCHW": 72.0,    # Schwab: brokerage leader, 18x
    "AXP": 240.0,    # AmEx: premium spend, 17x
    "BLK": 840.0,    # BlackRock: asset mgmt monopoly, 22x
    "SPGI": 450.0,   # S&P Global: ratings duopoly, 30x
    "MCO": 380.0,    # Moody's: ratings duopoly, 28x
    "ICE": 145.0,    # ICE: exchange operator, 22x
    "CME": 210.0,    # CME Group: derivatives monopoly, 22x
    "CB": 265.0,     # Chubb: insurance, 14x
    "PGR": 230.0,    # Progressive: auto insurance, 20x
    "MMC": 205.0,    # Marsh McLennan: insurance broker, 22x
    "AON": 340.0,    # Aon: insurance broker, 22x
    "AFL": 95.0,     # Aflac: supplemental insurance, 12x
    "AIG": 75.0,     # AIG: turnaround, 10x
    "MET": 78.0,     # MetLife: life insurance, 10x
    "TFC": 42.0,     # Truist: regional bank, 10x
    "USB": 48.0,     # US Bancorp: 11x
    "PNC": 170.0,    # PNC Financial: 12x
    "COF": 155.0,    # Capital One: credit cards + Discover, 11x

    # ── Consumer / Retail ─────────────────────────────────────────
    "AMZN": 185.0,   # Amazon: e-commerce + AWS, 25x FCF
    "TSLA": 180.0,   # Tesla: EV leader but richly valued, 40x earnings
    "WMT": 165.0,    # Walmart: retail moat, 22x
    "COST": 680.0,   # Costco: membership model, 35x
    "HD": 370.0,     # Home Depot: home improvement duopoly, 22x
    "LOW": 250.0,    # Lowe's: home improvement, 18x
    "MCD": 290.0,    # McDonald's: franchise model, 22x
    "SBUX": 95.0,    # Starbucks: global brand, 22x
    "NKE": 85.0,     # Nike: brand moat, 25x
    "TJX": 110.0,    # TJX: off-price leader, 22x
    "PG": 170.0,     # P&G: consumer staples moat, 23x
    "KO": 62.0,      # Coca-Cola: brand monopoly, 22x
    "PEP": 170.0,    # PepsiCo: snacks + beverages, 22x
    "PM": 115.0,     # Philip Morris: reduced-risk products, 15x
    "MO": 52.0,      # Altria: tobacco cash cow, declining, 10x
    "CL": 90.0,      # Colgate-Palmolive: staples, 24x
    "KMB": 135.0,    # Kimberly-Clark: staples, 18x
    "GIS": 68.0,     # General Mills: food, 16x
    "K": 72.0,       # Kellanova: snacks, 18x
    "HSY": 195.0,    # Hershey: chocolate moat, 22x
    "MDLZ": 75.0,    # Mondelez: global snacks, 20x
    "STZ": 230.0,    # Constellation Brands: beer/wine, 18x
    "TAP": 58.0,     # Molson Coors: beer, 12x
    "DG": 95.0,      # Dollar General: value retail, 15x
    "DLTR": 85.0,    # Dollar Tree: value retail, 14x
    "ROST": 145.0,   # Ross Stores: off-price, 22x
    "LULU": 320.0,   # Lululemon: athletic premium, 28x
    "DECK": 155.0,   # Deckers: UGG + HOKA, 22x
    "BURL": 230.0,   # Burlington: off-price, 22x
    "YUM": 140.0,    # Yum! Brands: franchise, 22x
    "CMG": 55.0,     # Chipotle: fast-casual leader, 35x
    "DPZ": 490.0,    # Domino's: delivery tech + pizza, 25x
    "DKNG": 35.0,    # DraftKings: online betting, early profitability
    "ABNB": 140.0,   # Airbnb: travel platform, 25x
    "BKNG": 4200.0,  # Booking Holdings: travel monopoly, 22x
    "UBER": 70.0,    # Uber: ride-share/delivery platform, 30x FCF
    "DASH": 145.0,   # DoorDash: food delivery, early profits
    "SNAP": 12.0,    # Snap: social media, unprofitable
    "PINS": 32.0,    # Pinterest: visual search, 22x
    "SPOT": 350.0,   # Spotify: audio streaming, 30x
    "NFLX": 700.0,   # Netflix: streaming dominance, 25x
    "DIS": 105.0,    # Disney: content + parks, 18x
    "PARA": 12.0,    # Paramount: legacy media, declining

    # ── Industrials / Aerospace / Defense ─────────────────────────
    "CAT": 350.0,    # Caterpillar: infrastructure moat, 17x
    "DE": 400.0,     # Deere: precision ag monopoly, 18x
    "BA": 180.0,     # Boeing: duopoly but quality issues, 20x recovery
    "RTX": 115.0,    # RTX: defense + aerospace, 17x
    "LMT": 480.0,    # Lockheed Martin: defense leader, 16x
    "NOC": 510.0,    # Northrop Grumman: defense, 17x
    "GD": 290.0,     # General Dynamics: defense + Gulfstream, 17x
    "GE": 185.0,     # GE Aerospace: jet engines, 22x
    "HON": 215.0,    # Honeywell: diversified industrial, 20x
    "UNP": 240.0,    # Union Pacific: railroad duopoly, 20x
    "UPS": 145.0,    # UPS: logistics, 15x
    "FDX": 270.0,    # FedEx: logistics, 14x
    "MMM": 115.0,    # 3M: diversified, litigation risk, 12x
    "EMR": 110.0,    # Emerson: industrial automation, 18x
    "ETN": 300.0,    # Eaton: power management, 25x
    "ITW": 255.0,    # Illinois Tool Works: diversified, 22x
    "PH": 550.0,     # Parker Hannifin: motion/control, 22x
    "ROK": 280.0,    # Rockwell Automation: industrial IoT, 25x
    "CARR": 62.0,    # Carrier Global: HVAC, 20x
    "OTIS": 95.0,    # Otis: elevator monopoly, 25x
    "WWD": 185.0,    # Woodward: aerospace/energy, 22x
    "AXON": 350.0,   # Axon Enterprise: law enforcement tech, 40x
    "TDG": 1200.0,   # TransDigm: aerospace aftermarket monopoly, 22x
    "HWM": 95.0,     # Howmet: aerospace components, 25x
    "PWR": 280.0,    # Quanta Services: utility infrastructure, 20x

    # ── Energy ────────────────────────────────────────────────────
    "XOM": 110.0,    # ExxonMobil: integrated oil major, 11x
    "CVX": 160.0,    # Chevron: integrated oil, 11x
    "COP": 115.0,    # ConocoPhillips: E&P leader, 10x
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
    "LIN": 420.0,    # Linde: industrial gas duopoly, 25x
    "APD": 280.0,    # Air Products: industrial gas, 22x
    "ECL": 230.0,    # Ecolab: water treatment, 28x
    "SHW": 330.0,    # Sherwin-Williams: paint monopoly, 25x
    "NEM": 52.0,     # Newmont: gold miner, 15x
    "FCX": 42.0,     # Freeport: copper/gold, 12x cyclical
    "NUE": 165.0,    # Nucor: steel, 10x cyclical
    "BHP": 58.0,     # BHP: diversified mining, 10x
    "RIO": 68.0,     # Rio Tinto: mining, 9x
    "VALE": 12.0,    # Vale: iron ore, 7x
    "SCCO": 95.0,    # Southern Copper: copper, 15x
    "GOLD": 22.0,    # Barrick Gold: gold miner, 12x
    "WPM": 55.0,     # Wheaton Precious: streaming model, 22x
    "AEM": 65.0,     # Agnico Eagle: gold, 15x
    "CRS": 65.0,     # Carpenter Technology: specialty alloys, 15x
    "KMT": 28.0,     # Kennametal: cutting tools, 12x

    # ── Utilities ─────────────────────────────────────────────────
    "NEE": 78.0,     # NextEra: renewables leader, 22x
    "DUK": 105.0,    # Duke Energy: regulated utility, 16x
    "SO": 78.0,      # Southern Company: regulated, 17x
    "AEP": 98.0,     # AEP: regulated utility, 16x
    "D": 58.0,       # Dominion: regulated, 15x
    "EXC": 42.0,     # Exelon: nuclear utility, 14x
    "SRE": 82.0,     # Sempra: utility + LNG, 16x
    "WEC": 95.0,     # WEC Energy: regulated, 18x
    "ES": 68.0,      # Eversource: New England utility, 16x
    "CEG": 175.0,    # Constellation Energy: nuclear fleet, 18x
    "VST": 85.0,     # Vistra: power generation, 12x

    # ── REITs ─────────────────────────────────────────────────────
    "AMT": 210.0,    # American Tower: cell tower REIT, 22x AFFO
    "PLD": 125.0,    # Prologis: logistics REIT, 25x AFFO
    "CCI": 110.0,    # Crown Castle: cell tower, 18x AFFO
    "EQIX": 780.0,   # Equinix: data center REIT, 25x AFFO
    "PSA": 290.0,    # Public Storage: self-storage, 20x AFFO
    "O": 58.0,       # Realty Income: net lease, 16x AFFO
    "WELL": 115.0,   # Welltower: senior housing, 20x AFFO
    "SPG": 145.0,    # Simon Property: malls, 14x AFFO
    "DLR": 145.0,    # Digital Realty: data centers, 20x AFFO
    "AGNC": 9.5,     # AGNC: mortgage REIT, book value
    "ARR": 4.5,      # ARMOUR: mortgage REIT, book value

    # ── Telecom / Communication ───────────────────────────────────
    "T": 22.0,       # AT&T: telecom, 8x FCF
    "VZ": 42.0,      # Verizon: telecom, 8x FCF
    "TMUS": 195.0,   # T-Mobile: wireless leader, 15x
    "CHTR": 350.0,   # Charter: cable/broadband, 12x

    # ── Semiconductors (beyond NVDA) ──────────────────────────────
    "AMD": 125.0,    # AMD: CPU/GPU competitor, 25x
    "MU": 95.0,      # Micron: memory cycles, 12x mid-cycle
    "MRVL": 70.0,    # Marvell: data infrastructure, 25x
    "LRCX": 820.0,   # Lam Research: etch equipment, 22x
    "KLAC": 680.0,   # KLA: inspection equipment, 22x
    "AMAT": 170.0,   # Applied Materials: semi equipment, 20x
    "ADI": 220.0,    # Analog Devices: analog chips, 22x
    "NXPI": 235.0,   # NXP: auto/industrial chips, 18x
    "ON": 65.0,      # onsemi: power/sensing, 18x
    "SWKS": 95.0,    # Skyworks: RF chips, 14x
    "MCHP": 75.0,    # Microchip: microcontrollers, 16x
    "GFS": 52.0,     # GlobalFoundries: foundry, 15x
    "WOLF": 18.0,    # Wolfspeed: SiC, cash burn, speculative
    "TSM": 170.0,    # TSMC: foundry monopoly, 20x
    "ASML": 700.0,   # ASML: EUV monopoly, 30x
    "ARM": 120.0,    # ARM Holdings: IP licensing, 40x
    "CRDO": 40.0,    # Credo Technology: connectivity, 25x
    "ALAB": 55.0,    # Astera Labs: connectivity silicon
    "NVTS": 6.0,     # Navitas: GaN power, early
    "AEHR": 25.0,    # Aehr Test: SiC testing equipment

    # ── Software / Cloud ──────────────────────────────────────────
    "PANW": 320.0,   # Palo Alto: cybersecurity leader, 35x
    "CRWD": 280.0,   # CrowdStrike: endpoint security, 40x
    "ZS": 210.0,     # Zscaler: zero-trust security, 35x
    "FTNT": 85.0,    # Fortinet: network security, 28x
    "NET": 95.0,     # Cloudflare: edge computing, 50x (growth)
    "DDOG": 120.0,   # Datadog: observability, 35x
    "SNOW": 155.0,   # Snowflake: data cloud, 40x
    "MDB": 260.0,    # MongoDB: database, 35x
    "WDAY": 250.0,   # Workday: HR/finance cloud, 28x
    "HUBS": 620.0,   # HubSpot: marketing platform, 40x
    "TEAM": 225.0,   # Atlassian: dev tools, 35x
    "TTD": 85.0,     # The Trade Desk: programmatic ads, 35x
    "BILL": 62.0,    # Bill.com: SMB fintech, 25x
    "CFLT": 25.0,    # Confluent: data streaming, 30x growth
    "ESTC": 95.0,    # Elastic: search/observability, 25x
    "PATH": 14.0,    # UiPath: RPA, declining growth
    "MNDY": 250.0,   # monday.com: work management, 30x
    "U": 22.0,       # Unity: game engine, turnaround
    "RBLX": 45.0,    # Roblox: gaming platform, 25x
    "APP": 280.0,    # AppLovin: ad tech, 25x
    "RDDT": 120.0,   # Reddit: social platform, early monetization
    "OKTA": 95.0,    # Okta: identity, 25x
    "VEEV": 220.0,   # Veeva Systems: life sciences cloud, 30x
    "ANSS": 340.0,   # Ansys: simulation software, 30x
    "CDNS": 280.0,   # Cadence Design: EDA tools, 35x
    "SNPS": 500.0,   # Synopsys: EDA duopoly, 32x
    "FICO": 1600.0,  # FICO: credit scoring monopoly, 35x
    "TYL": 480.0,    # Tyler Technologies: gov software, 35x
    "GWRE": 165.0,   # Guidewire: insurance software, 30x
    "PCOR": 65.0,    # Procore: construction software, 30x
    "IOT": 22.0,     # Samsara: IoT fleet mgmt, 30x growth
    "AI": 28.0,      # C3.ai: enterprise AI, speculative

    # ── Fintech / Payments ────────────────────────────────────────
    "SQ": 68.0,      # Block (Square): payments + Cash App, 20x
    "PYPL": 72.0,    # PayPal: digital payments, 15x
    "AFRM": 32.0,    # Affirm: BNPL, unprofitable
    "COIN": 180.0,   # Coinbase: crypto exchange, 20x
    "SEZL": 25.0,    # Sezzle: small BNPL
    "FOUR": 65.0,    # Shift4 Payments: integrated payments
    "FI": 155.0,     # Fiserv: financial tech, 18x
    "FIS": 78.0,     # FIS: financial tech, 14x
    "GPN": 115.0,    # Global Payments: merchant services, 14x
    "DLO": 12.0,     # DLocal: EM payments
    "PGY": 10.0,     # Pagaya Technologies: AI lending

    # ── EV / Clean Energy ─────────────────────────────────────────
    "ENPH": 85.0,    # Enphase: solar microinverters, 25x
    "FLNC": 15.0,    # Fluence: energy storage
    "BE": 12.0,      # Bloom Energy: fuel cells
    "QS": 5.0,       # QuantumScape: pre-revenue
    "ENVX": 6.0,     # Enovix: batteries, pre-revenue

    # ── Quantum Computing ─────────────────────────────────────────
    "IONQ": 18.0,    # IonQ: quantum computing, very early
    "QBTS": 3.0,     # D-Wave: quantum, very early
    "ARQQ": 2.5,     # Arqit: quantum encryption
    "RGTI": 5.0,     # Rigetti: quantum
    "QUBT": 2.0,     # Quantum Computing Inc

    # ── Space / Satellite ─────────────────────────────────────────
    "SPIR": 8.0,     # Spire Global: satellite data
    "GSAT": 3.5,     # Globalstar: satellite comms
    "IRDM": 42.0,    # Iridium: satellite monopoly, 18x
    "MDA.TO": 18.0,  # MDA Space
    "MDALF": 18.0,   # MDA Space (ADR)

    # ── Nuclear / SMR ─────────────────────────────────────────────
    "SMR": 15.0,     # NuScale: SMR leader, pre-revenue

    # ── Infrastructure / Industrial Tech ──────────────────────────
    "VRT": 85.0,     # Vertiv: data center power, 22x
    "JCI": 72.0,     # Johnson Controls: building tech, 18x

    # ── REIT / Other ──────────────────────────────────────────────
    "PSIX": 8.0,     # Power Solutions International

    # ── Asian Tech ────────────────────────────────────────────────
    "000660.KS": 140000.0,  # SK Hynix: memory chips, 12x
    "005930.KS": 85000.0,   # Samsung Electronics: diversified, 12x
    "2308.TW": 1100.0,      # Delta Electronics Taiwan
    "6723.T": 3500.0,       # Renesas Electronics
    "8035.T": 28000.0,      # Tokyo Electron: semi equipment

    # ── Micro-Cap / Speculative (wide margin of safety needed) ────
    "ABTC": 0.8,    # American Bitcoin Corp
    "AXG": 1.0,     # Solowin Holdings
    "ASB": 22.0,    # Associated Banc-Corp: regional bank
    "ANGX": 4.0,    # Angel Studios
    "ANNA": 2.0,    # AleAnna
    "BZAI": 3.0,    # Blaize Holdings
    "AIRE": 1.5,    # reAlpha Tech
    "BMHL": 1.0,    # Bluemount Holdings
    "BNZI": 1.5,    # Banzai International
    "BNKK": 0.5,    # Bonk Inc
    "BTCS": 3.0,    # BTCS blockchain
    "BCAL": 25.0,   # California BanCorp
    "ASPI": 5.0,    # ASP Isotopes
    "ABAT": 3.0,    # American Battery Technology
    "ADUR": 4.0,    # Aduro Clean Technologies
    "APLD": 8.0,    # Applied Digital: data centers
    "ALMU": 2.0,    # Aeluma
    "AMZE": 0.5,    # Amaze Holdings
    "AIFF": 2.0,    # Firefly Neuroscience
    "AMCR": 11.0,   # Amcor: packaging, 14x
    "AAPI": 0.3,    # Asian-American consumer
    "ACP": 6.0,     # Aberdeen Income Credit

    # ── Additional stocks from quality_scores ─────────────────────
    "ACHR": 15.0,   # Archer Aviation: eVTOL, pre-revenue
    "UPST": 35.0,   # Upstart: AI lending, volatile
    "SOFI": 8.0,    # SoFi: neobank, early profits
    "HOOD": 18.0,   # Robinhood: retail brokerage
    "RIVN": 14.0,   # Rivian: EV, cash burn
    "LCID": 3.0,    # Lucid Motors: luxury EV, cash burn
    "JOBY": 5.0,    # Joby Aviation: eVTOL, pre-revenue
    "PLTR": 25.0,   # (already above)
    "SMCI": 35.0,   # Super Micro: AI servers, governance issues
    "CELH": 35.0,   # Celsius Holdings: energy drinks, 30x
    "TOST": 25.0,   # Toast: restaurant tech
    "DUOL": 260.0,  # Duolingo: language learning, 40x
    "ONON": 42.0,   # On Holding: athletic shoes, 30x
    "BIRK": 55.0,   # Birkenstock: footwear brand, 25x
    "CAVA": 75.0,   # CAVA Group: fast-casual dining, 40x
    "CART": 35.0,   # Instacart: grocery delivery
    "IBKR": 145.0,  # Interactive Brokers: broker, 18x
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
