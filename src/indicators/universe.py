"""
Test universe for indicators backtesting.
120 assets: 100+ stocks, all major indices, gold, silver, MicroStrategy, Bitcoin.
"""

# Core universe (from csi_mega_harness.py, 101 assets)
UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "CRM", "ADBE",
    "NFLX", "CRWD",
    # Semiconductors
    "AVGO", "ORCL", "INTC", "CSCO", "IBM", "QCOM", "TXN", "MU", "AMAT", "LRCX",
    "MRVL", "ON",
    # Cloud / SaaS
    "NET", "DDOG", "SNOW", "PLTR", "SHOP",
    # Fintech / Payments
    "PYPL", "V", "MA", "AXP", "SOFI", "HOOD", "AFRM",
    # Banks / Finance
    "JPM", "BAC", "GS", "MS", "SCHW", "BRK-B", "WFC", "C", "BLK",
    # Telecom / Media
    "T", "VZ", "TMUS", "DIS", "CMCSA",
    # Healthcare / Pharma
    "JNJ", "UNH", "PFE", "ABBV", "MRNA", "LLY", "MRK", "BMY", "GILD", "AMGN",
    "TMO", "DHR", "ABT",
    # Defence
    "LMT", "RTX", "NOC", "GD",
    # Industrials
    "CAT", "DE", "BA", "UPS", "GE",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Consumer
    "HD", "NKE", "SBUX", "PG", "KO", "COST", "WMT", "TGT", "LOW", "MCD",
    # High-beta / speculative
    "UPST", "IONQ", "DKNG", "SNAP", "COIN", "ARM", "UBER",
    # Crypto-adjacent
    "MSTR",
    # Indices
    "SPY", "QQQ", "IWM", "DIA",
    # Commodities
    "GLD", "SLV",
    # Crypto
    "BTC-USD",
    # Additional for 120 total
    "OXY", "FCX", "NEM", "NUE", "LIN",
    "PEP", "ABNB", "SQ", "ROKU", "ZM",
    "PANW", "FTNT", "ZS", "ENPH", "FSLR",
    "RIVN", "LCID", "F", "GM",
]

# Sector mapping for analytics
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOGL": "Tech", "AMZN": "Consumer",
    "META": "Tech", "TSLA": "Consumer", "AMD": "Semi", "CRM": "Tech", "ADBE": "Tech",
    "NFLX": "Media", "CRWD": "Cyber", "AVGO": "Semi", "ORCL": "Tech", "INTC": "Semi",
    "CSCO": "Tech", "IBM": "Tech", "QCOM": "Semi", "TXN": "Semi", "MU": "Semi",
    "AMAT": "Semi", "LRCX": "Semi", "MRVL": "Semi", "ON": "Semi",
    "NET": "Cloud", "DDOG": "Cloud", "SNOW": "Cloud", "PLTR": "Cloud", "SHOP": "Cloud",
    "PYPL": "Fintech", "V": "Fintech", "MA": "Fintech", "AXP": "Fintech",
    "SOFI": "Fintech", "HOOD": "Fintech", "AFRM": "Fintech",
    "JPM": "Bank", "BAC": "Bank", "GS": "Bank", "MS": "Bank", "SCHW": "Bank",
    "BRK-B": "Finance", "WFC": "Bank", "C": "Bank", "BLK": "Finance",
    "T": "Telecom", "VZ": "Telecom", "TMUS": "Telecom", "DIS": "Media", "CMCSA": "Media",
    "JNJ": "Pharma", "UNH": "Health", "PFE": "Pharma", "ABBV": "Pharma", "MRNA": "Pharma",
    "LLY": "Pharma", "MRK": "Pharma", "BMY": "Pharma", "GILD": "Pharma", "AMGN": "Pharma",
    "TMO": "Health", "DHR": "Health", "ABT": "Health",
    "LMT": "Defence", "RTX": "Defence", "NOC": "Defence", "GD": "Defence",
    "CAT": "Industrial", "DE": "Industrial", "BA": "Industrial", "UPS": "Industrial", "GE": "Industrial",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "OXY": "Energy",
    "HD": "Consumer", "NKE": "Consumer", "SBUX": "Consumer", "PG": "Consumer", "KO": "Consumer",
    "COST": "Consumer", "WMT": "Consumer", "TGT": "Consumer", "LOW": "Consumer", "MCD": "Consumer",
    "PEP": "Consumer",
    "UPST": "Fintech", "IONQ": "Tech", "DKNG": "Consumer", "SNAP": "Media",
    "COIN": "Crypto", "ARM": "Semi", "UBER": "Consumer", "MSTR": "Crypto",
    "SPY": "Index", "QQQ": "Index", "IWM": "Index", "DIA": "Index",
    "GLD": "Commodity", "SLV": "Commodity", "BTC-USD": "Crypto",
    "FCX": "Materials", "NEM": "Materials", "NUE": "Materials", "LIN": "Materials",
    "ABNB": "Consumer", "SQ": "Fintech", "ROKU": "Media", "ZM": "Tech",
    "PANW": "Cyber", "FTNT": "Cyber", "ZS": "Cyber",
    "ENPH": "Energy", "FSLR": "Energy",
    "RIVN": "Consumer", "LCID": "Consumer", "F": "Consumer", "GM": "Consumer",
}
