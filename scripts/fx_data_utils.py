#!/usr/bin/env python3
"""
fx_data_utils.py

Data fetching and utility functions for FX signals.
Separates data acquisition and currency conversion logic from signal computation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# -------------------------
# Asset Universe Configuration
# -------------------------

# Default asset universe: comprehensive list of FX pairs, commodities, stocks, and ETFs
# This centralized constant is used by fx_pln_jpy_signals.py and tuning/tune_q_mle.py
# Individual scripts can override via command-line arguments
DEFAULT_ASSET_UNIVERSE = [
    # FX pairs
    "PLNJPY=X",
    # Commodities
    "GC=F",  # Gold futures
    "SI=F",  # Silver futures
    # Cryptocurrency
    "BTC-USD",
    "MSTR",  # MicroStrategy

    # -------------------------
    # Major Indices and Broad Market ETFs
    # -------------------------
    "SPY",   # SPDR S&P 500 ETF
    "VOO",   # Vanguard S&P 500 ETF
    "GLD",   # Gold ETF
    "SLV",   # Silver ETF

    # -------------------------
    # S&P 100 Companies by Sector
    # -------------------------

    # Information Technology
    "AAPL",   # Apple Inc.
    "ACN",    # Accenture
    "ADBE",   # Adobe Inc.
    "AMD",    # Advanced Micro Devices
    "AVGO",   # Broadcom
    "CRM",    # Salesforce
    "CSCO",   # Cisco
    "IBM",    # IBM
    "INTC",   # Intel
    "INTU",   # Intuit
    "MSFT",   # Microsoft
    "NOW",    # ServiceNow
    "NVDA",   # Nvidia
    "ORCL",   # Oracle Corporation
    "PLTR",   # Palantir Technologies
    "QCOM",   # Qualcomm
    "SMH",    # VanEck Semiconductor ETF
    "TXN",    # Texas Instruments
    "GOOG",   # Alphabet Inc. (Class C)
    "GOOGL",  # Alphabet Inc. (Class A)
    "META",   # Meta Platforms
    "NFLX",   # Netflix, Inc.

    # Health Care
    "ABBV",   # AbbVie
    "ABT",    # Abbott Laboratories
    "AMGN",   # Amgen
    "BMY",    # Bristol Myers Squibb
    "CVS",    # CVS Health
    "DHR",    # Danaher Corporation
    "GILD",   # Gilead Sciences
    "ISRG",   # Intuitive Surgical
    "JNJ",    # Johnson & Johnson
    "LLY",    # Eli Lilly and Company
    "MDT",    # Medtronic
    "MRK",    # Merck & Co.
    "NVO",    # Novo Nordisk
    "PFE",    # Pfizer
    "TMO",    # Thermo Fisher Scientific
    "UNH",    # UnitedHealth Group

    # Financials
    "AIG",    # American International Group
    "AXP",    # American Express
    "BAC",    # Bank of America
    "BK",     # BNY Mellon
    "BLK",    # BlackRock
    "BRK.B",  # Berkshire Hathaway (Class B)
    "C",      # Citigroup
    "COF",    # Capital One
    "GS",     # Goldman Sachs
    "IBKR",   # Interactive Brokers
    "JPM",    # JPMorgan Chase
    "MA",     # Mastercard
    "MET",    # MetLife
    "MS",     # Morgan Stanley
    "PYPL",   # PayPal
    "SCHW",   # Charles Schwab Corporation
    "USB",    # U.S. Bancorp
    "V",      # Visa Inc.
    "WFC",    # Wells Fargo
    "HOOD",   # Robinhood

    # Consumer Discretionary
    "AMZN",   # Amazon
    "BKNG",   # Booking Holdings
    "GM",     # General Motors
    "HD",     # Home Depot
    "LOW",    # Lowe's
    "MCD",    # McDonald's
    "NKE",    # Nike, Inc.
    "SBUX",   # Starbucks
    "TGT",    # Target Corporation
    "TSLA",   # Tesla, Inc.

    # Industrials
    "CAT",    # Caterpillar Inc.
    "DE",     # Deere & Company
    "EMR",    # Emerson Electric
    "FDX",    # FedEx
    "MMM",    # 3M
    "UBER",   # Uber
    "UNP",    # Union Pacific Corporation
    "UPS",    # United Parcel Service

    # Military
    "ACHR",   # Archer Aviation Inc
    "AIR",    # AAR Corp
    "AIRI",   # Air Industries Group
    "AIRO",   # AIRO Group Holdings Inc
    "AOUT",   # American Outdoor Brands Inc
    "ASTC",   # Astrotech Corp
    "ATI",    # ATI Inc
    "ATRO",   # Astronics Corporation
    "AVAV",   # AeroVironment, Inc.
    "AXON",   # Axon Enterprise, Inc.
    "AZ",     # A2Z Cust2Mate Solutions Corp
    "BA",     # Boeing Co
    "BAH",    # Booz Allen Hamilton Holding Corp
    "BETA",   # Beta Technologies Inc
    "BWXT",   # BWX Technologies, Inc.
    "BYRN",   # Byrna Technologies Inc
    "CACI",   # CACI International Inc
    "CAE",    # CAE Inc
    "CDRE",   # Cadre Holdings Inc
    "CODA",   # Coda Octopus Group Inc
    "CVU",    # CPI Aerostructures Inc
    "CW",     # Curtiss-Wright Corporation
    "DCO",    # Ducommun Incorporated
    "DFSC",   # Defsec Technologies Inc
    "DPRO",   # Draganfly Inc
    "DRS",    # Leonardo DRS Inc
    "EH",     # Ehang Holdings Ltd
    "EMBJ",   # Embraer SA
    "ESLT",   # Elbit Systems Ltd
    "EVEX",   # Eve Holding Inc
    "EVTL",   # Vertical Aerospace Ltd
    "FJET",   # Starfighters Space Inc
    "FLY",    # Firefly Aerospace Inc
    "FTAI",   # FTAI Aviation
    "GD",     # General Dynamics Corp
    "GE",     # GE Aerospace
    "GPUS",   # Hyperscale Data Inc
    "HEI",    # HEICO Corporation
    "HEIA",   # HEICO Corp. (Class A)
    "HII",    # Huntington Ingalls Industries, Inc.
    "HOVR",   # New Horizon Aircraft Ltd
    "HWM",    # Howmet Aerospace Inc.
    "HXL",    # Hexcel Corporation
    "HON",    # Honeywell International Inc
    "ISSC",   # Innovative Solutions & Support Inc
    "JOBY",   # Joby Aviation
    "KITT",   # Nauticus Robotics Inc
    "KRMN",   # Karman Holdings Inc
    "KTOS",   # Kratos Defense & Security Solutions, Inc.
    "LDOS",   # Leidos Holdings, Inc.
    "LHX",    # L3Harris Technologies Inc
    "LMT",    # Lockheed Martin Corp
    "LOAR",   # Loar Holdings Inc
    "LUNR",   # Intuitive Machines Inc
    "MANT",   # ManTech International Corporation
    "MNTS",   # Momentus Inc
    "MOG.A",  # Moog Inc
    "MRCY",   # Mercury Systems, Inc.
    "MSA",    # MSA Safety Incorporated
    "NOC",    # Northrop Grumman Corp
    "NPK",    # National Presto Industries Inc
    "OPXS",   # Optex Systems Holdings Inc
    "OSK",    # Oshkosh Corporation
    "PEW",    # Grabagun Digital Holdings Inc
    "PKE",    # Park Aerospace Corp
    "PL",     # Planet Labs PBC
    "POWW",   # Outdoor Holding Co
    "PRZO",   # Parazero Technologies Ltd
    "RCAT",   # Red Cat Holdings, Inc.
    "RDW",    # Redwire Corp
    "RGR",    # Sturm Ruger & Co Inc
    "RKLB",   # Rocket Lab Corp
    "RTX",    # RTX Corp
    "SAIC",   # Science Applications International Corp
    "SARO",   # StandardAero Inc
    "SATL",   # Satellogic Inc
    "SIDU",   # Sidus Space Inc
    "SIF",    # SIFCO Industries Inc
    "SKYH",   # Sky Harbour Group Corp
    "SPAI",   # Safe Pro Group Inc
    "SPCE",   # Virgin Galactic Holdings Inc
    "SPR",    # Spirit AeroSystems Holdings, Inc.
    "SWBI",   # Smith & Wesson Brands Inc
    "TATT",   # TAT Technologies Ltd
    "TDG",    # TransDigm Group Inc
    "TDY",    # Teledyne Technologies Incorporated
    "TGI",    # Triumph Group, Inc.
    "TXT",    # Textron Inc.
    "VSAT",   # ViaSat Inc
    "VSEC",   # VSE Corporation
    "VTSI",   # Virtra Inc
    "VVX",    # V2X, Inc.
    "VWAV",   # Visionwave Holdings Inc
    "VOYG",   # Voyager Technologies Inc
    "WWD",    # Woodward, Inc.
    "RHM.DE", # Rheinmetall (German defense)
    "AIR.PA", # Airbus (European aerospace)
    "HO.PA",  # Thales (French defense electronics)
    "HAG.DE", # Hensoldt (German defense electronics)
    "BA.L",   # BAE Systems (British defense)
    "FACC.VI",# FACC AG (Austrian aerospace components)
    "MTX.DE", # MTU Aero Engines (German aerospace)

    # Communication Services
    "CMCSA",  # Comcast
    "DIS",    # Walt Disney Company (The)
    "T",      # AT&T
    "TMUS",   # T-Mobile US
    "VZ",     # Verizon

    # Consumer Staples
    "CL",     # Colgate-Palmolive
    "COST",   # Costco
    "KO",     # Coca-Cola Company (The)
    "MDLZ",   # Mondelēz International
    "MO",     # Altria
    "PEP",    # PepsiCo
    "PG",     # Procter & Gamble
    "PM",     # Philip Morris International
    "WMT",    # Walmart

    # Energy
    "COP",    # ConocoPhillips
    "CVX",    # Chevron Corporation
    "XOM",    # ExxonMobil

    # Utilities
#     "DUK",    # Duke Energy
    "NEE",    # NextEra Energy
    "SO",     # Southern Company

    # Real Estate
#     "AMT",    # American Tower
    "SPG",    # Simon Property Group

    # Materials
    "LIN",    # Linde plc
    "NEM",    # Newmont Mining

    # Asian Tech & Manufacturing
    "005930.KS", # Samsung Electronics (Korean)

    # -------------------------
    # VanEck ETFs
    # -------------------------
    "AFK",    # VanEck Africa Index ETF
    "ANGL",   # VanEck Fallen Angel High Yield Bond ETF
#     "BRF",    # VanEck Brazil Small-Cap ETF
    "CNXT",   # VanEck ChiNext ETF
#     "DURA",   # VanEck Morningstar Durable Dividend ETF
    "EGPT",   # VanEck Egypt Index ETF
#     "EMLC",   # VanEck J.P. Morgan EM Local Currency Bond ETF
    "FLTR",   # VanEck Investment Grade Floating Rate ETF
#     "GDX",    # VanEck Gold Miners ETF
#     "GDXJ",   # VanEck Junior Gold Miners ETF
    "GLIN",   # VanEck India Growth Leaders ETF
    "MOTG",   # VanEck Morningstar Global Wide Moat ETF
#     "GRNB",   # VanEck Green Bond ETF
#     "HYEM",   # VanEck Emerging Markets High Yield Bond ETF
    "IDX",    # VanEck Indonesia Index ETF
#     "ITM",    # VanEck Intermediate Muni ETF
    "MLN",    # VanEck Long Muni ETF
#     "MOAT",   # VanEck Morningstar Wide Moat ETF
#     "MOO",    # VanEck Agribusiness ETF
#     "MOTI",   # VanEck Morningstar International Moat ETF
    "NLR",    # VanEck Uranium+Nuclear Energy ETF
#     "OIH",    # VanEck Oil Services ETF
#     "PPH",    # VanEck Pharmaceutical ETF
#     "REMX",   # VanEck Rare Earth/Strategic Metals ETF
#     "RSX",    # VanEck Russia ETF
#     "RSXJ",   # VanEck Russia Small-Cap ETF
#     "RTH",    # VanEck Retail ETF
#     "SLX",    # VanEck Steel ETF
#     "SMOG",   # VanEck Low Carbon Energy ETF
#     "VNM",    # VanEck Vietnam ETF
#     "ESPO",   # VanEck Video Gaming and eSports UCITS ETF
#     "GFA",    # VanEck Global Fallen Angel High Yield Bond UCITS ETF
#     "HDRO",   # VanEck Hydrogen Economy UCITS ETF
#     "TCBT",   # VanEck iBoxx EUR Corporates UCITS ETF
#     "TDIV",   # VanEck Morningstar Developed Markets Dividend Leaders UCITS ETF
#     "TEET",   # VanEck Sustainable European Equal Weight UCITS ETF
#     "TGBT",   # VanEck iBoxx EUR Sovereign Diversified 1-10 UCITS ETF
#     "TRET",   # VanEck Global Real Estate UCITS ETF
#     "TSWE",   # VanEck Sustainable World Equal Weight UCITS ETF
#     "TAT",    # VanEck iBoxx EUR Sovereign Capped AAA-AA 1-5 UCITS ETF
]

MAPPING = {
    # Prefer active, liquid proxies first to avoid Yahoo "possibly delisted" noise
    "GOOO": ["GOOG", "GOOGL", "GOOO"],
    "GLDW": ["GLDM", "GLD", "GLDW"],
    "SGLP": ["SGLP.L", "SGLP", "SGLP.LON"],
    "GLDE": ["GLD", "IAU", "GLDE"],
    "FACC": ["FACC.VI", "FACC"],
    "SLVI": ["SLV", "SLVP", "SLVI"],
    "TKA": ["TKA.DE", "TKA"],
    # Netflix and Novo Nordisk
    "NFLX": ["NFLX"],
    "NOVO": ["NVO", "NOVO-B.CO", "NOVOB.CO", "NOVO-B.CO"],
    # Kratos (alias to KTOS)
    "KRATOS": ["KTOS"],
    # Requested blue chips and defense/aero additions
    "RHEINMETALL": ["RHM.DE", "RHM.F"],
    "AIRBUS": ["AIR.PA", "AIR.DE"],
    "RENK": ["R3NK.DE", "RNK.DE"],
    "NORTHROP": ["NOC"],
    "NORTHROP GRUMMAN": ["NOC"],
    "NORTHRUP": ["NOC"],
    "NORTHRUP GRUNMAN": ["NOC"],
    "NVIDIA": ["NVDA"],
    "MICROSOFT": ["MSFT"],
    "APPLE": ["AAPL"],
    "AMD": ["AMD"],
    "UBER": ["UBER"],
    "TESLA": ["TSLA"],
    "VANGUARD SP 500": ["VOO", "VUSA.L"],
    "VANGARD SP 500": ["VOO", "VUSA.L"],
    "VANGUARD S&P 500": ["VOO", "VUSA.L"],
    "THALES": ["HO.PA"],
    "HENSOLDT": ["HAG.DE"],
    "SAMSUNG": ["005930.KS", "005935.KS"],
    "TKMS AG & CO": ["TKA.DE", "TKAMY"],
    # Additional defense, aerospace, and mining companies
    "BAE SYSTEMS": ["BA.L"],
    "BAE": ["BA.L"],
    "NEWMONT": ["NEM"],
    "NEWMONT CORP": ["NEM"],
    "HOWMET": ["HWM"],
    "HOWMET AEROSPACE": ["HWM"],
    "BROADCOM": ["AVGO"],
    # SPY, S&P 500, Magnificent 7, and Semiconductor ETFs
    "SPY": ["SPY"],
    "SP500": ["^GSPC", "SPY"],
    "S&P500": ["^GSPC", "SPY"],
    "S&P 500": ["^GSPC", "SPY"],
    "MAGNIFICENT 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "MAGNIFICENT SEVEN": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "MAG7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "GOOGLE": ["GOOGL", "GOOG"],
    "ALPHABET": ["GOOGL", "GOOG"],
    "AMAZON": ["AMZN"],
    "META": ["META"],
    "FACEBOOK": ["META"],
    "SEMICONDUCTOR ETF": ["SMH", "SOXX"],
    "SMH": ["SMH"],
    "SOXX": ["SOXX"],
    # keep identity candidates (with improved MTX mapping to MTU Aero Engines on XETRA)
    "RKLB": ["RKLB"],
    "MTX": ["MTX.DE", "MTX"],
    "IBKR": ["IBKR"],
    "HOOD": ["HOOD"],

    # S&P 100 additions by sector

    # Information Technology
    "ACCENTURE": ["ACN"],
    "ADOBE": ["ADBE"],
    "SALESFORCE": ["CRM"],
    "CISCO": ["CSCO"],
    "INTEL": ["INTC"],
    "ORACLE": ["ORCL"],
    "PALANTIR": ["PLTR"],
    "QUALCOMM": ["QCOM"],
    "TEXAS INSTRUMENTS": ["TXN"],

    # Health Care
    "ABBVIE": ["ABBV"],
    "ABBOTT LABS": ["ABT"],
    "AMGEN": ["AMGN"],
    "BRISTOL MYERS SQUIBB": ["BMY"],
    "CVS HEALTH": ["CVS"],
    "DANAHER": ["DHR"],
    "GILEAD SCIENCES": ["GILD"],
    "INTUITIVE SURGICAL": ["ISRG"],
    "JOHNSON & JOHNSON": ["JNJ"],
    "ELI LILLY": ["LLY"],
    "MEDTRONIC": ["MDT"],
    "MERCK": ["MRK"],
    "THERMO FISHER SCIENTIFIC": ["TMO"],
    "UNITEDHEALTH": ["UNH"],

    # Financials
    "AMERICAN INTERNATIONAL GROUP": ["AIG"],
    "AMERICAN EXPRESS": ["AXP"],
    "BANK OF AMERICA": ["BAC"],
    "BNY MELLON": ["BK"],
    "BLACKROCK": ["BLK"],
    "BERKSHIRE HATHAWAY": ["BRK.B"],
    "CITIGROUP": ["C"],
    "CAPITAL ONE": ["COF"],
    "GOLDMAN SACHS": ["GS"],
    "JPMORGAN CHASE": ["JPM"],
    "MASTERCARD": ["MA"],
    "METLIFE": ["MET"],
    "MORGAN STANLEY": ["MS"],
    "PAYPAL": ["PYPL"],
    "CHARLES SCHWAB": ["SCHW"],
    "US BANCORP": ["USB"],
    "VISA": ["V"],
    "WELLS FARGO": ["WFC"],

    # Consumer Discretionary
    "BOOKING HOLDINGS": ["BKNG"],
    "GENERAL MOTORS": ["GM"],
    "HOME DEPOT": ["HD"],
    "LOWES": ["LOW"],
    "MCDONALDS": ["MCD"],
    "NIKE": ["NKE"],
    "STARBUCKS": ["SBUX"],
    "TARGET": ["TGT"],

    # Industrials
    "CATERPILLAR": ["CAT"],
    "DEERE & COMPANY": ["DE"],
    "EMERSON ELECTRIC": ["EMR"],
    "FEDEX": ["FDX"],
    "3M": ["MMM"],
    "UNION PACIFIC": ["UNP"],
    "UPS": ["UPS"],

    # Military
    "BOEING": ["BA"],
    "ARCHER AVIATION": ["ACHR"],
    "AAR CORP": ["AIR"],
    "AIR INDUSTRIES GROUP": ["AIRI"],
    "AIRO GROUP HOLDINGS": ["AIRO"],
    "AMERICAN OUTDOOR BRANDS": ["AOUT"],
    "ASTROTECH": ["ASTC"],
    "ATI": ["ATI"],
    "ASTRONICS": ["ATRO"],
    "AEROVIRONMENT": ["AVAV"],
    "AXON ENTERPRISE": ["AXON"],
    "A2Z CUST2MATE SOLUTIONS": ["AZ"],
    "BOOZ ALLEN HAMILTON": ["BAH"],
    "BETA TECHNOLOGIES": ["BETA"],
    "BWX TECHNOLOGIES": ["BWXT"],
    "BYRNA TECHNOLOGIES": ["BYRN"],
    "CACI INTERNATIONAL": ["CACI"],
    "CAE": ["CAE"],
    "CADRE HOLDINGS": ["CDRE"],
    "CODA OCTOPUS GROUP": ["CODA"],
    "CPI AEROSTRUCTURES": ["CVU"],
    "CURTISS-WRIGHT": ["CW"],
    "DUCOMMUN": ["DCO"],
    "DEFSEC TECHNOLOGIES": ["DFSC"],
    "DRAGANFLY": ["DPRO"],
    "LEONARDO DRS": ["DRS"],
    "EHANG HOLDINGS": ["EH"],
    "EMBRAER": ["EMBJ"],
    "ELBIT SYSTEMS": ["ESLT"],
    "EVE HOLDING": ["EVEX"],
    "VERTICAL AEROSPACE": ["EVTL"],
    "STARFIGHTERS SPACE": ["FJET"],
    "FIREFLY AEROSPACE": ["FLY"],
    "FTAI AVIATION": ["FTAI"],
    "GENERAL DYNAMICS": ["GD"],
    "GE AEROSPACE": ["GE"],
    "HYPERSCALE DATA": ["GPUS"],
    "HEICO": ["HEI", "HEIA"],
    "HUNTINGTON INGALLS INDUSTRIES": ["HII"],
    "NEW HORIZON AIRCRAFT": ["HOVR"],
    "HOWMET AEROSPACE": ["HWM"],
    "HEXCEL": ["HXL"],
    "HONEYWELL INTERNATIONAL": ["HON"],
    "INNOVATIVE SOLUTIONS & SUPPORT": ["ISSC"],
    "JOBY AVIATION": ["JOBY"],
    "NAUTICUS ROBOTICS": ["KITT"],
    "KARMAN HOLDINGS": ["KRMN"],
    "KRATOS DEFENSE & SECURITY SOLUTIONS": ["KTOS"],
    "LEIDOS HOLDINGS": ["LDOS"],
    "L3HARRIS TECHNOLOGIES": ["LHX"],
    "LOCKHEED MARTIN": ["LMT"],
    "LOAR HOLDINGS": ["LOAR"],
    "INTUITIVE MACHINES": ["LUNR"],
    "MANTECH INTERNATIONAL": ["MANT"],
    "MOMENTUS": ["MNTS"],
    "MOOG": ["MOG.A"],
    "MERCURY SYSTEMS": ["MRCY"],
    "MSA SAFETY": ["MSA"],
    "NORTHROP GRUMMAN": ["NOC"],
    "NATIONAL PRESTO INDUSTRIES": ["NPK"],
    "OPTEX SYSTEMS HOLDINGS": ["OPXS"],
    "OSHKOSH": ["OSK"],
    "GRABAGUN DIGITAL HOLDINGS": ["PEW"],
    "PARK AEROSPACE": ["PKE"],
    "PLANET LABS": ["PL"],
    "OUTDOOR HOLDING": ["POWW"],
    "PARAZERO TECHNOLOGIES": ["PRZO"],
    "RED CAT HOLDINGS": ["RCAT"],
    "REDWIRE": ["RDW"],
    "STURM RUGER & CO": ["RGR"],
    "ROCKET LAB": ["RKLB"],
    "RTX": ["RTX"],
    "SCIENCE APPLICATIONS INTERNATIONAL": ["SAIC"],
    "STANDARDAERO": ["SARO"],
    "SATELLOGIC": ["SATL"],
    "SIDUS SPACE": ["SIDU"],
    "SIFCO INDUSTRIES": ["SIF"],
    "SKY HARBOUR GROUP": ["SKYH"],
    "SAFE PRO GROUP": ["SPAI"],
    "VIRGIN GALACTIC HOLDINGS": ["SPCE"],
    "SPIRIT AEROSYSTEMS HOLDINGS": ["SPR"],
    "SMITH & WESSON BRANDS": ["SWBI"],
    "TAT TECHNOLOGIES": ["TATT"],
    "TRANSDIGM GROUP": ["TDG"],
    "TELEDYNE TECHNOLOGIES": ["TDY"],
    "TRIUMPH GROUP": ["TGI"],
    "TEXTRON": ["TXT"],
    "VIASAT": ["VSAT"],
    "VSE": ["VSEC"],
    "VIRTRA": ["VTSI"],
    "V2X": ["VVX"],
    "VISIONWAVE HOLDINGS": ["VWAV"],
    "VOYAGER TECHNOLOGIES": ["VOYG"],
    "WOODWARD": ["WWD"],

    # Communication Services
    "COMCAST": ["CMCSA"],
    "DISNEY": ["DIS"],
    "AT&T": ["T"],
    "T-MOBILE": ["TMUS"],
    "VERIZON": ["VZ"],

    # Consumer Staples
    "COLGATE-PALMOLIVE": ["CL"],
    "COSTCO": ["COST"],
    "COCA-COLA": ["KO"],
    "MONDELEZ": ["MDLZ"],
    "ALTRIA": ["MO"],
    "PEPSICO": ["PEP"],
    "PROCTER & GAMBLE": ["PG"],
    "PHILIP MORRIS": ["PM"],
    "WALMART": ["WMT"],

    # Energy
    "CONOCOPHILLIPS": ["COP"],
    "CHEVRON": ["CVX"],
    "EXXONMOBIL": ["XOM"],

    # Utilities
#         "DUKE ENERGY": ["DUK"],
    "NEXTERA ENERGY": ["NEE"],
    "SOUTHERN COMPANY": ["SO"],

    # Real Estate
#         "AMERICAN TOWER": ["AMT"],
#         "SIMON PROPERTY GROUP": ["SPG"],

    # Materials
    "LINDE": ["LIN"],
    # Vaneck ETFs
    "VANECK SEMICONDUCTOR": ["SMH"],
#         "VANECK GOLD MINERS": ["GDX"],
#         "VANECK JUNIOR GOLD MINERS": ["GDXJ"],
    "VANECK OIL SERVICES": ["OIH"],
#         "VANECK RETAIL": ["RTH"],
    "VANECK AGRIBUSINESS": ["MOO"],
    "VANECK GAMING ETF": ["ESPO"],
#         "VANECK AFRICA INDEX": ["AFK"],
    "VANECK FALLEN ANGEL HIGH YIELD BOND": ["ANGL"],
#         "VANECK BRAZIL SMALL-CAP": ["BRF"],
    "VANECK CHINEXT": ["CNXT"],
    "VANECK MORNINGSTAR DURABLE DIVIDEND": ["DURA"],
    "VANECK EGYPT INDEX": ["EGPT"],
#         "VANECK JP MORGAN EM LOCAL CURRENCY BOND": ["EMLC"],
    "VANECK INVESTMENT GRADE FLOATING RATE": ["FLTR"],
    "VANECK INDIA GROWTH LEADERS": ["GLIN"],
    "VANECK MORNINGSTAR GLOBAL WIDE MOAT": ["MOTG"],
#         "VANECK GREEN BOND": ["GRNB"],
#         "VANECK EMERGING MARKETS HIGH YIELD BOND": ["HYEM"],
#         "VANECK INDONESIA INDEX": ["IDX"],
    "VANECK INTERMEDIATE MUNI": ["ITM"],
#         "VANECK LONG MUNI": ["MLN"],
#         "VANECK MORNINGSTAR WIDE MOAT": ["MOAT"],
    "VANECK MORNINGSTAR INTERNATIONAL MOAT": ["MOTI"],
#         "VANECK URANIUM+NUCLEAR ENERGY": ["NLR"],
#         "VANECK PHARMACEUTICAL": ["PPH"],
#         "VANECK RARE EARTH/STRATEGIC METALS": ["REMX"],
#         "VANECK RUSSIA": ["RSX"],
#         "VANECK RUSSIA SMALL-CAP": ["RSXJ"],
#         "VANECK STEEL": ["SLX"],
#         "VANECK LOW CARBON ENERGY": ["SMOG"],
#         "VANECK VIETNAM": ["VNM"],
#         "VANECK GLOBAL FALLEN ANGEL HIGH YIELD BOND UCITS": ["GFA"],
#         "VANECK HYDROGEN ECONOMY UCITS": ["HDRO"],
#         "VANECK IBOXX EUR CORPORATES UCITS": ["TCBT"],
#         "VANECK MORNINGSTAR DEVELOPED MARKETS DIVIDEND LEADERS UCITS": ["TDIV"],
#         "VANECK SUSTAINABLE EUROPEAN EQUAL WEIGHT UCITS": ["TEET"],
#         "VANECK IBOXX EUR SOVEREIGN DIVERSIFIED 1-10 UCITS": ["TGBT"],
#         "VANECK GLOBAL REAL ESTATE UCITS": ["TRET"],
#         "VANECK SUSTAINABLE WORLD EQUAL WEIGHT UCITS": ["TSWE"],
#         "VANECK IBOXX EUR SOVEREIGN CAPPED AAA-AA 1-5 UCITS": ["TAT"],
}


def get_default_asset_universe() -> List[str]:
    """
    Get the default asset universe.

    Returns a copy of the centralized asset list to prevent external modification.
    This function provides a clean API for retrieving the asset universe.

    Returns:
        List of asset symbols/names
    """
    return DEFAULT_ASSET_UNIVERSE.copy()


# -------------------------
# Utils
# -------------------------

def norm_cdf(x: float) -> float:
    # Numerically stable normal CDF via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _to_float(x) -> float:
    try:
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        if hasattr(x, "item"):
            return float(x.item())
        arr = np.asarray(x)
        if arr.size == 1:
            return float(arr.reshape(()).item())
        return float("nan")
    except Exception:
        return float("nan")


def safe_last(s: pd.Series) -> float:
    try:
        return _to_float(s.iloc[-1])
    except Exception:
        return float("nan")


def winsorize(x, p: float = 0.01):
    """Winsorize a Series or DataFrame column-wise using scalar thresholds.
    - Robust to pandas alignment quirks
    - Avoids deprecated float(Series) paths by using numpy percentiles
    - Gracefully handles empty/singleton inputs by returning them unchanged
    """
    if isinstance(x, pd.DataFrame):
        return x.apply(lambda s: winsorize(s, p))
    if isinstance(x, pd.Series):
        vals = x.to_numpy(dtype=float)
        if vals.size < 3:
            return x  # not enough data to estimate tails
        lo_hi = np.nanpercentile(vals, [100 * p, 100 * (1 - p)])
        lo = float(lo_hi[0])
        hi = float(lo_hi[1])
        clipped = np.clip(vals, lo, hi)
        return pd.Series(clipped, index=x.index, name=getattr(x, "name", None))
    # Fallback: treat as array-like
    arr = np.asarray(x, dtype=float)
    if arr.size < 3:
        return arr
    lo_hi = np.nanpercentile(arr, [100 * p, 100 * (1 - p)])
    lo = float(lo_hi[0])
    hi = float(lo_hi[1])
    return np.clip(arr, lo, hi)


# -------------------------
# Data
# -------------------------

def _download_prices(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Robust Yahoo fetch with multiple strategies.
    Returns a DataFrame with OHLC columns (if available).
    - Tries yf.download first
    - Falls back to Ticker.history
    - Tries again without auto_adjust
    Normalizes DatetimeIndex to tz-naive for stability.
    """
    # Try standard download
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if df is not None and not df.empty:
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
    except Exception:
        pass
    # Try Ticker.history
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(start=start, end=end, auto_adjust=True)
        if df is not None and not df.empty:
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
    except Exception:
        pass
    # Try without auto_adjust
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, threads=False)
        if df is not None and not df.empty:
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
    except Exception:
        pass
    return pd.DataFrame()


# Display-name cache for full asset names (e.g., company longName)
_DISPLAY_NAME_CACHE: Dict[str, str] = {}

def _resolve_display_name(symbol: str) -> str:
    """Return a human-friendly display name for a Yahoo symbol.
    Tries yfinance fast_info/info longName/shortName; falls back to the symbol itself.
    Caches results for repeated use.
    """
    sym = (symbol or "").strip()
    if not sym:
        return symbol
    if sym in _DISPLAY_NAME_CACHE:
        return _DISPLAY_NAME_CACHE[sym]
    name: Optional[str] = None
    try:
        tk = yf.Ticker(sym)
        # Try info first (often has longName)
        try:
            info = tk.info or {}
            name = info.get("longName") or info.get("shortName") or info.get("name")
        except Exception:
            name = None
        if not name:
            # Try fast_info (may have shortName on some tickers)
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                if isinstance(fi, dict):
                    name = fi.get("shortName") or fi.get("longName")
                else:
                    name = getattr(fi, "shortName", None) or getattr(fi, "longName", None)
    except Exception:
        name = None
    disp = str(name).strip() if name else sym
    _DISPLAY_NAME_CACHE[sym] = disp
    return disp


def _fetch_px_symbol(symbol: str, start: Optional[str], end: Optional[str]) -> pd.Series:
    data = _download_prices(symbol, start, end)
    if data is None or data.empty:
        raise RuntimeError(f"No data for {symbol}")
    for col in ("Close", "Adj Close"):
        if isinstance(data, pd.DataFrame) and col in data.columns:
            px = data[col].dropna()
            px.name = "px"
            return px
    if isinstance(data, pd.Series):
        px = data.dropna()
        px.name = "px"
        return px
    raise RuntimeError(f"No price column found for {symbol}")


def _fetch_with_fallback(symbols: List[str], start: Optional[str], end: Optional[str]) -> Tuple[pd.Series, str]:
    last_err: Optional[Exception] = None
    for sym in symbols:
        try:
            px = _fetch_px_symbol(sym, start, end)
            return px, sym
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError(f"No data for symbols: {symbols}")


def fetch_px(pair: str, start: Optional[str], end: Optional[str]) -> pd.Series:
    # Backward-compatible: fetch for specified pair
    return _fetch_px_symbol(pair, start, end)


def fetch_usd_to_pln_exchange_rate(start: Optional[str], end: Optional[str]) -> pd.Series:
    """Fetch USD/PLN exchange rate as a Series. Tries multiple routes:
    1) USDPLN=X directly
    2) Invert PLNUSD=X
    3) Cross via EUR: USDPLN = EURPLN / EURUSD
    """
    # 1) Direct USDPLN
    try:
        s, used = _fetch_with_fallback(["USDPLN=X"], start, end)
        return s
    except Exception:
        pass
    # 2) Invert PLNUSD
    try:
        s, used = _fetch_with_fallback(["PLNUSD=X"], start, end)
        inv = (1.0 / s)
        inv.name = "px"
        return inv
    except Exception:
        pass
    # 3) Cross via EUR
    try:
        eurpln, _ = _fetch_with_fallback(["EURPLN=X"], start, end)
        eurusd, _ = _fetch_with_fallback(["EURUSD=X"], start, end)
        df = pd.concat([eurpln, eurusd], axis=1, join="inner").dropna()
        df.columns = ["eurpln", "eurusd"]
        cross = (df["eurpln"] / df["eurusd"]).rename("px")
        return cross
    except Exception as e:
        raise RuntimeError(f"Unable to get USDPLN via direct, inverse, or EUR cross: {e}")


def detect_quote_currency(symbol: str) -> str:
    """Try to detect the quote currency for a Yahoo symbol.
    Returns uppercase ISO code like 'USD','EUR','GBP','GBp','JPY', or '' if unknown.
    """
    try:
        tk = yf.Ticker(symbol)
        # fast_info first
        cur = None
        try:
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                cur = fi.get("currency") if isinstance(fi, dict) else getattr(fi, "currency", None)
        except Exception:
            cur = None
        if not cur:
            info = tk.info or {}
            cur = info.get("currency")
        if cur:
            return str(cur).upper()
    except Exception:
        pass
    # Heuristics from suffix
    s = symbol.upper()
    if s.endswith(".DE") or s.endswith(".F") or s.endswith(".BE") or s.endswith(".XETRA"):
        return "EUR"
    if s.endswith(".PA"):
        return "EUR"
    if s.endswith(".L") or s.endswith(".LON"):
        # London often in GBX (pence)
        return "GBX"
    if s.endswith(".VI"):
        return "EUR"
    if s.endswith(".CO"):
        return "DKK"
    if s.endswith(".TO") or s.endswith(".TSX"):
        return "CAD"
    if s.endswith(".SZ") or s.endswith(".SS"):
        return "CNY"
    if s.endswith(".KS") or s.endswith(".KQ"):
        return "KRW"
    # Default to USD
    return "USD"


def _as_series(x) -> pd.Series:
    """Coerce input to a 1-D pandas Series if possible; otherwise return empty Series."""
    if isinstance(x, pd.Series):
        # Squeeze potential 2D values inside the Series
        vals = np.asarray(x.values)
        if vals.ndim == 2 and vals.shape[1] == 1:
            return pd.Series(vals.ravel(), index=x.index, name=getattr(x, "name", None))
        return x
    if isinstance(x, pd.DataFrame):
        # If single column, squeeze; else take the first column
        if x.shape[1] >= 1:
            s = x.iloc[:, 0]
            s.name = getattr(s, "name", x.columns[0])
            return _as_series(s)
        return pd.Series(dtype=float)
    try:
        arr = np.asarray(x)
        if arr.ndim == 0:
            return pd.Series([_to_float(arr)])
        if arr.ndim == 1:
            return pd.Series(arr)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return pd.Series(arr.ravel())
    except Exception:
        pass
    return pd.Series(dtype=float)


def _ensure_float_series(s: pd.Series) -> pd.Series:
    """Ensure a 1-D float Series free of nested arrays/objects.
    - Coerces to Series via _as_series
    - Converts to numeric dtype; non-convertible entries become NaN
    """
    s = _as_series(s)
    if s.empty:
        return s
    # Try fast astype to float
    try:
        s = s.astype(float)
        return s
    except Exception:
        pass
    # Fallback: to_numeric coercion
    try:
        s = pd.to_numeric(s, errors="coerce")
    except Exception:
        # Last resort: build from numpy values squeezed to 1-D
        vals = np.asarray(s.values)
        if vals.ndim > 1:
            vals = vals.ravel()
        s = pd.Series(vals, index=s.index)
        s = pd.to_numeric(s, errors="coerce")
    return s


def _align_fx_asof(native_px: pd.Series, fx_px: pd.Series, max_gap_days: int = 7) -> pd.Series:
    """Align FX series to native dates using asof within a tolerance window.
    Falls back to forward direction if backward match is missing.
    Returns aligned FX indexed by native dates (NaN where no match within tolerance)."""
    # Ensure inputs are Series
    native_px = _as_series(native_px)
    fx_px = _as_series(fx_px)
    if native_px.empty:
        return pd.Series(index=native_px.index, dtype=float)
    if fx_px.empty:
        return pd.Series(index=native_px.index, dtype=float)
    # Build merge frames
    left = native_px.rename("native").to_frame().reset_index()
    left = left.rename(columns={left.columns[0]: "date"})
    right = fx_px.rename("fx").to_frame().reset_index()
    right = right.rename(columns={right.columns[0]: "date"})
    # Sort and asof merge
    left = left.sort_values("date")
    right = right.sort_values("date")
    tol = pd.Timedelta(days=max_gap_days)
    back = pd.merge_asof(left, right, on="date", direction="backward", tolerance=tol)
    fwd = pd.merge_asof(left, right, on="date", direction="forward", tolerance=tol)
    fx_aligned = back["fx"].fillna(fwd["fx"])  # prefer backward, then forward
    fx_aligned.index = pd.to_datetime(left["date"])  # align index to native dates
    return fx_aligned


def convert_currency_to_pln(quote_ccy: str, start: Optional[str], end: Optional[str], native_index: Optional[pd.DatetimeIndex] = None) -> pd.Series:
    """Return a Series of FX rate in PLN per 1 unit of quote_ccy.
    Expands the fetch window to cover the native price index +/- 30 days for overlap robustness."""
    q = (quote_ccy or "").upper().strip()
    # Expand window around native index
    s_ext, e_ext = start, end
    if native_index is not None and len(native_index) > 0:
        try:
            s_ext = (pd.to_datetime(native_index.min()) - pd.Timedelta(days=30)).date().isoformat()
            e_ext = (pd.to_datetime(native_index.max()) + pd.Timedelta(days=5)).date().isoformat()
        except Exception:
            pass

    def fetch(sym_list: List[str]) -> pd.Series:
        s, _ = _fetch_with_fallback(sym_list, s_ext, e_ext)
        return s

    if q in ("PLN", "PLN "):
        # Return a flat-1 series over the native index for easy alignment
        return pd.Series(1.0, index=pd.DatetimeIndex(native_index) if native_index is not None else [pd.Timestamp("1970-01-01")])
    if q == "USD":
        return fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
    if q == "EUR":
        try:
            eurpln, _ = _fetch_with_fallback(["EURPLN=X"], s_ext, e_ext)
            return eurpln
        except Exception:
            # EURPLN via USD: EURPLN = EURUSD * USDPLN
            eurusd, _ = _fetch_with_fallback(["EURUSD=X"], s_ext, e_ext)
            return eurusd * fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
    if q in ("GBP", "GBX", "GBPp", "GBP P", "GBp"):
        gbppln, _ = _fetch_with_fallback(["GBPPLN=X"], s_ext, e_ext)
        return gbppln * (0.01 if q in ("GBX", "GBPp", "GBP P", "GBp") else 1.0)
    if q == "JPY":
        try:
            plnjpy, _ = _fetch_with_fallback(["PLNJPY=X"], s_ext, e_ext)
            return 1.0 / plnjpy
        except Exception:
            jpypln, _ = _fetch_with_fallback(["JPYPLN=X"], s_ext, e_ext)
            return jpypln
    if q == "CAD":
        try:
            cadpln, _ = _fetch_with_fallback(["CADPLN=X"], s_ext, e_ext)
            return cadpln
        except Exception:
            # Try CADUSD cross
            try:
                usdcad, _ = _fetch_with_fallback(["USDCAD=X"], s_ext, e_ext)
                cadusd = 1.0 / usdcad
            except Exception:
                cadusd, _ = _fetch_with_fallback(["CADUSD=X"], s_ext, e_ext)
            return cadusd * fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
    if q == "CHF":
        try:
            chfpln, _ = _fetch_with_fallback(["CHFPLN=X"], s_ext, e_ext)
            return chfpln
        except Exception:
            try:
                usdchf, _ = _fetch_with_fallback(["USDCHF=X"], s_ext, e_ext)
                chfusd = 1.0 / usdchf
            except Exception:
                chfusd, _ = _fetch_with_fallback(["CHFUSD=X"], s_ext, e_ext)
            return chfusd * fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
    if q == "AUD":
        try:
            audpln, _ = _fetch_with_fallback(["AUDPLN=X"], s_ext, e_ext)
            return audpln
        except Exception:
            audusd, _ = _fetch_with_fallback(["AUDUSD=X"], s_ext, e_ext)
            return audusd * fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
    if q == "SEK":
        try:
            sekpln, _ = _fetch_with_fallback(["SEKPLN=X"], s_ext, e_ext)
            return sekpln
        except Exception:
            eurpln, _ = _fetch_with_fallback(["EURPLN=X"], s_ext, e_ext)
            eurseK, _ = _fetch_with_fallback(["EURSEK=X"], s_ext, e_ext)
            return eurpln / eurseK
    if q == "NOK":
        try:
            nokpln, _ = _fetch_with_fallback(["NOKPLN=X"], s_ext, e_ext)
            return nokpln
        except Exception:
            eurpln, _ = _fetch_with_fallback(["EURPLN=X"], s_ext, e_ext)
            eurnok, _ = _fetch_with_fallback(["EURNOK=X"], s_ext, e_ext)
            return eurpln / eurnok
    if q == "DKK":
        try:
            dkkpln, _ = _fetch_with_fallback(["DKKPLN=X"], s_ext, e_ext)
            return dkkpln
        except Exception:
            eurpln, _ = _fetch_with_fallback(["EURPLN=X"], s_ext, e_ext)
            eurdkk, _ = _fetch_with_fallback(["EURDKK=X"], s_ext, e_ext)
            return eurpln / eurdkk
    if q == "HKD":
        try:
            hkdpln, _ = _fetch_with_fallback(["HKDPLN=X"], s_ext, e_ext)
            return hkdpln
        except Exception:
            usdpln = fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
            usdhkd, _ = _fetch_with_fallback(["USDHKD=X"], s_ext, e_ext)
            return usdpln / usdhkd
    if q == "KRW":
        # PLN per KRW = (PLN per USD) / (KRW per USD)
        try:
            usdpln = fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
            usdkrw, _ = _fetch_with_fallback(["USDKRW=X"], s_ext, e_ext)
            # Align by native index if available
            if native_index is not None and len(native_index) > 0:
                usdpln = usdpln.reindex(pd.DatetimeIndex(native_index)).ffill().bfill()
                usdkrw = usdkrw.reindex(pd.DatetimeIndex(native_index)).ffill().bfill()
            return (usdpln / usdkrw).rename("px")
        except Exception:
            # Fallback: try KRWUSD and invert
            try:
                krwusd, _ = _fetch_with_fallback(["KRWUSD=X"], s_ext, e_ext)
                usdpln = fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
                if native_index is not None and len(native_index) > 0:
                    usdpln = usdpln.reindex(pd.DatetimeIndex(native_index)).ffill().bfill()
                    krwusd = krwusd.reindex(pd.DatetimeIndex(native_index)).ffill().bfill()
                return (usdpln * krwusd).rename("px")
            except Exception:
                pass
        # As last resort, assume USD (may be wrong for KRW assets but avoids crash)
        return fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
    # Default: assume USD
    return fetch_usd_to_pln_exchange_rate(s_ext, e_ext)


def convert_price_series_to_pln(native_px: pd.Series, quote_ccy: str, start: Optional[str], end: Optional[str]) -> Tuple[pd.Series, str]:
    """Convert a native price series quoted in quote_ccy into PLN.
    Returns (pln_series, units_suffix).
    """
    sfx = "(PLN)"
    native_px = _ensure_float_series(native_px)
    # Get FX leg over the native range (with padding)
    fx = convert_currency_to_pln(quote_ccy, start, end, native_index=native_px.index)
    # Try increasingly permissive alignments
    fx_al = _align_fx_asof(native_px, fx, max_gap_days=7)
    if fx_al.isna().all():
        fx_al = _align_fx_asof(native_px, fx, max_gap_days=14)
    if fx_al.isna().all():
        fx_al = _align_fx_asof(native_px, fx, max_gap_days=30)
    # Fallback: strict calendar alignment with ffill/bfill
    if fx_al.isna().all():
        fx_al = fx.reindex(native_px.index).ffill().bfill()
    fx_al = _ensure_float_series(fx_al)
    pln = (native_px * fx_al).dropna()
    pln.name = "px"
    return pln, sfx


def _resolve_symbol_candidates(asset: str) -> List[str]:
    a = asset.strip()
    u = a.upper()
    mapping = {
        # Prefer active, liquid proxies first to avoid Yahoo "possibly delisted" noise
        "GOOO": ["GOOG", "GOOGL", "GOOO"],
        "GLDW": ["GLDM", "GLD", "GLDW"],
        "SGLP": ["SGLP.L", "SGLP", "SGLP.LON"],
        "GLDE": ["GLD", "IAU", "GLDE"],
        "FACC": ["FACC.VI", "FACC"],
        "SLVI": ["SLV", "SLVP", "SLVI"],
        "TKA": ["TKA.DE", "TKA"],
        # Netflix and Novo Nordisk
        "NFLX": ["NFLX"],
        "NOVO": ["NVO", "NOVO-B.CO", "NOVOB.CO", "NOVO-B.CO"],
        # Kratos (alias to KTOS)
        "KRATOS": ["KTOS"],
        # Requested blue chips and defense/aero additions
        "RHEINMETALL": ["RHM.DE", "RHM.F"],
        "AIRBUS": ["AIR.PA", "AIR.DE"],
        "RENK": ["R3NK.DE", "RNK.DE"],
        "NORTHROP": ["NOC"],
        "NORTHROP GRUMMAN": ["NOC"],
        "NORTHRUP": ["NOC"],
        "NORTHRUP GRUNMAN": ["NOC"],
        "NVIDIA": ["NVDA"],
        "MICROSOFT": ["MSFT"],
        "APPLE": ["AAPL"],
        "AMD": ["AMD"],
        "UBER": ["UBER"],
        "TESLA": ["TSLA"],
        "VANGUARD SP 500": ["VOO", "VUSA.L"],
        "VANGARD SP 500": ["VOO", "VUSA.L"],
        "VANGUARD S&P 500": ["VOO", "VUSA.L"],
        "THALES": ["HO.PA"],
        "HENSOLDT": ["HAG.DE"],
        "SAMSUNG": ["005930.KS", "005935.KS"],
        "TKMS AG & CO": ["TKA.DE", "TKAMY"],
        # Additional defense, aerospace, and mining companies
        "BAE SYSTEMS": ["BA.L"],
        "BAE": ["BA.L"],
        "NEWMONT": ["NEM"],
        "NEWMONT CORP": ["NEM"],
        "HOWMET": ["HWM"],
        "HOWMET AEROSPACE": ["HWM"],
        "BROADCOM": ["AVGO"],
        # SPY, S&P 500, Magnificent 7, and Semiconductor ETFs
        "SPY": ["SPY"],
        "SP500": ["^GSPC", "SPY"],
        "S&P500": ["^GSPC", "SPY"],
        "S&P 500": ["^GSPC", "SPY"],
        "MAGNIFICENT 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
        "MAGNIFICENT SEVEN": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
        "MAG7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
        "GOOGLE": ["GOOGL", "GOOG"],
        "ALPHABET": ["GOOGL", "GOOG"],
        "AMAZON": ["AMZN"],
        "META": ["META"],
        "FACEBOOK": ["META"],
        "SEMICONDUCTOR ETF": ["SMH", "SOXX"],
        "SMH": ["SMH"],
        "SOXX": ["SOXX"],
        # keep identity candidates (with improved MTX mapping to MTU Aero Engines on XETRA)
        "RKLB": ["RKLB"],
        "MTX": ["MTX.DE", "MTX"],
        "IBKR": ["IBKR"],
        "HOOD": ["HOOD"],

        # S&P 100 additions by sector

        # Information Technology
        "ACCENTURE": ["ACN"],
        "ADOBE": ["ADBE"],
        "SALESFORCE": ["CRM"],
        "CISCO": ["CSCO"],
        "INTEL": ["INTC"],
        "ORACLE": ["ORCL"],
        "PALANTIR": ["PLTR"],
        "QUALCOMM": ["QCOM"],
        "TEXAS INSTRUMENTS": ["TXN"],

        # Health Care
        "ABBVIE": ["ABBV"],
        "ABBOTT LABS": ["ABT"],
        "AMGEN": ["AMGN"],
        "BRISTOL MYERS SQUIBB": ["BMY"],
        "CVS HEALTH": ["CVS"],
        "DANAHER": ["DHR"],
        "GILEAD SCIENCES": ["GILD"],
        "INTUITIVE SURGICAL": ["ISRG"],
        "JOHNSON & JOHNSON": ["JNJ"],
        "ELI LILLY": ["LLY"],
        "MEDTRONIC": ["MDT"],
        "MERCK": ["MRK"],
        "THERMO FISHER SCIENTIFIC": ["TMO"],
        "UNITEDHEALTH": ["UNH"],

        # Financials
        "AMERICAN INTERNATIONAL GROUP": ["AIG"],
        "AMERICAN EXPRESS": ["AXP"],
        "BANK OF AMERICA": ["BAC"],
        "BNY MELLON": ["BK"],
        "BLACKROCK": ["BLK"],
        "BERKSHIRE HATHAWAY": ["BRK.B"],
        "CITIGROUP": ["C"],
        "CAPITAL ONE": ["COF"],
        "GOLDMAN SACHS": ["GS"],
        "JPMORGAN CHASE": ["JPM"],
        "MASTERCARD": ["MA"],
        "METLIFE": ["MET"],
        "MORGAN STANLEY": ["MS"],
        "PAYPAL": ["PYPL"],
        "CHARLES SCHWAB": ["SCHW"],
        "US BANCORP": ["USB"],
        "VISA": ["V"],
        "WELLS FARGO": ["WFC"],

        # Consumer Discretionary
        "BOOKING HOLDINGS": ["BKNG"],
        "GENERAL MOTORS": ["GM"],
        "HOME DEPOT": ["HD"],
        "LOWES": ["LOW"],
        "MCDONALDS": ["MCD"],
        "NIKE": ["NKE"],
        "STARBUCKS": ["SBUX"],
        "TARGET": ["TGT"],

        # Industrials
        "BOEING": ["BA"],
        "CATERPILLAR": ["CAT"],
        "DEERE & COMPANY": ["DE"],
        "EMERSON ELECTRIC": ["EMR"],
        "FEDEX": ["FDX"],
        "GENERAL DYNAMICS": ["GD"],
        "GE AEROSPACE": ["GE"],
        "HONEYWELL": ["HON"],
        "LOCKHEED MARTIN": ["LMT"],
        "3M": ["MMM"],
        "RTX CORPORATION": ["RTX"],
        "UNION PACIFIC": ["UNP"],
        "UPS": ["UPS"],

        # Communication Services
        "COMCAST": ["CMCSA"],
        "DISNEY": ["DIS"],
        "AT&T": ["T"],
        "T-MOBILE": ["TMUS"],
        "VERIZON": ["VZ"],

        # Consumer Staples
        "COLGATE-PALMOLIVE": ["CL"],
        "COSTCO": ["COST"],
        "COCA-COLA": ["KO"],
        "MONDELEZ": ["MDLZ"],
        "ALTRIA": ["MO"],
        "PEPSICO": ["PEP"],
        "PROCTER & GAMBLE": ["PG"],
        "PHILIP MORRIS": ["PM"],
        "WALMART": ["WMT"],

        # Energy
        "CONOCOPHILLIPS": ["COP"],
        "CHEVRON": ["CVX"],
        "EXXONMOBIL": ["XOM"],

        # Utilities
#         "DUKE ENERGY": ["DUK"],
        "NEXTERA ENERGY": ["NEE"],
        "SOUTHERN COMPANY": ["SO"],

        # Real Estate
#         "AMERICAN TOWER": ["AMT"],
#         "SIMON PROPERTY GROUP": ["SPG"],

        # Materials
        "LINDE": ["LIN"],
        # Vaneck ETFs
        "VANECK SEMICONDUCTOR": ["SMH"],
#         "VANECK GOLD MINERS": ["GDX"],
#         "VANECK JUNIOR GOLD MINERS": ["GDXJ"],
        "VANECK OIL SERVICES": ["OIH"],
#         "VANECK RETAIL": ["RTH"],
        "VANECK AGRIBUSINESS": ["MOO"],
        "VANECK GAMING ETF": ["ESPO"],
#         "VANECK AFRICA INDEX": ["AFK"],
        "VANECK FALLEN ANGEL HIGH YIELD BOND": ["ANGL"],
#         "VANECK BRAZIL SMALL-CAP": ["BRF"],
        "VANECK CHINEXT": ["CNXT"],
        "VANECK MORNINGSTAR DURABLE DIVIDEND": ["DURA"],
        "VANECK EGYPT INDEX": ["EGPT"],
#         "VANECK JP MORGAN EM LOCAL CURRENCY BOND": ["EMLC"],
        "VANECK INVESTMENT GRADE FLOATING RATE": ["FLTR"],
        "VANECK INDIA GROWTH LEADERS": ["GLIN"],
        "VANECK MORNINGSTAR GLOBAL WIDE MOAT": ["MOTG"],
#         "VANECK GREEN BOND": ["GRNB"],
#         "VANECK EMERGING MARKETS HIGH YIELD BOND": ["HYEM"],
#         "VANECK INDONESIA INDEX": ["IDX"],
        "VANECK INTERMEDIATE MUNI": ["ITM"],
#         "VANECK LONG MUNI": ["MLN"],
#         "VANECK MORNINGSTAR WIDE MOAT": ["MOAT"],
        "VANECK MORNINGSTAR INTERNATIONAL MOAT": ["MOTI"],
#         "VANECK URANIUM+NUCLEAR ENERGY": ["NLR"],
#         "VANECK PHARMACEUTICAL": ["PPH"],
#         "VANECK RARE EARTH/STRATEGIC METALS": ["REMX"],
#         "VANECK RUSSIA": ["RSX"],
#         "VANECK RUSSIA SMALL-CAP": ["RSXJ"],
#         "VANECK STEEL": ["SLX"],
#         "VANECK LOW CARBON ENERGY": ["SMOG"],
#         "VANECK VIETNAM": ["VNM"],
#         "VANECK GLOBAL FALLEN ANGEL HIGH YIELD BOND UCITS": ["GFA"],
#         "VANECK HYDROGEN ECONOMY UCITS": ["HDRO"],
#         "VANECK IBOXX EUR CORPORATES UCITS": ["TCBT"],
#         "VANECK MORNINGSTAR DEVELOPED MARKETS DIVIDEND LEADERS UCITS": ["TDIV"],
#         "VANECK SUSTAINABLE EUROPEAN EQUAL WEIGHT UCITS": ["TEET"],
#         "VANECK IBOXX EUR SOVEREIGN DIVERSIFIED 1-10 UCITS": ["TGBT"],
#         "VANECK GLOBAL REAL ESTATE UCITS": ["TRET"],
#         "VANECK SUSTAINABLE WORLD EQUAL WEIGHT UCITS": ["TSWE"],
#         "VANECK IBOXX EUR SOVEREIGN CAPPED AAA-AA 1-5 UCITS": ["TAT"],
    }
    # For known special assets already handled elsewhere, leave as-is
    special = {"PLNJPY=X", "BTC-USD", "BTCUSD=X", "MSTR", "GC=F", "SI=F", "XAU=X", "XAG=X", "XAUUSD=X", "XAGUSD=X"}
    if u in special:
        return [u]
    if u in MAPPING:
        return MAPPING[u]
    return [u]
