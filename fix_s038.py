"""Fix corrupted s037/s038 boundary in trend_momentum.py."""
filepath = "src/indicators/families/trend_momentum.py"
with open(filepath, "r") as f:
    content = f.read()

old = '''    raw = raw.rolling(2nt ---
    prev_close = close.shift(1)'''

new = '''    return _clip_signal(raw)


def s038_gap_analysis(ind):
    """038 | Hong Kong HSI Opening Gap Strategy
    School: Hong Kong (HKEX Proprietary)

    Gap = Open - Close_prev.  Gap_Size_Z = Gap_Pct / StdDev(Gap_Pct, 60).
    Fill_Probability = logistic(Gap_Size_Z).
    Small gaps (fill_prob > 0.65): fade gap direction.
    Large gaps (fill_prob < 0.35, Z > 2): trade WITH gap if momentum confirms.
    """
    close = ind["close"]
    open_p = ind["open"]
    high = ind["high"]
    low = ind["low"]
    prev_close = close.shift(1)'''

assert old in content, f"Old text not found!"
content = content.replace(old, new, 1)

with open(filepath, "w") as f:
    f.write(content)

print("s037/s038 boundary fixed OK")
