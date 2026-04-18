"""Fix corrupted s041 function in trend_momentum.py - remove duplicate code/docstring."""
filepath = "src/indicators/families/trend_momentum.py"
with open(filepath, "r") as f:
    content = f.read()

# The corruption: after the function header and initial variable setup,
# there's a duplicate block: first pp/std_r1 calculation, then raw docstring text,
# then the real close/high/low/volume re-assignment and proper code.
# We need to remove from "pp = (prev_h..." through to the second "close = ind["close"]"

old = '''    # --- Standard Pivots ---
    pp = (prev_h + prev_l + prev_c) / 3
    std_r1 = 2 * pp - prev_l

    Chicago Floor Traders: Support/Resistance from universal pivot calculations.

    Three pivot systems computed from prior bar's H/L/C:
      Standard:  PP = (H+L+C)/3, R1 = 2PP-L, S1 = 2PP-H, R2 = PP+(H-L), S2 = PP-(H-L),
                 R3 = H+2(PP-L), S3 = L-2(H-PP)
      Fibonacci: R1 = PP+0.382*(H-L), S1 = PP-0.382*(H-L), R2/S2 at 0.618, R3/S3 at 1.000
      Camarilla: R4 = C+1.1*(H-L)/2, S4 = C-1.1*(H-L)/2

    Confluence = zone where 2+ pivot types converge within 0.3% band.
    Confluence_Strength = count of converging levels at that zone.

    Signal:
      Long  at support confluence: price near support zone with strength >= 3
      Short at resistance confluence: price near resistance zone with strength >= 3
      Breakout: price closes beyond R3/S3 with volume = trend continuation
    """
    close = ind["close"]
    high = ind["high"]
    low = ind["low"]
    volume = ind["volume"]

    prev_h = high.shift(1)
    prev_l = low.shift(1)
    prev_c = close.shift(1)
    prev_range = prev_h - prev_l

    # --- Standard Pivots ---
    pp = (prev_h + prev_l + prev_c) / 3
    std_r1 = 2 * pp - prev_l
    std_s1 = 2 * pp - prev_h'''

new = '''    # --- Standard Pivots ---
    pp = (prev_h + prev_l + prev_c) / 3
    std_r1 = 2 * pp - prev_l
    std_s1 = 2 * pp - prev_h'''

assert old in content, "S041 corruption pattern not found!"
content = content.replace(old, new, 1)

# Now check if there's a second duplicate section further down
# The function should end with return _clip_signal(raw) and then the raw is computed once
# Let me also check for the second duplicate of the pivot code
# The original function had the pivots computed twice - find and remove the second copy

# Check for "fib_s3 = pp - 1.000 * prev_range\n\n    # --- Camarilla" appearing twice
count = content.count("fib_s3 = pp - 1.000 * prev_range")
if count > 1:
    print(f"WARNING: Found {count} occurrences of fib_s3 computation - need to fix duplicate")
    # Find the second block that starts with "    fib_s3 = pp - 1.000 * prev_range"
    # and continues through to the second "return _clip_signal(raw)" for s041
    
    # Find position of the second duplicate block
    # Pattern: after the first "raw = (sup_recent - res_recent" block, 
    # there's "fib_s3 = pp - 1.000" which starts the duplicate
    
    marker1 = "    raw = (sup_recent - res_recent\n           + breakout_signal\n           + above_pp * 10\n           + ind[\"mom_score\"] * 15)\n\n    fib_s3 = pp - 1.000 * prev_range"
    if marker1 in content:
        # Find where the duplicate ends (at the second return _clip_signal for this function)
        # Replace the duplicate block
        marker2 = "    fib_s3 = pp - 1.000 * prev_range\n\n    # --- Camarilla Pivots ---"
        idx = content.find(marker1)
        if idx >= 0:
            # We need to keep "raw = ..." and "return _clip_signal(raw)" 
            # but remove everything between them that's duplicate
            # Actually the duplicate starts right after the first raw calculation
            
            # Find the end: the second "return _clip_signal(raw)" in s041
            after_first_raw = content.find("    return _clip_signal(raw)\n\n\ndef s042_", idx)
            if after_first_raw >= 0:
                # Everything from "fib_s3 = pp" to the second "return _clip_signal(raw)\n\ndef s042_"
                # needs to be trimmed to just "return _clip_signal(raw)\n\ndef s042_"
                dup_start = content.find("\n    fib_s3 = pp - 1.000 * prev_range\n\n    # --- Camarilla", idx)
                if dup_start >= 0:
                    dup_end = after_first_raw + len("    return _clip_signal(raw)")
                    content = content[:dup_start] + "\n\n    return _clip_signal(raw)" + content[dup_end:]
                    print("Removed duplicate s041 code block")
else:
    print("No fib_s3 duplicate found - good")

with open(filepath, "w") as f:
    f.write(content)

print("S041 corruption fix complete")
