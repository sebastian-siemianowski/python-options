import os
d = "src/arena/experimental_models"
os.makedirs(d, exist_ok=True)
c = open("model_template.txt").read() if os.path.exists("model_template.txt") else ""
if not c:
    print("Need model_template.txt")
else:
    with open(f"{d}/hyv_01.py", "w") as f: f.write(c)
    print("Created")
