import re
import csv
import sys
from pathlib import Path

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "my.log"
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "training_metrics.csv"

PATTERN = re.compile(
    r"End of one epoch.*?"
    r"train_loss=([\d.]+).*?"
    r"eval_loss=([\d.]+).*?"
    r"eval_psnr=([-\d.]+).*?"
    r"eval_ssim=([\d.]+)"
    r"(?:.*?eval_mae=([\d.]+))?"
    r"(?:.*?eval_rmse=([\d.]+))?"
    r"(?:.*?eval_zncc=([\d.]+))?",
    re.DOTALL,
)

COLUMNS = ["epoch", "train_loss", "eval_loss", "eval_psnr", "eval_ssim",
           "eval_mae", "eval_rmse", "eval_zncc"]

rows = []
epoch = 1

with open(LOG_FILE, "r") as f:
    content = f.read()

# Match line by line for robustness
line_pattern = re.compile(
    r"End of one epoch.*?"
    r"train_loss=([-\d.]+).*?"
    r"eval_loss=([-\d.]+).*?"
    r"eval_psnr=([-\d.]+).*?"
    r"eval_ssim=([-\d.]+)"
    r"(?:.*?eval_mae=([-\d.]+))?"
    r"(?:.*?eval_rmse=([-\d.]+))?"
    r"(?:.*?eval_zncc=([-\d.]+))?"
)

for line in open(LOG_FILE):
    m = line_pattern.search(line)
    if m:
        rows.append({
            "epoch":      epoch,
            "train_loss": m.group(1),
            "eval_loss":  m.group(2),
            "eval_psnr":  m.group(3),
            "eval_ssim":  m.group(4),
            "eval_mae":   m.group(5) or "",
            "eval_rmse":  m.group(6) or "",
            "eval_zncc":  m.group(7) or "",
        })
        epoch += 1

with open(OUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Parsed {len(rows)} epochs → {OUT_FILE}")