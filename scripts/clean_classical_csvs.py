#!/usr/bin/env python3
"""Clean classical bounds CSVs by keeping the last row per (cipher, rounds).
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results' / 'thesis'

for fname in ['simon_classical_bounds.csv', 'speck_classical_bounds.csv']:
    p = RES / fname
    if not p.exists():
        continue
    rows = []
    with p.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # keep last occurrence per (cipher, rounds)
    kept = {}
    for r in rows:
        key = (r['cipher'], r['rounds'])
        kept[key] = r
    out_rows = list(kept.values())
    # sort by rounds
    out_rows.sort(key=lambda r: (r['cipher'], int(r['rounds'])))
    with p.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['cipher','rounds','max_characteristic_prob','delta_left','delta_right'])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)
    print(f'Cleaned {p}; kept {len(out_rows)} rows')
