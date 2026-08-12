"""Stream the (5.2 GB) Best Ball Mania V round-1 CSV from stdin and keep only the
first N complete drafts, written to a local CSV. Exits as soon as N drafts are
collected so curl only downloads a fraction of the file.

Usage:
    curl -s <URL> | python training/sample_bbm.py <N> <out.csv>
"""
import csv
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/raw/bbm_v_sample.csv"

reader = csv.reader(sys.stdin)
header = next(reader)
di = header.index("draft_id")

seen_order = []          # draft_ids in first-seen order
rows_by_draft = {}
current = None

for row in reader:
    if not row:
        continue
    d = row[di]
    if d != current:
        current = d
        if d not in rows_by_draft:
            if len(seen_order) >= N:
                break     # we already have N full drafts; the (N+1)th started
            seen_order.append(d)
            rows_by_draft[d] = []
    rows_by_draft[d].append(row)

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    for d in seen_order:
        w.writerows(rows_by_draft[d])

print(f"Collected {len(seen_order)} complete drafts -> {OUT}")
print(f"Total rows: {sum(len(v) for v in rows_by_draft.values())}")
