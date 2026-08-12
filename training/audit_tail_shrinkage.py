"""
Tail-shrinkage audit — manual-inspection report.

Run: python training/audit_tail_shrinkage.py
Writes: data/processed/tail_shrinkage_audit.md  (and prints a summary)

Shows, at each player's HISTORICAL level (anchoring disabled to isolate SHAPE):
  - aggregate p50/p90/p95/p99 change by position x sample-size bucket,
  - the players whose p99 ceiling drops the most (the fake small-sample tails),
  - established studs, to confirm their real ceiling is largely preserved.
"""

from pathlib import Path

import numpy as np

from outcome_model import CorrelatedOutcomeModel, SHRINK_PRIOR

OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "tail_shrinkage_audit.md"


def pct(grid: np.ndarray, p: float) -> float:
    """p-th percentile (0-100) of a quantile grid."""
    return float(grid[int(round(p / 100.0 * (len(grid) - 1)))])


def bucket(n: int) -> str:
    if n < 20:
        return "small (10-19)"
    if n < 40:
        return "medium (20-39)"
    return "large (40+)"


def main():
    # tail_shrink builds both _raw_own and _own; anchor off to isolate shape.
    m = CorrelatedOutcomeModel(anchor_to_projection=False, tail_shrink=True).build()

    rows = []
    for pid, raw in m._raw_own.items():
        shr = m._own[pid]
        n = m._n_weeks[pid]
        mean = float(raw.mean())
        rows.append(dict(
            name=m._display.get(pid, pid), pos=m._pos_of.get(pid, "?"), n=n,
            mean=mean, bucket=bucket(n),
            raw_p95=pct(raw, 95), shr_p95=pct(shr, 95),
            raw_p99=pct(raw, 99), shr_p99=pct(shr, 99),
            raw_p99_ratio=pct(raw, 99) / mean if mean > 0 else 0,
            shr_p99_ratio=pct(shr, 99) / mean if mean > 0 else 0,
        ))
    for r in rows:
        r["drop_p99_pct"] = (r["raw_p99"] - r["shr_p99"]) / r["raw_p99"] if r["raw_p99"] else 0
        r["drop_p95_pct"] = (r["raw_p95"] - r["shr_p95"]) / r["raw_p95"] if r["raw_p95"] else 0

    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    emit(f"# Tail-Shrinkage Audit (SHRINK_PRIOR={SHRINK_PRIOR})")
    emit(f"_Players with own history: {len(rows)}. Level held at historical mean "
         f"(anchoring off) to isolate shape._\n")

    # ---- Aggregate by position x sample-size bucket ----------------------
    emit("## Mean ceiling drop by position x sample-size bucket")
    emit("Expect: small-sample tails drop hard, large-sample studs barely move.\n")
    emit("| Position | Bucket | n players | mean p95 drop | mean p99 drop |")
    emit("|---|---|---|---|---|")
    for pos in ("QB", "RB", "WR", "TE"):
        for b in ("small (10-19)", "medium (20-39)", "large (40+)"):
            cell = [r for r in rows if r["pos"] == pos and r["bucket"] == b]
            if not cell:
                continue
            emit(f"| {pos} | {b} | {len(cell)} | "
                 f"{np.mean([r['drop_p95_pct'] for r in cell]):.1%} | "
                 f"{np.mean([r['drop_p99_pct'] for r in cell]):.1%} |")

    # ---- Most-affected (fake tails being killed) -------------------------
    emit("\n## Top 20 ceiling reductions (the fake tails we're killing)")
    emit("| Player | Pos | n | hist mean | raw p99 | shrunk p99 | p99 drop | raw p99/mean -> shrunk |")
    emit("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: x["drop_p99_pct"], reverse=True)[:20]:
        emit(f"| {r['name']} | {r['pos']} | {r['n']} | {r['mean']:.1f} | "
             f"{r['raw_p99']:.1f} | {r['shr_p99']:.1f} | {r['drop_p99_pct']:.0%} | "
             f"{r['raw_p99_ratio']:.2f}x -> {r['shr_p99_ratio']:.2f}x |")

    # ---- Established studs (must keep ceiling) ----------------------------
    emit("\n## Established studs (largest sample, highest mean) — should barely move")
    studs = [r for r in rows if r["n"] >= 40]
    studs = sorted(studs, key=lambda x: x["mean"], reverse=True)[:15]
    emit("| Player | Pos | n | hist mean | raw p95 | shrunk p95 | p95 drop | raw p99 | shrunk p99 |")
    emit("|---|---|---|---|---|---|---|---|---|")
    for r in studs:
        emit(f"| {r['name']} | {r['pos']} | {r['n']} | {r['mean']:.1f} | "
             f"{r['raw_p95']:.1f} | {r['shr_p95']:.1f} | {r['drop_p95_pct']:.0%} | "
             f"{r['raw_p99']:.1f} | {r['shr_p99']:.1f} |")

    # ---- Sanity headline numbers -----------------------------------------
    small = [r for r in rows if r["n"] < 20]
    large = [r for r in rows if r["n"] >= 40]
    emit("\n## Headline")
    emit(f"- Small-sample (n<20): mean p99 drop = "
         f"{np.mean([r['drop_p99_pct'] for r in small]):.1%} "
         f"(n={len(small)} players)")
    emit(f"- Large-sample (n>=40): mean p99 drop = "
         f"{np.mean([r['drop_p99_pct'] for r in large]):.1%} "
         f"(n={len(large)} players)")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
