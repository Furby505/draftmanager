"""
DK ADP -> projection-universe match audit.

Run: python training/audit_dk_adp_match.py
Writes: data/processed/dk_adp_match_audit.md
Exits NON-ZERO (fails loudly) if any top-250-by-ADP DK player is unmatched.

Tells you, for every unmatched DK ADP player, whether it's an irrelevant
deep-tail guy or a useful player that should be on the board, and (for the
useful ones) whether it's a recoverable name issue or genuinely absent from the
projection universe (needs projection coverage, not a matching fix).
"""

import json
import sys
from pathlib import Path

from fetch_dk_adp import (PROJECTIONS, _index_projections, _resolve_default_source,
                          _read_rows, _surname, match_to_projections)

OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "dk_adp_match_audit.md"
TOP_N = 250


def main():
    src = _resolve_default_source()
    if src is None:
        print("No DK ADP source found; nothing to audit.")
        sys.exit(0)
    rows = _read_rows(src)
    proj = json.loads(PROJECTIONS.read_text())
    mapping, matched, unmatched = match_to_projections(rows, proj)

    # Rank every DK player by ADP (1 = earliest).
    ranked = sorted(rows, key=lambda r: r["adp"])
    rank_of = {id(r): i + 1 for i, r in enumerate(ranked)}

    # For each unmatched, suggest whether a same-surname+pos projection exists
    # (i.e., likely a name/team variant) vs truly absent.
    _, by_spt = _index_projections(proj)
    by_surpos = {}
    for rec in proj:
        from outcome_model import normalize_name
        nm = normalize_name(rec["player_display_name"])
        by_surpos.setdefault((_surname(nm), str(rec.get("position", "")).upper()),
                             []).append(rec)

    un = []
    for r in unmatched:
        cands = by_surpos.get((_surname(r["nm"]), str(r["pos"] or "").upper()), [])
        suggestion = ("; ".join(f"{c['player_display_name']} ({c['recent_team']})"
                                for c in cands[:3]) if cands else "")
        un.append({**r, "rank": rank_of[id(r)],
                   "category": "name/team variant?" if cands else "absent from projections",
                   "suggestion": suggestion})
    un.sort(key=lambda x: x["rank"])
    top_un = [u for u in un if u["rank"] <= TOP_N]

    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    emit(f"# DK ADP -> Projection Match Audit ({src.name})")
    emit(f"- DK ADP players: **{len(rows)}**")
    emit(f"- Matched to projections: **{len(matched)}** "
         f"(of which {sum(1 for m in matched if m['how'].startswith('alias'))} via "
         f"surname+pos+team alias)")
    emit(f"- Unmatched: **{len(unmatched)}**  |  unmatched in top {TOP_N} by ADP: "
         f"**{len(top_un)}**\n")

    emit("## Alias matches recovered (name variants now on the board)")
    aliases = [m for m in matched if m["how"].startswith("alias")]
    if aliases:
        emit("| DK name | -> projection | pos | team |")
        emit("|---|---|---|---|")
        for m in sorted(aliases, key=lambda x: x["adp"]):
            emit(f"| {m['nm']} | {m['proj_name']} | {m['pos']} | {m['team']} |")
    else:
        emit("_none_")

    emit("\n## Unmatched DK players (all)")
    emit("| ADP rank | adp | name | pos | team | category | possible projection |")
    emit("|---|---|---|---|---|---|---|")
    for u in un:
        emit(f"| {u['rank']} | {u['adp']:.1f} | {u['nm']} | {u['pos']} | {u['team']} "
             f"| {u['category']} | {u['suggestion']} |")

    emit("\n## Verdict")
    if top_un:
        absent = [u for u in top_un if u["category"] == "absent from projections"]
        variant = [u for u in top_un if u["category"] != "absent from projections"]
        emit(f"- **{len(top_un)} top-{TOP_N} DK players are UNMATCHED** — these are "
             "draftable players the board cannot represent.")
        if variant:
            emit(f"  - {len(variant)} look like recoverable name/team variants "
                 "(matching bug — investigate the suggested projection).")
        if absent:
            emit(f"  - {len(absent)} are absent from the projection universe "
                 "(FA/uncovered rookies): "
                 + ", ".join(f"{u['nm']}({u['pos']},{u['team']},#{u['rank']})"
                             for u in absent))
            emit("  - These need PROJECTION coverage, not a matching fix. Most "
                 "impactful to add: "
                 + ", ".join(u["nm"] for u in absent[:5]) + ".")
    else:
        emit(f"- No top-{TOP_N} DK players unmatched. Remaining unmatched are "
             "deep-tail / FA dart throws.")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    emit(f"\nWrote {OUT}")

    if top_un:
        print(f"\n*** FAIL: {len(top_un)} top-{TOP_N} DK ADP players unmatched "
              "(see audit). ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
