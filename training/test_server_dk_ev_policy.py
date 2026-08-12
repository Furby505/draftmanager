"""
Smoke checks for the DK EV policy wiring in server/server.py.

These tests call the FastAPI handler in-process. They do not touch the Chrome
extension and do not start a server.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import server.server as srv  # noqa: E402
from pydantic import ValidationError  # noqa: E402


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


class SpyModel:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.called = False

    def predict(self, X):
        self.called = True
        return self.wrapped.predict(X)


def dk_available(limit: int = 90, extras: tuple[str, ...] = ()) -> list[str]:
    names: list[str] = []
    for name, _ in sorted(srv.DK_ADP_INDEX.items(), key=lambda kv: kv[1]):
        p = srv.match_player(name)
        if p:
            nm = p.get("player_display_name")
            if nm and nm not in names:
                names.append(nm)
        if len(names) >= limit:
            break
    for name in extras:
        p = srv.match_player(name)
        if p:
            nm = p.get("player_display_name")
            if nm and nm not in names:
                names.append(nm)
    return names


def main() -> int:
    check("DK EV policy artifact loaded", srv.DK_EV_POLICY is not None)
    check("DK policy has full feature set", len(srv.DK_EV_FEATURE_COLS) >= 70,
          str(len(srv.DK_EV_FEATURE_COLS)))
    check("half-PPR DK EV policy artifact loaded",
          srv.DK_EV_POLICY_ARTIFACTS["half"]["model"] is not None)
    check("full-PPR scoring selects current served artifact",
          srv.dk_policy_artifact_for_scoring("full")["model_path"].name == "model_dk_ev_policy.joblib")
    check("half-PPR scoring selects half artifact",
          srv.dk_policy_artifact_for_scoring("half")["model_path"].name == "model_dk_ev_policy_halfppr.joblib")
    check("half-PPR scoring aliases normalize",
          srv.dk_policy_artifact_for_scoring("0.5ppr")["model_path"].name == "model_dk_ev_policy_halfppr.joblib")
    check("full-PPR scoring aliases normalize",
          srv.dk_policy_artifact_for_scoring("ppr")["model_path"].name == "model_dk_ev_policy.joblib")
    check("full-PPR trained max rank preserved",
          srv._dk_policy_trained_candidate_pool(srv.DK_EV_POLICY_ARTIFACTS["full"]) == 8)
    check("half-PPR trained max rank preserved",
          srv._dk_policy_trained_candidate_pool(srv.DK_EV_POLICY_ARTIFACTS["half"]) == 8)
    check("DK ADP loaded", len(srv.DK_ADP_INDEX) >= 300, str(len(srv.DK_ADP_INDEX)))
    check("Underdog ADP loaded", len(srv.UNDERDOG_ADP_INDEX) >= 250,
          str(len(srv.UNDERDOG_ADP_INDEX)))
    check("configured candidate pool starts at 12", srv._dk_candidate_pool_depth(1, 12) == 12)
    check("configured candidate pool expands late", srv._dk_candidate_pool_depth(169, 12) == 32)
    check("served pool is capped by trained artifact depth",
          srv._dk_effective_candidate_pool_depth(169, 12) <= srv.DK_TRAINED_CANDIDATE_POOL,
          str(srv.DK_TRAINED_CANDIDATE_POOL))
    by_pos = {pos: [] for pos in srv.DK_POS}
    for name, _ in sorted(srv.DK_ADP_INDEX.items(), key=lambda kv: kv[1]):
        p = srv.match_player(name)
        if p and p.get("position") in by_pos:
            by_pos[p["position"]].append(p)
        if all(len(v) >= 3 for v in by_pos.values()):
            break
    rbs = by_pos["RB"]
    if len(rbs) >= 3:
        gap = srv._dk_same_pos_replacement_gap(
            rbs[1], rbs, srv.normalize(rbs[1].get("player_display_name", ""))
        )
        expected = srv._dk_adp(rbs[2]) - srv._dk_adp(rbs[1])
        check("server same-position replacement gap uses next worse player",
              abs(gap - expected) < 1e-9 and gap >= 0,
              f"gap={gap} expected={expected}")
    else:
        check("server same-position replacement gap fixture", False, f"RBs={len(rbs)}")
    soft_ranks = srv._dk_live_candidate_ranks(
        by_pos["RB"][:2] + by_pos["WR"][:2],
        {"QB": 1, "RB": 7, "WR": 3, "TE": 1},
        picks_left=10,
    )
    check("server live pool keeps over-soft positions available",
          all(srv.normalize(p.get("player_display_name", "")) in soft_ranks for p in by_pos["RB"][:2]),
          str(soft_ranks))
    min_ranks = srv._dk_live_candidate_ranks(
        by_pos["RB"][:1] + by_pos["TE"][:1],
        {"QB": 1, "RB": 2, "WR": 3, "TE": 0},
        picks_left=1,
    )
    check("server legal pool forces unmet minimums in final picks",
          set(min_ranks) == {srv.normalize(by_pos["TE"][0].get("player_display_name", ""))},
          str(min_ranks))

    names = dk_available(80, ("Parker Washington",))
    req = srv.RankRequest(
        available_players=names,
        drafted_players=[],
        current_pick=1,
        total_teams=12,
        total_rounds=20,
        my_pick_position=1,
    )
    empty_half_resp = srv.rank_players(req.model_copy(update={"available_players": [], "scoring": "half"}))
    check("empty half-PPR response preserves scoring",
          empty_half_resp.scoring == "half" and empty_half_resp.dk_policy_model == "",
          f"{empty_half_resp.scoring} {empty_half_resp.dk_policy_model}")
    resp = srv.rank_players(req)
    half_resp = srv.rank_players(req.model_copy(update={"scoring": "half"}))
    top = resp.recommendations[0]
    half_top = half_resp.recommendations[0]
    check("full-PPR response reports served artifact",
          resp.scoring == "full"
          and resp.dk_policy_label == "full-PPR"
          and resp.dk_policy_model == "model_dk_ev_policy.joblib"
          and resp.dk_policy_max_candidate_rank == 8,
          f"{resp.scoring} {resp.dk_policy_label} {resp.dk_policy_model} {resp.dk_policy_max_candidate_rank}")
    check("half-PPR response reports half artifact",
          half_resp.scoring == "half"
          and half_resp.dk_policy_label == "half-PPR"
          and half_resp.dk_policy_model == "model_dk_ev_policy_halfppr.joblib"
          and half_resp.dk_policy_max_candidate_rank == 8,
          f"{half_resp.scoring} {half_resp.dk_policy_label} {half_resp.dk_policy_model} {half_resp.dk_policy_max_candidate_rank}")
    srv.set_platform({"platform": "underdog"})
    try:
        underdog_resp = srv.rank_players(req.model_copy(update={"scoring": "full", "total_rounds": 18}))
        check("Underdog platform forces half-PPR artifact",
              underdog_resp.scoring == "half"
              and underdog_resp.dk_policy_label == "half-PPR"
              and underdog_resp.dk_policy_model == "model_dk_ev_policy_halfppr.joblib",
              f"{underdog_resp.scoring} {underdog_resp.dk_policy_label} {underdog_resp.dk_policy_model}")
        check("Underdog platform still uses EV policy",
              underdog_resp.recommendations[0].ranking_source == "dk_ev_policy",
              underdog_resp.recommendations[0].ranking_source)
        underdog_top = underdog_resp.recommendations[0]
        underdog_source_adp = srv.UNDERDOG_ADP_INDEX.get(srv.normalize(underdog_top.name), 0.0)
        check("Underdog platform displays Underdog ADP",
              bool(underdog_source_adp) and abs(underdog_top.adp - underdog_source_adp) < 0.05,
              f"{underdog_top.name} displayed={underdog_top.adp} source={underdog_source_adp}")
        page_adp_resp = srv.rank_players(req.model_copy(update={
            "scoring": "half",
            "total_rounds": 18,
            "page_adp": {"Ja'Marr Chase": 6.6},
        }))
        chase = next((r for r in page_adp_resp.recommendations if r.name == "Ja'Marr Chase"), None)
        check("Underdog page ADP overrides saved board",
              chase is not None
              and abs(chase.adp - 6.6) < 0.05
              and page_adp_resp.adp_source == "page+underdog_adp"
              and page_adp_resp.page_adp_count == 1,
              f"adp={getattr(chase, 'adp', None)} source={page_adp_resp.adp_source} count={page_adp_resp.page_adp_count}")
        bad_page_adp_resp = srv.rank_players(req.model_copy(update={
            "scoring": "half",
            "total_rounds": 18,
            "page_adp": {"CeeDee Lamb": 1.0},
        }))
        bad_ceedee = next((r for r in bad_page_adp_resp.recommendations if r.name == "CeeDee Lamb"), None)
        check("Underdog bad page ADP is rejected",
              bad_ceedee is not None
              and abs(bad_ceedee.adp - srv.UNDERDOG_ADP_INDEX[srv.normalize("CeeDee Lamb")]) < 0.05
              and bad_page_adp_resp.page_adp_count == 0,
              f"adp={getattr(bad_ceedee, 'adp', None)} count={bad_page_adp_resp.page_adp_count}")
    finally:
        srv.set_platform({"platform": "draftkings"})
    original_half_model = srv.DK_EV_POLICY_ARTIFACTS["half"]["model"]
    spy_half_model = SpyModel(original_half_model)
    srv.DK_EV_POLICY_ARTIFACTS["half"]["model"] = spy_half_model
    try:
        srv.rank_players(req.model_copy(update={"scoring": "half"}))
        check("half-PPR rank invokes half artifact model", spy_half_model.called)
    finally:
        srv.DK_EV_POLICY_ARTIFACTS["half"]["model"] = original_half_model
    parker_rank = next(
        (i for i, r in enumerate(resp.recommendations, start=1) if r.name.lower() == "parker washington"),
        None,
    )
    check("rank uses DK EV policy", top.ranking_source == "dk_ev_policy", top.ranking_source)
    check("half-PPR rank uses DK EV policy",
          half_top.ranking_source == "dk_ev_policy", half_top.ranking_source)
    check("pick 1 stays in current ADP candidate pool", top.adp <= 12.0, f"{top.name} adp={top.adp}")
    check("Parker is not a first-pick recommendation", parker_rank is None or parker_rank > 20,
          f"rank={parker_rank}")

    available = [srv.match_player(name) for name in names]
    available = [p for p in available if p]
    top_player = srv.match_player(top.name)
    if top_player:
        row = srv._dk_policy_feature_row(req, top, top_player, available, [])
        missing_cols = [col for col in srv.DK_EV_FEATURE_COLS if col not in row]
        check("live DK policy row emits trained features", not missing_cols,
              ",".join(missing_cols[:5]))
        half_cols = srv.DK_EV_POLICY_ARTIFACTS["half"]["feature_cols"]
        missing_half_cols = [col for col in half_cols if col not in row]
        check("live DK policy row emits half-PPR trained features", not missing_half_cols,
              ",".join(missing_half_cols[:5]))
        check("half-PPR model includes newer live features",
              {"cand_handcuff_depth", "cand_playoff_matchup"}.issubset(set(half_cols)))
        check("live DK policy row includes ADP minus current pick",
              row["cand_adp_minus_pick"] == row["cand_adp"] - req.current_pick,
              str(row.get("cand_adp_minus_pick")))
    else:
        check("live DK policy feature fixture", False, top.name)

    def rec(name: str, pos: str, adp: float) -> srv.PlayerRec:
        return srv.PlayerRec(
            name=name,
            position=pos,
            team="TST",
            proj_points=0.0,
            proj_boom_rate=0.0,
            vor=0.0,
            age=0.0,
            value_tier="test",
            adp=adp,
            dk_ev_score=0.12,
            dk_ev_rank=0,
            ranking_source="dk_ev_policy",
        )

    # The indifference-band RERANKER was reverted 2026-06-17: a paired backtest
    # (training/paired_tiebreaker.py) showed it is neutral at best and significantly
    # harmful as the band widens, because the model already trains on survival/need/
    # stack/bye. apply_dk_ev_tiebreaker now only annotates margins; model order is kept.
    tied = [
        rec("Likely Later WR", "WR", 100.0),
        rec("Urgent RB", "RB", 5.0),
        rec("Also Later TE", "TE", 110.0),
    ]
    annotated = srv.apply_dk_ev_tiebreaker(req, tied, [])
    check("DK EV margins do NOT reorder model picks", [r.name for r in annotated] == [r.name for r in tied],
          ",".join(r.name for r in annotated))
    check("DK EV annotate refreshes displayed ranks",
          [r.dk_ev_rank for r in annotated] == list(range(1, len(annotated) + 1)),
          str([r.dk_ev_rank for r in annotated]))
    check("DK EV annotate sets margins",
          annotated[0].dk_ev_margin_to_top == 0.0 and annotated[-1].dk_ev_margin_to_next == 0.0,
          str([(r.dk_ev_margin_to_top, r.dk_ev_margin_to_next) for r in annotated]))

    legacy_req = req.model_copy(update={"use_dk_ev_policy": False})
    legacy_resp = srv.rank_players(legacy_req)
    check("policy can be disabled", legacy_resp.recommendations[0].ranking_source != "dk_ev_policy",
          legacy_resp.recommendations[0].ranking_source)

    qb_names = [
        p["player_display_name"]
        for p in srv.ALL_PLAYERS
        if p.get("position") == "QB"
    ][:5]
    capped_picks = [
        srv.DraftedPlayer(name=name, position="QB", round=i + 1, by_user=True)
        for i, name in enumerate(qb_names)
    ]
    capped_resp = srv.rank_players(req.model_copy(update={"drafted_players": capped_picks}))
    check("DK policy respects QB hard cap",
          all(r.position != "QB" for r in capped_resp.recommendations[:20]),
          ",".join(r.name for r in capped_resp.recommendations[:5]))

    roster_only_req = req.model_copy(update={
        "drafted_players": [],
        "team_rosters": {
            "1": [{"name": top.name, "position": top.position, "round": 1}],
        },
    })
    roster_only_resp = srv.rank_players(roster_only_req)
    check("team_rosters remove drafted players from recommendations",
          all(r.name != top.name for r in roster_only_resp.recommendations),
          top.name)

    try:
        srv.RankRequest(available_players=names, total_teams=12, my_pick_position=13)
    except ValidationError:
        invalid_rejected = True
    else:
        invalid_rejected = False
    check("invalid draft geometry is rejected", invalid_rejected)

    try:
        srv.RankRequest(available_players=names, scoring="mystery")
    except ValidationError:
        invalid_scoring_rejected = True
    else:
        invalid_scoring_rejected = False
    check("invalid scoring is rejected", invalid_scoring_rejected)

    print()
    if check.failed:
        print(f"{check.failed} CHECK(S) FAILED")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
