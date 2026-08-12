"""
Fit the fast DK EV pick policy from rollout labels.

This distills the expensive rollout simulator into a server-ready regressor.
It does not use the legacy team-value model or old projection-board ranks. The
target is within-state EV edge, so the model learns which candidate is better
than the other choices available at the same pick.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_policy import CANDIDATE_DEPTH_SCHEDULE, DEFAULT_CANDIDATES, FEATURE_COLS


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "data" / "processed" / "dk_ev_rollouts.csv"
DEFAULT_MODEL_OUT = ROOT / "server" / "models" / "model_dk_ev_policy.joblib"
DEFAULT_FEATURES_OUT = ROOT / "server" / "models" / "dk_ev_policy_feature_cols.json"
DEFAULT_META_OUT = ROOT / "server" / "models" / "dk_ev_policy_meta.json"
DEFAULT_AUDIT_OUT = ROOT / "data" / "processed" / "dk_ev_policy_audit.md"
DEFAULT_PRED_OUT = ROOT / "data" / "processed" / "dk_ev_policy_holdout_predictions.csv"


def load_training_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {
        "run_seed", "state_id", "pick_no", "candidate_name", "candidate_pos",
        "candidate_team", "cand_adp", "round", "candidate_rank_in_state", "ev_mean",
        *FEATURE_COLS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["ev_mean"]).copy()
    df["_state_key"] = (
        df["run_seed"].astype(str) + ":" +
        df["state_id"].astype(str) + ":" +
        df["pick_no"].astype(str)
    )
    df["_state_avg_ev"] = df.groupby("_state_key")["ev_mean"].transform("mean")
    df["_state_max_ev"] = df.groupby("_state_key")["ev_mean"].transform("max")
    df["ev_edge"] = df["ev_mean"] - df["_state_avg_ev"]
    df["state_win"] = (df["ev_mean"] == df["_state_max_ev"]).astype(int)
    return df


def seed_holdout_split(
    df: pd.DataFrame,
    holdout_frac: float,
    random_state: int,
) -> tuple[pd.Series, pd.Series, list[int]]:
    seeds = np.array(sorted(df["run_seed"].dropna().astype(int).unique()))
    if len(seeds) < 3:
        rng = np.random.default_rng(random_state)
        mask = pd.Series(rng.random(len(df)) >= holdout_frac, index=df.index)
        return mask, ~mask, []

    n_holdout = max(1, int(round(len(seeds) * holdout_frac)))
    rng = np.random.default_rng(random_state)
    early_seeds = np.array(sorted(df.loc[df["round"].eq(1), "run_seed"].dropna().astype(int).unique()))
    forced: list[int] = []
    protected: set[int] = set()
    if len(early_seeds) >= 2 and n_holdout >= 2:
        # Keep first-round examples on both sides of the split.
        forced = [int(early_seeds[-1])]
        protected = {int(s) for s in early_seeds}
    need = max(0, n_holdout - len(forced))
    remaining = np.array([s for s in seeds if int(s) not in forced and int(s) not in protected])
    if len(remaining) < need:
        # In broad runs every seed can contain early-round states. Keep the
        # forced early seed in holdout, but relax protection so the split still
        # has enough independent run seeds to sample from.
        remaining = np.array([s for s in seeds if int(s) not in forced])
    sampled = rng.choice(remaining, size=need, replace=False)
    holdout_seeds = np.sort(np.array([*forced, *sampled], dtype=int))
    valid_mask = df["run_seed"].astype(int).isin(holdout_seeds)
    train_mask = ~valid_mask
    return train_mask, valid_mask, [int(s) for s in holdout_seeds]


def make_model(random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.035,
        max_iter=450,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.05,
        validation_fraction=0.12,
        n_iter_no_change=30,
        random_state=random_state,
    )


def infer_base_candidate_depth(df: pd.DataFrame) -> int:
    """Recover the base candidate depth used to generate rollout labels."""
    if "base_candidates" in df.columns:
        vals = pd.to_numeric(df["base_candidates"], errors="coerce").dropna()
        if not vals.empty:
            return int(vals.mode().iloc[0])
    early = df[pd.to_numeric(df["round"], errors="coerce").le(2)]
    if not early.empty:
        return int(pd.to_numeric(early["candidate_rank_in_state"], errors="coerce").max())
    return DEFAULT_CANDIDATES


def state_choice_metrics(valid: pd.DataFrame) -> dict:
    rows = []
    for _, g in valid.groupby("_state_key", sort=False):
        if len(g) < 2:
            continue
        true_best = g.loc[g["ev_mean"].idxmax()]
        pred_pick = g.loc[g["pred_edge"].idxmax()]
        adp_pick = g.loc[g["candidate_rank_in_state"].idxmin()]
        rows.append({
            "state_key": g["_state_key"].iloc[0],
            "round": int(g["round"].iloc[0]),
            "slot_in_round": int(g["slot_in_round"].iloc[0]) if "slot_in_round" in g else -1,
            "picks_until_next_pick": int(g["picks_until_next_pick"].iloc[0]) if "picks_until_next_pick" in g else -1,
            "pick_no": int(g["pick_no"].iloc[0]),
            "model_hit": int(pred_pick.name == true_best.name),
            "adp_hit": int(adp_pick.name == true_best.name),
            "model_regret": float(true_best["ev_mean"] - pred_pick["ev_mean"]),
            "adp_regret": float(true_best["ev_mean"] - adp_pick["ev_mean"]),
            "model_edge": float(pred_pick["ev_edge"]),
            "adp_edge": float(adp_pick["ev_edge"]),
            "model_name": pred_pick["candidate_name"],
            "model_pos": pred_pick["candidate_pos"],
            "model_adp": float(pred_pick["cand_adp"]),
            "model_rank": int(pred_pick["candidate_rank_in_state"]),
            "true_name": true_best["candidate_name"],
            "true_pos": true_best["candidate_pos"],
            "true_adp": float(true_best["cand_adp"]),
            "true_rank": int(true_best["candidate_rank_in_state"]),
        })
    if not rows:
        return {"state_rows": pd.DataFrame(), "summary": {}}

    choices = pd.DataFrame(rows)
    summary = {
        "states": int(len(choices)),
        "model_top1_accuracy": float(choices["model_hit"].mean()),
        "adp_top1_accuracy": float(choices["adp_hit"].mean()),
        "model_avg_regret": float(choices["model_regret"].mean()),
        "adp_avg_regret": float(choices["adp_regret"].mean()),
        "model_avg_chosen_edge": float(choices["model_edge"].mean()),
        "adp_avg_chosen_edge": float(choices["adp_edge"].mean()),
    }
    return {"state_rows": choices, "summary": summary}


def sanity_checks(valid: pd.DataFrame) -> list[dict]:
    checks = []

    early = valid[valid["round"] <= 6].copy()
    if not early.empty:
        rows = []
        for (n_qb, pos), g in early.groupby(["n_QB", "candidate_pos"]):
            rows.append({
                "n_QB": int(n_qb),
                "pos": pos,
                "rows": int(len(g)),
                "avg_pred_edge": float(g["pred_edge"].mean()),
                "avg_true_edge": float(g["ev_edge"].mean()),
            })
        checks.append({"name": "early_round_qb_pressure", "rows": rows})

    r1 = valid[valid["round"] == 1].copy()
    if not r1.empty:
        rows = []
        for pos, g in r1.groupby("candidate_pos"):
            rows.append({
                "pos": pos,
                "rows": int(len(g)),
                "avg_pred_edge": float(g["pred_edge"].mean()),
                "avg_true_edge": float(g["ev_edge"].mean()),
            })
        checks.append({"name": "round_1_by_position", "rows": rows})

    for col in ["cand_stack_depth_after_pick", "cand_pos_comfort_need_before_next", "cand_pos_next_best_adp_gap"]:
        if col in valid.columns:
            sub = valid.dropna(subset=[col]).copy()
            if len(sub) >= 100:
                sub["_bin"] = pd.qcut(sub[col].rank(method="first"), 4, labels=False, duplicates="drop")
                rows = []
                for b, g in sub.groupby("_bin"):
                    rows.append({
                        "bin": int(b),
                        "rows": int(len(g)),
                        "feature_min": float(g[col].min()),
                        "feature_max": float(g[col].max()),
                        "avg_pred_edge": float(g["pred_edge"].mean()),
                        "avg_true_edge": float(g["ev_edge"].mean()),
                    })
                checks.append({"name": col, "rows": rows})

    parker = valid[valid["candidate_name"].str.lower().eq("parker washington")]
    if not parker.empty:
        checks.append({
            "name": "parker_washington_holdout",
            "rows": [{
                "rows": int(len(parker)),
                "round_min": int(parker["round"].min()),
                "round_max": int(parker["round"].max()),
                "avg_adp": float(parker["cand_adp"].mean()),
                "avg_pred_edge": float(parker["pred_edge"].mean()),
                "avg_true_edge": float(parker["ev_edge"].mean()),
            }],
        })

    return checks


def choice_sanity_checks(choices: pd.DataFrame) -> list[dict]:
    """Choice-level holdout diagnostics grouped by snake slot.

    Row-level edge metrics can look acceptable while the distilled policy fails
    in specific draft slots. These tables make that visible in the fit audit,
    especially for 18-round half-PPR validation runs.
    """
    checks: list[dict] = []
    if choices.empty or "slot_in_round" not in choices.columns:
        return checks

    rows = []
    for slot, g in choices.groupby("slot_in_round"):
        rows.append({
            "slot": int(slot),
            "states": int(len(g)),
            "top1": float(g["model_hit"].mean()),
            "model_regret": float(g["model_regret"].mean()),
            "adp_regret": float(g["adp_regret"].mean()),
            "model_rank": float(g["model_rank"].mean()),
            "true_rank": float(g["true_rank"].mean()),
        })
    checks.append({"name": "choice_by_slot", "rows": rows})

    def pos_rows(frame: pd.DataFrame, label: str) -> None:
        if frame.empty:
            return
        out = []
        for slot, g in frame.groupby("slot_in_round"):
            row = {"slot": int(slot), "states": int(len(g))}
            for pos in ("QB", "RB", "WR", "TE"):
                row[f"model_{pos}"] = float((g["model_pos"] == pos).mean())
                row[f"true_{pos}"] = float((g["true_pos"] == pos).mean())
            out.append(row)
        checks.append({"name": label, "rows": out})

    pos_rows(choices, "choice_position_by_slot")
    pos_rows(choices[choices["round"].between(3, 5)], "early_choice_position_by_slot")
    return checks


def write_audit(path: Path, meta: dict, metrics: dict, checks: list[dict]) -> None:
    label = meta.get("label", "DK EV Policy")
    lines = [
        f"# {label} Audit",
        "",
        f"This model is trained from rollout labels for `{label}`.",
        "",
        "## Artifacts",
        f"- Model: `{meta['model_out']}`",
        f"- Features: `{meta['features_out']}`",
        f"- Target: `{meta['target']}`",
        f"- Rows: {meta['rows']:,}",
        f"- Features: {meta['features']}",
        f"- Holdout seeds: {', '.join(map(str, meta['holdout_seeds'])) or 'random row split'}",
        "",
        "## Holdout Metrics",
        f"- Edge MAE: {metrics['edge_mae']:.8f}",
        f"- Edge RMSE: {metrics['edge_rmse']:.8f}",
        f"- Edge correlation: {metrics['edge_corr']:.3f}",
        f"- Model top-1 accuracy by draft state: {metrics['model_top1_accuracy']:.1%}",
        f"- ADP baseline top-1 accuracy by draft state: {metrics['adp_top1_accuracy']:.1%}",
        f"- Model average regret: {metrics['model_avg_regret']:.8f}",
        f"- ADP baseline average regret: {metrics['adp_avg_regret']:.8f}",
        f"- Model average chosen edge: {metrics['model_avg_chosen_edge']:.8f}",
        f"- ADP baseline average chosen edge: {metrics['adp_avg_chosen_edge']:.8f}",
        "",
        "## Sanity Checks",
    ]

    for check in checks:
        lines.extend(["", f"### {check['name']}"])
        rows = check["rows"]
        if not rows:
            lines.append("_No rows._")
            continue
        cols = list(rows[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in rows:
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.8f}" if abs(val) < 0.01 else f"{val:.3f}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_policy(args) -> dict:
    df = load_training_frame(args.input)
    train_mask, valid_mask, holdout_seeds = seed_holdout_split(df, args.holdout_frac, args.random_state)
    train = df[train_mask].copy()
    valid = df[valid_mask].copy()
    if train.empty or valid.empty:
        raise ValueError("Train/validation split produced an empty side.")

    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["ev_edge"]
    X_valid = valid[FEATURE_COLS].fillna(0)
    y_valid = valid["ev_edge"]

    holdout_model = make_model(args.random_state)
    holdout_model.fit(X_train, y_train)
    valid["pred_edge"] = holdout_model.predict(X_valid)

    edge_mae = mean_absolute_error(y_valid, valid["pred_edge"])
    edge_rmse = mean_squared_error(y_valid, valid["pred_edge"]) ** 0.5
    edge_corr = float(np.corrcoef(y_valid, valid["pred_edge"])[0, 1])
    choice = state_choice_metrics(valid)
    choice_summary = {
        "states": 0,
        "model_top1_accuracy": float("nan"),
        "adp_top1_accuracy": float("nan"),
        "model_avg_regret": float("nan"),
        "adp_avg_regret": float("nan"),
        "model_avg_chosen_edge": float("nan"),
        "adp_avg_chosen_edge": float("nan"),
        **choice["summary"],
    }

    metrics = {
        "edge_mae": float(edge_mae),
        "edge_rmse": float(edge_rmse),
        "edge_corr": edge_corr,
        **choice_summary,
    }

    final_model = make_model(args.random_state)
    final_model.fit(df[FEATURE_COLS].fillna(0), df["ev_edge"])

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.model_out)
    args.features_out.write_text(json.dumps(FEATURE_COLS, indent=2), encoding="utf-8")

    pred_cols = [
        "_state_key", "run_seed", "state_id", "pick_no", "round",
        "candidate_name", "candidate_pos", "candidate_team", "cand_adp",
        "candidate_rank_in_state", "ev_mean", "ev_edge", "pred_edge",
    ]
    pred_cols = list(dict.fromkeys(c for c in [*pred_cols, *FEATURE_COLS] if c in valid.columns))
    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    valid[pred_cols].to_csv(args.pred_out, index=False)

    meta = {
        "model_kind": "HistGradientBoostingRegressor",
        "target": "ev_edge",
        "rows": int(len(df)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(valid)),
        "states": int(df["_state_key"].nunique()),
        "holdout_states": int(valid["_state_key"].nunique()),
        "holdout_seeds": holdout_seeds,
        "features": len(FEATURE_COLS),
        "label": args.label,
        "base_candidate_depth": infer_base_candidate_depth(df),
        "candidate_depth_schedule": [
            {"round": int(round_no), "depth": int(depth)}
            for round_no, depth in CANDIDATE_DEPTH_SCHEDULE
        ],
        "max_candidate_rank": int(df["candidate_rank_in_state"].max()),
        "model_out": str(args.model_out),
        "features_out": str(args.features_out),
        "audit_out": str(args.audit_out),
        "pred_out": str(args.pred_out),
        "metrics": metrics,
    }
    args.meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_audit(
        args.audit_out,
        meta,
        metrics,
        [*sanity_checks(valid), *choice_sanity_checks(choice["state_rows"])],
    )
    return meta


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_IN)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES_OUT)
    p.add_argument("--meta-out", type=Path, default=DEFAULT_META_OUT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT_OUT)
    p.add_argument("--pred-out", type=Path, default=DEFAULT_PRED_OUT)
    p.add_argument("--label", default="DK EV Policy",
                   help="human-readable label for audit/meta output")
    p.add_argument("--holdout-frac", type=float, default=0.20)
    p.add_argument("--random-state", type=int, default=20260612)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    meta = fit_policy(args)
    metrics = meta["metrics"]
    print(f"Saved model -> {meta['model_out']}")
    print(f"Saved features -> {meta['features_out']}")
    print(f"Saved audit -> {meta['audit_out']}")
    print(
        "Holdout: "
        f"MAE={metrics['edge_mae']:.8f} "
        f"corr={metrics['edge_corr']:.3f} "
        f"top1={metrics['model_top1_accuracy']:.1%} "
        f"adp_top1={metrics['adp_top1_accuracy']:.1%} "
        f"regret={metrics['model_avg_regret']:.8f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
