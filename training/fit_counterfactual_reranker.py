"""Fit a reversible second-stage reranker from counterfactual rollout labels.

The reranker is audit-only by default. It learns from full-draft counterfactual
EV labels produced by audit_counterfactual_picks.py, then reports whether it
would have selected better candidates than the current policy on held-out states.

Run:
    python training/fit_counterfactual_reranker.py --input data/processed/counterfactual_pick_audit.csv
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


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "server" / "models"
PROCESSED = ROOT / "data" / "processed"

DEFAULT_IN = PROCESSED / "counterfactual_pick_audit.csv"
DEFAULT_BASE_FEATURES = MODELS / "dk_ev_policy_feature_cols.json"
DEFAULT_MODEL_OUT = MODELS / "model_dk_ev_counterfactual_reranker.joblib"
DEFAULT_FEATURES_OUT = MODELS / "dk_ev_counterfactual_reranker_feature_cols.json"
DEFAULT_META_OUT = MODELS / "dk_ev_counterfactual_reranker_meta.json"
DEFAULT_AUDIT_OUT = PROCESSED / "counterfactual_reranker_audit.md"
DEFAULT_PRED_OUT = PROCESSED / "counterfactual_reranker_holdout_predictions.csv"


def load_feature_cols(base_features: Path) -> list[str]:
    cols = json.loads(base_features.read_text())
    return [*cols, "model_score", "candidate_rank_in_state"]


def load_frame(path: Path, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {
        "state_key",
        "state_id",
        "overall_pick",
        "round",
        "candidate_name",
        "candidate_pos",
        "rollout_ev",
        "model_score",
        "candidate_rank_in_state",
        "is_policy_pick",
        *feature_cols,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Missing required columns. Re-run audit_counterfactual_picks.py after "
            f"the feature-output patch. Missing: {missing}"
        )

    df = df.dropna(subset=["rollout_ev"]).copy()
    df["_state_avg_rollout_ev"] = df.groupby("state_key")["rollout_ev"].transform("mean")
    df["_state_max_rollout_ev"] = df.groupby("state_key")["rollout_ev"].transform("max")
    df["rollout_ev_edge"] = df["rollout_ev"] - df["_state_avg_rollout_ev"]
    df["rollout_best"] = (df["rollout_ev"] == df["_state_max_rollout_ev"]).astype(int)
    return df


def split_by_state(df: pd.DataFrame, holdout_frac: float, seed: int) -> tuple[pd.Series, pd.Series, list[str]]:
    states = np.array(sorted(df["state_key"].astype(str).unique()))
    if len(states) < 5:
        rng = np.random.default_rng(seed)
        valid_mask = pd.Series(rng.random(len(df)) < holdout_frac, index=df.index)
        return ~valid_mask, valid_mask, []

    n_holdout = max(1, int(round(len(states) * holdout_frac)))
    rng = np.random.default_rng(seed)
    holdout = set(rng.choice(states, size=n_holdout, replace=False))
    valid_mask = df["state_key"].astype(str).isin(holdout)
    return ~valid_mask, valid_mask, sorted(holdout)


def make_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.035,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.05,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=seed,
    )


def choice_metrics(df: pd.DataFrame, pred_col: str) -> dict:
    rows = []
    for state_key, g in df.groupby("state_key", sort=False):
        true_best = g.loc[g["rollout_ev"].idxmax()]
        rerank_pick = g.loc[g[pred_col].idxmax()]
        policy_pick = g[g["is_policy_pick"].eq(1)].iloc[0]
        base_pick = g.loc[g["model_score"].idxmax()]
        adp_pick = g.loc[g["candidate_rank_in_state"].idxmin()]
        rows.append({
            "state_key": state_key,
            "overall_pick": int(g["overall_pick"].iloc[0]),
            "round": int(g["round"].iloc[0]),
            "true_name": true_best["candidate_name"],
            "true_pos": true_best["candidate_pos"],
            "rerank_name": rerank_pick["candidate_name"],
            "rerank_pos": rerank_pick["candidate_pos"],
            "policy_name": policy_pick["candidate_name"],
            "policy_pos": policy_pick["candidate_pos"],
            "base_name": base_pick["candidate_name"],
            "base_pos": base_pick["candidate_pos"],
            "adp_name": adp_pick["candidate_name"],
            "adp_pos": adp_pick["candidate_pos"],
            "true_ev": float(true_best["rollout_ev"]),
            "rerank_ev": float(rerank_pick["rollout_ev"]),
            "policy_ev": float(policy_pick["rollout_ev"]),
            "base_ev": float(base_pick["rollout_ev"]),
            "adp_ev": float(adp_pick["rollout_ev"]),
            "rerank_hit": int(rerank_pick.name == true_best.name),
            "policy_hit": int(policy_pick.name == true_best.name),
            "base_hit": int(base_pick.name == true_best.name),
            "adp_hit": int(adp_pick.name == true_best.name),
            "rerank_regret": float(true_best["rollout_ev"] - rerank_pick["rollout_ev"]),
            "policy_regret": float(true_best["rollout_ev"] - policy_pick["rollout_ev"]),
            "base_regret": float(true_best["rollout_ev"] - base_pick["rollout_ev"]),
            "adp_regret": float(true_best["rollout_ev"] - adp_pick["rollout_ev"]),
            "rerank_lift_vs_policy": float(rerank_pick["rollout_ev"] - policy_pick["rollout_ev"]),
            "rerank_lift_vs_base": float(rerank_pick["rollout_ev"] - base_pick["rollout_ev"]),
        })

    choices = pd.DataFrame(rows)
    summary = {
        "states": int(len(choices)),
        "reranker_top1": float(choices["rerank_hit"].mean()),
        "policy_top1": float(choices["policy_hit"].mean()),
        "base_model_top1": float(choices["base_hit"].mean()),
        "adp_top1": float(choices["adp_hit"].mean()),
        "reranker_avg_regret": float(choices["rerank_regret"].mean()),
        "policy_avg_regret": float(choices["policy_regret"].mean()),
        "base_model_avg_regret": float(choices["base_regret"].mean()),
        "adp_avg_regret": float(choices["adp_regret"].mean()),
        "reranker_avg_lift_vs_policy": float(choices["rerank_lift_vs_policy"].mean()),
        "reranker_h2h_vs_policy": float((choices["rerank_lift_vs_policy"] > 0).mean()),
        "reranker_avg_lift_vs_base": float(choices["rerank_lift_vs_base"].mean()),
        "reranker_h2h_vs_base": float((choices["rerank_lift_vs_base"] > 0).mean()),
    }
    return {"choices": choices, "summary": summary}


def write_report(path: Path, meta: dict, metrics: dict, choices: pd.DataFrame) -> None:
    biggest = choices.sort_values("rerank_lift_vs_policy", ascending=False).head(10)
    worst = choices.sort_values("rerank_lift_vs_policy", ascending=True).head(10)
    lines = [
        "# Counterfactual Reranker Audit",
        "",
        "Second-stage reranker trained from full-draft counterfactual rollout labels.",
        "",
        "## Setup",
        "",
        f"- Input: `{meta['input']}`",
        f"- Rows: {meta['rows']:,}",
        f"- States: {meta['states']:,}",
        f"- Train states: {meta['train_states']:,}",
        f"- Holdout states: {meta['holdout_states']:,}",
        f"- Features: {meta['features']}",
        f"- Target: `{meta['target']}`",
        "- Artifact is separate from the live policy and is not wired into the server.",
        "",
        "## Holdout Metrics",
        "",
        f"- Edge MAE: {metrics['edge_mae']:.3e}",
        f"- Edge RMSE: {metrics['edge_rmse']:.3e}",
        f"- Edge corr: {metrics['edge_corr']:.3f}",
        "",
        "| picker | top-1 | avg regret |",
        "| --- | ---: | ---: |",
        f"| reranker | {metrics['reranker_top1']:.1%} | {metrics['reranker_avg_regret']:.3e} |",
        f"| current policy | {metrics['policy_top1']:.1%} | {metrics['policy_avg_regret']:.3e} |",
        f"| base model score | {metrics['base_model_top1']:.1%} | {metrics['base_model_avg_regret']:.3e} |",
        f"| ADP rank | {metrics['adp_top1']:.1%} | {metrics['adp_avg_regret']:.3e} |",
        "",
        "## Reranker Vs Current Policy",
        "",
        f"- Average lift: {metrics['reranker_avg_lift_vs_policy']:.3e}",
        f"- H2H: {metrics['reranker_h2h_vs_policy']:.1%}",
        "",
        "## Biggest Reranker Wins",
        "",
        "| pick | round | policy | reranker | true best | lift |",
        "| ---: | ---: | --- | --- | --- | ---: |",
    ]
    for _, r in biggest.iterrows():
        lines.append(
            f"| {int(r['overall_pick'])} | {int(r['round'])} | {r['policy_name']} ({r['policy_pos']}) | "
            f"{r['rerank_name']} ({r['rerank_pos']}) | {r['true_name']} ({r['true_pos']}) | "
            f"{float(r['rerank_lift_vs_policy']):.3e} |"
        )

    lines.extend([
        "",
        "## Biggest Reranker Losses",
        "",
        "| pick | round | policy | reranker | true best | lift |",
        "| ---: | ---: | --- | --- | --- | ---: |",
    ])
    for _, r in worst.iterrows():
        lines.append(
            f"| {int(r['overall_pick'])} | {int(r['round'])} | {r['policy_name']} ({r['policy_pos']}) | "
            f"{r['rerank_name']} ({r['rerank_pos']}) | {r['true_name']} ({r['true_pos']}) | "
            f"{float(r['rerank_lift_vs_policy']):.3e} |"
        )

    lines.extend([
        "",
        "## Read",
        "",
        "- This should not be trusted unless holdout lift and H2H beat the current policy on a much larger counterfactual dataset.",
        "- If it passes, the next step is a paired full-draft EV A/B against the current policy, still not live deployment.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit(args) -> dict:
    feature_cols = load_feature_cols(args.base_features)
    df = load_frame(args.input, feature_cols)
    train_mask, valid_mask, holdout_states = split_by_state(df, args.holdout_frac, args.random_state)
    train = df[train_mask].copy()
    valid = df[valid_mask].copy()
    if train.empty or valid.empty:
        raise ValueError("train/holdout split produced an empty side")

    model = make_model(args.random_state)
    model.fit(train[feature_cols].fillna(0), train["rollout_ev_edge"])
    valid["pred_rerank_edge"] = model.predict(valid[feature_cols].fillna(0))

    edge_mae = mean_absolute_error(valid["rollout_ev_edge"], valid["pred_rerank_edge"])
    edge_rmse = mean_squared_error(valid["rollout_ev_edge"], valid["pred_rerank_edge"]) ** 0.5
    edge_corr = float(np.corrcoef(valid["rollout_ev_edge"], valid["pred_rerank_edge"])[0, 1])
    choice = choice_metrics(valid, "pred_rerank_edge")
    metrics = {
        "edge_mae": float(edge_mae),
        "edge_rmse": float(edge_rmse),
        "edge_corr": edge_corr,
        **choice["summary"],
    }

    final_model = make_model(args.random_state)
    final_model.fit(df[feature_cols].fillna(0), df["rollout_ev_edge"])

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.model_out)
    args.features_out.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    choices = choice["choices"]
    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    choices.to_csv(args.pred_out, index=False)

    meta = {
        "model_kind": "HistGradientBoostingRegressor",
        "input": str(args.input),
        "model_out": str(args.model_out),
        "features_out": str(args.features_out),
        "audit_out": str(args.audit_out),
        "pred_out": str(args.pred_out),
        "target": "rollout_ev_edge",
        "rows": int(len(df)),
        "states": int(df["state_key"].nunique()),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(valid)),
        "train_states": int(train["state_key"].nunique()),
        "holdout_states": int(valid["state_key"].nunique()),
        "holdout_state_keys": holdout_states,
        "features": len(feature_cols),
        "metrics": metrics,
    }
    args.meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(args.audit_out, meta, metrics, choices)
    return meta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_IN)
    p.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES_OUT)
    p.add_argument("--meta-out", type=Path, default=DEFAULT_META_OUT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT_OUT)
    p.add_argument("--pred-out", type=Path, default=DEFAULT_PRED_OUT)
    p.add_argument("--holdout-frac", type=float, default=0.25)
    p.add_argument("--random-state", type=int, default=20260618)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    meta = fit(args)
    m = meta["metrics"]
    print(f"Saved model -> {meta['model_out']}")
    print(f"Saved audit -> {meta['audit_out']}")
    print(
        "Holdout: "
        f"rerank_top1={m['reranker_top1']:.1%} "
        f"policy_top1={m['policy_top1']:.1%} "
        f"lift={m['reranker_avg_lift_vs_policy']:.3e} "
        f"h2h={m['reranker_h2h_vs_policy']:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
