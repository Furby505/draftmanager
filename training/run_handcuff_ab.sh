#!/usr/bin/env bash
# Matched A/B for the handcuff layer: generate two rollout sets (handcuff ON vs OFF)
# with identical config/seed, fit each, then paired-backtest the two models on real
# 2024 results. The only difference between arms is the handcuff workload-inheritance
# (+ the cand_handcuff_depth feature), so the paired advance delta is its real effect.
set -e
cd "$(dirname "$0")/.."

D=data/processed
M=server/models
COMMON="--states 300 --candidates 8 --rollouts 2 --eval-seasons 400 --field-rooms 6 --field-seasons 120 --completion self --seed 20260617"

echo "=================  ARM 1/2: handcuff ON  ================="
python training/train_policy.py $COMMON --out $D/hc_on_rollouts.csv
python training/fit_dk_ev_policy.py --input $D/hc_on_rollouts.csv \
  --model-out $M/_hc_on.joblib --features-out $M/_hc_on_cols.json \
  --meta-out $M/_hc_on_meta.json --audit-out $D/_hc_on_audit.md --pred-out $D/_hc_on_pred.csv

echo "=================  ARM 2/2: handcuff OFF (control)  ================="
python training/train_policy.py $COMMON --no-handcuff --out $D/hc_off_rollouts.csv
python training/fit_dk_ev_policy.py --input $D/hc_off_rollouts.csv \
  --model-out $M/_hc_off.joblib --features-out $M/_hc_off_cols.json \
  --meta-out $M/_hc_off_meta.json --audit-out $D/_hc_off_audit.md --pred-out $D/_hc_off_pred.csv

echo "=================  PAIRED A/B (handcuff vs control)  ================="
python training/paired_models.py --drafts 800 --scoring full --candidates 8 \
  --a $M/_hc_on.joblib --a-cols $M/_hc_on_cols.json \
  --b $M/_hc_off.joblib --b-cols $M/_hc_off_cols.json \
  --label-a handcuff --label-b control \
  --out $D/handcuff_ab_audit.md

echo "=================  DONE  ================="
