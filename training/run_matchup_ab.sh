#!/usr/bin/env bash
# Isolated A/B for the playoff-matchup layer. BOTH arms keep handcuffs ON (already
# settled separately); the ONLY variable is matchup modulation. High-fidelity config
# (more eval-seasons + states) so both models fit competently -- the fast handcuff-A/B
# config gave near-random models (corr ~0.09), which makes any paired backtest noise.
# Arm A runs first so its holdout corr prints early as a fit-quality canary.
set -e
cd "$(dirname "$0")/.."

D=data/processed
M=server/models
COMMON="--states 500 --candidates 8 --rollouts 3 --eval-seasons 700 --field-rooms 6 --field-seasons 150 --completion self --seed 20260618"

echo "=================  ARM 1/2: matchup ON (handcuff ON)  ================="
python training/train_policy.py $COMMON --out $D/mu_on_rollouts.csv
python training/fit_dk_ev_policy.py --input $D/mu_on_rollouts.csv \
  --model-out $M/_mu_on.joblib --features-out $M/_mu_on_cols.json \
  --meta-out $M/_mu_on_meta.json --audit-out $D/_mu_on_audit.md --pred-out $D/_mu_on_pred.csv

echo "=================  ARM 2/2: matchup OFF / control (handcuff ON)  ================="
python training/train_policy.py $COMMON --no-matchup --out $D/mu_off_rollouts.csv
python training/fit_dk_ev_policy.py --input $D/mu_off_rollouts.csv \
  --model-out $M/_mu_off.joblib --features-out $M/_mu_off_cols.json \
  --meta-out $M/_mu_off_meta.json --audit-out $D/_mu_off_audit.md --pred-out $D/_mu_off_pred.csv

echo "=================  PAIRED A/B (matchup vs control)  ================="
python training/paired_models.py --drafts 800 --scoring full --candidates 8 \
  --a $M/_mu_on.joblib --a-cols $M/_mu_on_cols.json \
  --b $M/_mu_off.joblib --b-cols $M/_mu_off_cols.json \
  --label-a matchup --label-b control \
  --out $D/matchup_ab_audit.md

echo "=================  DONE  ================="
