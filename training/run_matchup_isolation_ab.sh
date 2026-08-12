#!/usr/bin/env bash
# CLEAN isolation A/B for the playoff-matchup layer with HANDCUFF OFF in BOTH arms.
# Purpose: ship ONLY the playoff feature -- the user is undecided on handcuff, so it
# stays off here. The matchup toggle is the ONLY variable between the two arms.
#
# Fidelity bumped to 900 states (champion-grade) so the fitted model reaches the
# live champion's ~0.5 edge-corr and is genuinely deploy-ready, not a noisy proxy.
#
#   (a) matchup-ON vs control  : pure causal matchup effect (identical depth, hc off)
#   (b) matchup-ON vs LIVE      : deploy gate -- does it actually beat what's served?
set -e
cd "$(dirname "$0")/.."

D=data/processed
M=server/models
COMMON="--states 900 --candidates 8 --rollouts 3 --eval-seasons 700 --field-rooms 6 --field-seasons 150 --completion self --no-handcuff --seed 20260618"

echo "=================  ARM 1/2: matchup ON  (handcuff OFF)  ================="
python training/train_policy.py $COMMON --out $D/miso_on_rollouts.csv
python training/fit_dk_ev_policy.py --input $D/miso_on_rollouts.csv \
  --model-out $M/_miso_on.joblib --features-out $M/_miso_on_cols.json \
  --meta-out $M/_miso_on_meta.json --audit-out $D/_miso_on_audit.md --pred-out $D/_miso_on_pred.csv

echo "=================  ARM 2/2: matchup OFF / control  (handcuff OFF)  ================="
python training/train_policy.py $COMMON --no-matchup --out $D/miso_off_rollouts.csv
python training/fit_dk_ev_policy.py --input $D/miso_off_rollouts.csv \
  --model-out $M/_miso_off.joblib --features-out $M/_miso_off_cols.json \
  --meta-out $M/_miso_off_meta.json --audit-out $D/_miso_off_audit.md --pred-out $D/_miso_off_pred.csv

echo "=================  PAIRED (a): matchup vs control (pure effect)  ================="
python training/paired_models.py --drafts 800 --scoring full --candidates 8 \
  --a $M/_miso_on.joblib --a-cols $M/_miso_on_cols.json \
  --b $M/_miso_off.joblib --b-cols $M/_miso_off_cols.json \
  --label-a matchup --label-b control \
  --out $D/matchup_isolation_ab.md

echo "=================  PAIRED (b): matchup vs LIVE champion (deploy gate)  ================="
python training/paired_models.py --drafts 800 --scoring full --candidates 8 \
  --a $M/_miso_on.joblib --a-cols $M/_miso_on_cols.json \
  --b $M/model_dk_ev_policy.joblib --b-cols $M/dk_ev_policy_feature_cols.json \
  --label-a matchup --label-b live \
  --out $D/matchup_vs_live.md

echo "=================  DONE  ================="
