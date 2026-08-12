#!/usr/bin/env bash
# FIELD-REALISM A/B — do the two 2026-06-25 levers (commit 15cd664) produce a
# better real-world drafter? Both default OFF; each is isolated against a SHARED
# matched control (same config, toggles off) so config never confounds the
# toggle. Harness comparisons must keep candidate-rank settings aligned; every
# arm here trains on the same candidate schedule, so evaluating all at
# --candidates 8 is fair.
#
#   A) --field-bias        : opponent rooms reach for hype / let vets fall
#   B) --conditional-field : later bracket rounds scored vs survivors only
#
# Each arm trains matched rollouts + fits a model; paired_models.py then judges
# every arm vs the control on REAL 2024 weekly results against the real Underdog
# field (top-2-of-12 advance + roster-score H2H). The levers only shape TRAINING
# labels, so nothing is plumbed into the evaluator — that is the point (test
# whether the training change transfers to the real field). 3 seeds kill seed-luck.
#
# Caveat (same as the matchup driver): runs at --candidates 8 which still expands
# to 22 via CANDIDATE_DEPTH_SCHEDULE, i.e. the expanding regime, NOT the live
# flat-8 regime. If the flat-8 question (findings #1) lands on flat-8, re-confirm
# any winner there before shipping.
#
# Env knobs:
#   SMOKE=1   tiny config to validate the pipeline end-to-end fast (minutes)
#   BOTH=1    also train+test a both-levers-ON arm vs control
#   SEEDS="…" override the seed list
#   COMMON="…" override the training config
set -u
cd "$(dirname "$0")/.."
D=data/processed
M=server/models
LOG=$D/field_realism_run.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if [ "${SMOKE:-0}" = "1" ]; then
  COMMON="${COMMON:---states 4 --candidates 4 --rollouts 1 --eval-seasons 60 --field-rooms 1 --field-seasons 40 --completion adp --no-handcuff}"
  DRAFTS="${DRAFTS:-20}"
  SEEDS="${SEEDS:-20260625}"
else
  COMMON="${COMMON:---states 700 --candidates 8 --rollouts 3 --eval-seasons 700 --field-rooms 6 --field-seasons 150 --completion self --no-handcuff}"
  DRAFTS="${DRAFTS:-800}"
  SEEDS="${SEEDS:-20260625 20260626 20260627}"
fi

# arm tag -> extra train_policy flags
declare -A ARM_FLAGS=( [ctl]="" [fb]="--field-bias" [cf]="--conditional-field" )
ARMS=(ctl fb cf)
if [ "${BOTH:-0}" = "1" ]; then ARM_FLAGS[both]="--field-bias --conditional-field"; ARMS+=(both); fi

train_arm(){  # $1=seed $2=arm
  local S=$1 arm=$2 mdl=$M/_fr_${2}_${1}.joblib
  if [ -f "$mdl" ]; then log "train $arm seed $S already done, skip"; return; fi
  log "=== seed $S: train arm '$arm' (${ARM_FLAGS[$arm]:-control}) ==="
  python training/train_policy.py $COMMON ${ARM_FLAGS[$arm]} --seed "$S" \
    --out $D/_fr_${arm}_${S}.csv 2>&1 | tee -a "$LOG"
  python training/fit_dk_ev_policy.py --input $D/_fr_${arm}_${S}.csv \
    --model-out "$mdl" --features-out $M/_fr_${arm}_${S}_cols.json \
    --meta-out $M/_fr_${arm}_${S}_meta.json --audit-out $D/_fr_${arm}_${S}_audit.md \
    --pred-out $D/_fr_${arm}_${S}_pred.csv 2>&1 | tee -a "$LOG"
}

pair_vs_ctl(){  # $1=seed $2=arm  -> paired A/B of arm vs control
  local S=$1 arm=$2 out=$D/field_realism_${2}_vs_ctl_seed_${1}.md
  if [ -f "$out" ]; then log "A/B $arm vs ctl seed $S already done, skip"; return; fi
  log "=== seed $S: paired A/B  $arm vs ctl ==="
  python training/paired_models.py --drafts "$DRAFTS" --scoring full --candidates 8 \
    --a $M/_fr_${arm}_${S}.joblib --a-cols $M/_fr_${arm}_${S}_cols.json \
    --b $M/_fr_ctl_${S}.joblib   --b-cols $M/_fr_ctl_${S}_cols.json \
    --label-a "$arm" --label-b control --out "$out" 2>&1 | tee -a "$LOG"
}

for S in $SEEDS; do
  for arm in "${ARMS[@]}"; do train_arm "$S" "$arm"; done
  for arm in "${ARMS[@]}"; do [ "$arm" = ctl ] || pair_vs_ctl "$S" "$arm"; done
done

log "=== SUMMARY ==="
{
  echo "# FIELD-REALISM A/B  ($(date '+%F %T'))  -- levers 15cd664, control shared per seed"
  echo
  for arm in "${ARMS[@]}"; do
    [ "$arm" = ctl ] && continue
    echo "## arm: $arm vs control"
    for S in $SEEDS; do
      f=$D/field_realism_${arm}_vs_ctl_seed_${S}.md
      echo "### seed $S"
      if [ -f "$f" ]; then
        grep -iE "advance|H2H|signif" "$f" | head -3
        printf "fit corr: %s=" "$arm"; python -c "import json;print('%.3f'%json.load(open('$M/_fr_${arm}_${S}_meta.json'))['metrics']['edge_corr'])" 2>/dev/null
        printf "          control=";    python -c "import json;print('%.3f'%json.load(open('$M/_fr_ctl_${S}_meta.json'))['metrics']['edge_corr'])" 2>/dev/null
      else echo "_(not produced)_"; fi
      echo
    done
  done
} > "$D/FIELD_REALISM_SUMMARY.md"
log "DONE -> data/processed/FIELD_REALISM_SUMMARY.md"
