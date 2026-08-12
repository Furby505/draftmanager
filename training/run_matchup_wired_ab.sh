#!/usr/bin/env bash
# Matchup A/B RETEST with the feature now actually wired into FEATURE_COLS
# (commit 85c50ff). Previously cand_playoff_matchup was computed but dropped at
# fit time, so the model had no direct handle on a player's wk15-17 schedule and
# matchup only reached it indirectly via labels -> weak, seed-noisy result.
#
# Clean isolation: BOTH arms handcuff-OFF, ONLY the matchup sim-toggle differs;
# the playoff-matchup FEATURE is present in both arms (informative only when the
# labels carry the signal, i.e. the ON arm). 3 seeds to kill seed-luck.
set -u
cd "$(dirname "$0")/.."
D=data/processed
M=server/models
LOG=$D/matchup_wired_run.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

COMMON="--states 700 --candidates 8 --rollouts 3 --eval-seasons 700 --field-rooms 6 --field-seasons 150 --completion self --no-handcuff"

for S in 20260622 20260623 20260624; do
  OUT=$D/matchup_wired_seed_${S}.md
  if [ -f "$OUT" ]; then log "seed $S already done, skip"; continue; fi
  log "=== seed $S: matchup ON ==="
  python training/train_policy.py $COMMON --seed $S --out $D/mw_on_${S}.csv 2>&1 | tee -a "$LOG"
  python training/fit_dk_ev_policy.py --input $D/mw_on_${S}.csv \
    --model-out $M/_mw_on_${S}.joblib --features-out $M/_mw_on_${S}_cols.json \
    --meta-out $M/_mw_on_${S}_meta.json --audit-out $D/_mw_on_${S}_audit.md --pred-out $D/_mw_on_${S}_pred.csv 2>&1 | tee -a "$LOG"
  log "=== seed $S: matchup OFF (control) ==="
  python training/train_policy.py $COMMON --no-matchup --seed $S --out $D/mw_off_${S}.csv 2>&1 | tee -a "$LOG"
  python training/fit_dk_ev_policy.py --input $D/mw_off_${S}.csv \
    --model-out $M/_mw_off_${S}.joblib --features-out $M/_mw_off_${S}_cols.json \
    --meta-out $M/_mw_off_${S}_meta.json --audit-out $D/_mw_off_${S}_audit.md --pred-out $D/_mw_off_${S}_pred.csv 2>&1 | tee -a "$LOG"
  log "=== seed $S: paired A/B ==="
  python training/paired_models.py --drafts 800 --scoring full --candidates 8 \
    --a $M/_mw_on_${S}.joblib --a-cols $M/_mw_on_${S}_cols.json \
    --b $M/_mw_off_${S}.joblib --b-cols $M/_mw_off_${S}_cols.json \
    --label-a matchup --label-b control --out $OUT 2>&1 | tee -a "$LOG"
done

log "=== SUMMARY ==="
{
  echo "# MATCHUP WIRED A/B  ($(date '+%F %T'))  -- feature in FEATURE_COLS (85c50ff)"
  echo
  for S in 20260622 20260623 20260624; do
    echo "## seed $S"
    if [ -f "$D/matchup_wired_seed_${S}.md" ]; then
      grep -iE "advance|lift|signif" "$D/matchup_wired_seed_${S}.md" | head -3
      printf "fit corr: matchup="; python -c "import json;print('%.3f'%json.load(open('$M/_mw_on_${S}_meta.json'))['metrics']['edge_corr'])" 2>/dev/null
      printf "          control="; python -c "import json;print('%.3f'%json.load(open('$M/_mw_off_${S}_meta.json'))['metrics']['edge_corr'])" 2>/dev/null
    else echo "_(not produced)_"; fi
    echo
  done
} > "$D/MATCHUP_WIRED_SUMMARY.md"
log "DONE -> data/processed/MATCHUP_WIRED_SUMMARY.md"
