# DK EV Policy Notes - 2026-06-17

## Fixed-Gap Checkpoint

- Fixed `cand_pos_next_best_adp_gap` semantics in training and server scoring:
  same-position gap now points to the next worse same-position replacement, not
  the best other same-position player.
- Added tests to keep same-position gaps non-negative and to ensure the server
  and training behavior stay aligned.
- Adjusted live DK candidate legality so hard caps and final minimum-roster
  pressure are enforced, while soft construction caps remain model context
  rather than hard live bans.
- Added lineage tooling to tie a served policy model to its feature columns,
  metadata, rollout labels, audits, validation reports, and code hashes.

## Active Retrain

The fixed-gap retrain is intentionally writing side artifacts instead of
overwriting the served policy model. Do not promote it until the finalizer fits
the side model and the promotion checker compares it against the current served
artifact.

Useful commands:

```powershell
python training/finalize_fixedgap_run.py
python training/finalize_fixedgap_run.py --fit-if-complete
python training/check_policy_promotion.py
```

## Tight EV Scores / Indifference Band

Live EV scores can be tighter than the model and rollout noise. After the
fixed-gap model is evaluated, add a server-side decision layer that treats
near-tied candidates as an indifference band instead of blindly trusting the
last decimal.

Within that band, rerank only the tied group with practical draft tiebreakers:

- ADP value and survival odds to the next pick.
- Roster construction need and final legality pressure.
- Correlation and stacking quality, including QB stacks and bring-backs.
- Bye-week and fragility risk.
- Draft uniqueness and chalk avoidance when EV is effectively tied.
- Live safety, especially avoiding large ADP reaches without a clear reason.

The band should be calibrated from holdout or paired-validation regret/noise
after the fixed-gap candidate exists. The server response should eventually
include fields such as `dk_ev_margin_to_top`, `dk_ev_margin_to_next`, and a
short `dk_tiebreak_reason`.
