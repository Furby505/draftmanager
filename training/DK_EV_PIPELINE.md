# DK EV Pipeline

This is the active DraftKings-specific training path.

Goal:

1. Use real DraftKings Best Ball ADP as the market/opponent reference.
2. Draft realistic DK rooms with ADP plus noise.
3. Simulate correlated weekly player outcomes.
4. Score rosters under DK Best Ball tournament rules.
5. Train/evaluate candidate picks by prize EV.

Canonical command:

```powershell
python training/run_dk_ev_training.py --append
```

Useful smoke command:

```powershell
python training/run_dk_ev_training.py --skip-tests --states 2 --candidates 3 --rollouts 1 --eval-seasons 100 --field-rooms 1 --field-seasons 40
```

Fit/update the served policy:

```powershell
python training/fit_dk_ev_policy.py
```

Candidate depth is round-aware. Production defaults score a base of 12
candidates and expand to 14/18/24/32 in rounds 3/7/11/15. Smoke runs with a
smaller `--candidates` value scale that schedule down. The server caps live
scoring at the trained artifact's `max_candidate_rank`; regenerate rollout
labels and refit before relying on wider pools.

Files in this lane:

- `training/best_ball.py`
- `training/opponent_field.py`
- `training/outcome_model.py`
- `training/season_sim.py`
- `training/bracket.py`
- `training/train_policy.py`
- `training/fit_dk_ev_policy.py`
- `training/run_dk_ev_training.py`
- `training/run_dk_ev_batches.py`
- `training/backtest_policy.py`
- `training/backtest_historical.py`

The old XGBoost/self-play marginal-points lane is retired. Do not reintroduce a
second model fallback in `server/server.py` unless there is a new validated
reason.
