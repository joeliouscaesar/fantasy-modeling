# Fantasy Modeling

Rank NFL players for fantasy drafts by **PAR — points above replacement**:

```
PAR = (points_per_game − replacement_points_per_game[position]) × expected_games_played
```

The two components are modeled separately (a per-game production rate × availability) so
each can be inspected on its own and reused in a later draft-optimization step.

## Quick start

```bash
uv run python build_par.py
```

This prints a games-model comparison and the top of the board, and writes the full table
to `outputs/par_table.csv` with columns:

| column | meaning |
|---|---|
| `expected_points_per_game` | FantasyPros season `FPTS` ÷ 17 (a "when healthy" rate) |
| `expected_games_played` | predicted by the games model (K/DST default to a full season) |
| `replacement_ppg` | per-game points of the last startable player at that position |
| `par_per_game` | `expected_points_per_game − replacement_ppg` |
| `PAR` | `par_per_game × expected_games_played` — the draft ranking metric |
| `par_rank` | overall rank by `PAR` (1 = best) |
| `adp` | FantasyPros average draft position (`AVG` column) |
| `adp_overall_rank`, `bye` | ADP consensus rank and bye week |
| `adp_delta` | `adp − par_rank`; **positive = going later than PAR justifies** (value) |

## Data sources

- **FantasyPros season projections** — CSV exports in `fantasyprosdata/` (one per position:
  QB/RB/WR/TE/K/DST). These provide the season point projections. Re-export annually.
- **FantasyPros ADP** — exports in `fantasyprosdp/` (`Std` / `Half_PPR` / `PPR`). Supplies
  the `AVG` draft position. Switch files via `ADP_SCORING` in `build_par.py`.
- **nflreadpy** — historical `load_player_stats` (games played + fantasy points),
  `load_players` (age, rookie season, draft), and `load_ff_playerids` (name → `gsis_id`
  bridge). Truly free, API-accessible *forward* projections basically don't exist, so
  FantasyPros supplies the points and nflreadpy supplies everything historical.

## Package layout

| Module | Role |
|---|---|
| `fantasy/projections.py` | Load FantasyPros CSVs → per-position `FPTS` and a per-game rate |
| `fantasy/history.py` | Historical games-played + features (age, experience, prior/2yr availability, career avg, draft round, position) |
| `fantasy/names.py` | Normalize FantasyPros names → `gsis_id` via `ff_playerids` (unambiguous matches only) |
| `fantasy/adp.py` | Load FantasyPros ADP exports and join them onto the projections |
| `fantasy/games_model.py` | Pluggable model registry + forward-chaining CV + prediction |
| `fantasy/par.py` | League config → replacement levels (FLEX-aware) → final PAR table |
| `build_par.py` | End-to-end orchestrator |

## Swapping games-played models

`games_model.default_models()` returns a dict of named estimators (two baselines + Ridge /
RandomForest / GradientBoosting), all sharing a `fit(X_df, y)` / `predict(X_df)` interface,
so any sklearn-style model drops in. `evaluate_models()` runs season-by-season
forward-chaining CV and reports **MAE in games**. Latest run:

```
grad_boost      3.37  ← auto-selected (best MAE)
random_forest   3.39
ridge           3.40
baseline_career 3.44
baseline_prior  3.58
```

The models beat the naive baselines, but only by ~0.2 games — availability is genuinely
hard to predict, worth knowing before over-trusting this component.

## Assumptions worth knowing (all are knobs)

1. **Per-game = `FPTS / 17`, availability applied separately.** This intentionally
   *re-discounts* injury-prone players (FantasyPros' season total already bakes in some
   durability risk, and the games model applies more). Change `nominal_games` in
   `projections.load_projections` to anchor to the FantasyPros total instead.
2. **League settings drive replacement level.** Default is **12-team,
   1QB / 2RB / 2WR / 1TE / 1FLEX / 1K / 1DST**. Edit `LeagueConfig` in `build_par.py` per
   league — this materially changes the board.
3. **The games model trains on players with ≥1 game**, so it's availability *conditional on
   being active/rostered* — a mild upward bias for the chronically injured. Rookies are
   modeled from age / draft / position; only K/DST and the handful of unmatched skill
   players fall back to a flat 17.
4. **The projections are Half-PPR** (verified by recomputing `FPTS` from the component
   stats), while `ADP_SCORING` currently defaults to the **`Std`** ADP file. Set it to
   `"Half_PPR"` to make the two consistent.
5. **⚠️ The QB projections export is truncated — only 10 QBs.** Replacement rank for QB is
   12, so the level falls back to the worst available QB, which *understates* replacement
   and *inflates* every QB's PAR. Mahomes, Herbert, Caleb Williams, Kyler Murray and ~28
   others are missing from the board entirely. Re-export the QB projections with all
   players included. `replacement_levels` raises a `UserWarning` whenever a pool is short.

## Roadmap

- **Draft optimization** — turn the PAR board into pick decisions (positional scarcity +
  roster-construction constraints). Not built yet.
- **Better availability signal** — add snap share and a proper "missed full season = 0
  games" signal (the biggest current correctness gap).
- **Bayesian per-game distributions** — replace the point per-game rate with a posterior
  predictive (see `jordy exploration.py` for the Gamma–Exponential prototype).
```
