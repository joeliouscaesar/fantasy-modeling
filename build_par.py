"""End-to-end: FantasyPros projections + a games-played model -> PAR table.

Run:  uv run python build_par.py

Outputs outputs/par_table.csv and prints a model comparison + the top of the board.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from fantasy import games_model as gm
from fantasy.adp import attach_adp, load_adp
from fantasy.history import build_target_frame, build_training_frame
from fantasy.par import LeagueConfig, build_par_table, replacement_levels
from fantasy.projections import load_projections

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # player names break cp1252 consoles

FIRST_TARGET = 2012  # first season used for training the games model
ADP_SCORING = "Std"  # "Std" | "Half_PPR" | "PPR" -- see fantasy/adp.py SCORING_FILES


def detect_last_completed_season() -> int:
    """Most recent season with regular-season stats available.

    nflreadpy raises (404) for a season whose parquet doesn't exist yet, so we step
    back a year on failure until we hit real data.
    """
    for year in range(date.today().year, 2000, -1):
        try:
            df = nfl.load_player_stats([year])
        except Exception:
            continue
        if df.height > 0 and (df["season_type"] == "REG").any():
            return year
    raise RuntimeError("No player_stats seasons found")


def main() -> None:
    last_completed = detect_last_completed_season()
    target_season = last_completed + 1
    print(f"Last completed season: {last_completed}  ->  projecting {target_season}\n")

    # 1. projections (per-game rate + gsis_id), plus ADP for draft optimization
    proj = load_projections("fantasyprosdata")
    adp = load_adp("fantasyprosdp", scoring=ADP_SCORING)
    proj = attach_adp(proj, adp)
    matched = proj.filter(pl.col("adp").is_not_null()).height
    print(f"ADP ({ADP_SCORING}): {adp.height} rows, matched to {matched}/{proj.height} projected players\n")

    # 2. games model: train, compare, pick best, predict
    train = build_training_frame(FIRST_TARGET, last_completed).to_pandas()
    comparison = gm.evaluate_models(train)
    print("Games-played model comparison (forward-chaining CV):")
    print(comparison.to_string(index=False))
    best = comparison.iloc[0]["model"]
    print(f"\nSelected model: {best}\n")

    # anchor the target frame on the projected players so rookies get modeled too
    proj_ids = proj.filter(pl.col("gsis_id").is_not_null())["gsis_id"].unique().to_list()
    target_feats, _ = build_target_frame(target_season, anchor_ids=proj_ids)
    target_pd = target_feats.to_pandas()
    target_pd["expected_games"] = gm.fit_predict_games(train, target_pd, model_name=best)
    expected_games = pl.from_pandas(target_pd[["gsis_id", "expected_games"]])

    # 3. assemble PAR
    league = LeagueConfig()  # 12-team, 1QB/2RB/2WR/1TE/1FLEX/1K/1DST -- edit as needed
    table = build_par_table(proj, expected_games, league)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "par_table.csv"
    table.write_csv(out_path)

    levels = replacement_levels(proj, league)
    print("Replacement cutoff rank by position:", league.replacement_rank())
    print("Replacement level (pts/game):", {k: round(v, 2) for k, v in levels.items()})
    with pl.Config(tbl_rows=30, tbl_cols=-1, float_precision=1, fmt_str_lengths=22):
        print("\nTop 30 by PAR (full detail in the CSV):")
        print(
            table.head(30).select(
                ["player", "team", "position", "expected_points_per_game",
                 "expected_games_played", "PAR", "par_rank", "adp", "adp_delta"]
            )
        )
        # K/DST excluded: everyone streams them, so their PAR isn't comparable to
        # skill positions and they'd otherwise dominate this view.
        print("\nBiggest values vs ADP (skill positions, PAR rank 1-100, going latest):")
        print(
            table.filter(
                pl.col("adp").is_not_null()
                & (pl.col("par_rank") <= 100)
                & pl.col("position").is_in(["QB", "RB", "WR", "TE"])
            )
            .sort("adp_delta", descending=True)
            .head(10)
            .select(["player", "team", "position", "PAR", "par_rank", "adp", "adp_delta"])
        )
    print(f"\nFull table ({table.height} players) written to {out_path}")


if __name__ == "__main__":
    main()
