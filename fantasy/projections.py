"""Load FantasyPros season projections into tidy per-position, per-game rates.

The exported CSVs have position-specific, duplicated column headers (ATT/YDS/TDS
appear for both passing and rushing, etc.), so we read them header-agnostically and
keep only the columns that are stable across files: the player name (first column),
team (second column), and season fantasy points FPTS (last column). Position comes
from the filename.

`points_per_game` treats the FantasyPros season total as a full-`nominal_games`
projection (a "when healthy" rate). Availability is applied separately by the games
model, so the final PAR = points_per_game_above_replacement * expected_games_played.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from .names import _norm_expr, load_id_bridge

# filename suffix -> position label
_FILE_POS = {
    "QB": "QB",
    "RB": "RB",
    "WR": "WR",
    "TE": "TE",
    "K": "K",
    "DST": "DST",
}

NOMINAL_GAMES = 17


def _load_one(path: Path, position: str) -> pl.DataFrame:
    # Header-agnostic read: headers are duplicated/inconsistent across files. We keep
    # the header row as data (has_header=False, no skip) so the *full-width* header
    # sets the column count -- otherwise the short garbage row below it would make
    # polars truncate every row and we'd read the wrong column as FPTS. The header and
    # garbage rows are dropped afterwards by the numeric FPTS cast.
    raw = pl.read_csv(
        path,
        has_header=False,
        truncate_ragged_lines=True,
        infer_schema_length=0,  # everything as string; we cast what we need
    )
    cols = raw.columns
    player_col, team_col, fpts_col = cols[0], cols[1], cols[-1]
    df = raw.select(
        pl.col(player_col).str.strip_chars().alias("player"),
        pl.col(team_col).str.strip_chars().alias("team"),
        pl.col(fpts_col).cast(pl.Float64, strict=False).alias("proj_fpts"),
    )
    # drop the blank/garbage rows FantasyPros leaves in the export
    df = df.filter(
        pl.col("player").is_not_null()
        & (pl.col("player").str.len_chars() > 0)
        & pl.col("proj_fpts").is_not_null()
    )
    return df.with_columns(pl.lit(position).alias("position"))


def load_projections(
    data_dir: str | Path = "fantasyprosdata",
    nominal_games: int = NOMINAL_GAMES,
) -> pl.DataFrame:
    """Return tidy projections with a per-game rate and a gsis_id where matchable.

    Columns: player, team, position, proj_fpts, points_per_game, norm_name, gsis_id.
    DST rows carry a null gsis_id (team defenses aren't in the player tables).
    """
    data_dir = Path(data_dir)
    frames: list[pl.DataFrame] = []
    for suffix, position in _FILE_POS.items():
        path = data_dir / f"FantasyPros_Fantasy_Football_Projections_{suffix}.csv"
        if path.exists():
            frames.append(_load_one(path, position))
    if not frames:
        raise FileNotFoundError(f"No FantasyPros projection CSVs found in {data_dir}")

    proj = pl.concat(frames)
    proj = proj.with_columns(
        (pl.col("proj_fpts") / nominal_games).alias("points_per_game"),
        _norm_expr("player"),
    )

    # attach gsis_id for non-DST via the id bridge (position-aware match)
    bridge = load_id_bridge().rename({"pos": "position"})
    proj = proj.join(bridge, on=["norm_name", "position"], how="left")
    return proj
