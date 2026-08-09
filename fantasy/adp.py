"""Load FantasyPros ADP (average draft position) exports.

The ADP CSVs carry a proper header, but the per-source columns differ between the
Std / Half-PPR / PPR exports (Sleeper, Yahoo, ESPN, ...), so we read by header name and
keep only the stable ones: `Rank`, `Player (Bye)`, `POS`, `AVG`.

`Player (Bye)` packs three fields into one string, in three shapes:
    "Jahmyr Gibbs   DET (6)"        -> name, team, bye
    "Houston Texans DST   (8)"      -> team defense (no team abbreviation)
    "Tyreek Hill"                   -> no team/bye (free agent)

ADP joins to the projections on (norm_name, position). Both sides are FantasyPros
exports sharing a naming convention, so this matches nearly perfectly -- and unlike a
gsis_id join it also covers K and DST, which aren't in the player id tables.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from .names import _norm_expr

SCORING_FILES = {
    "Std": "FantasyPros_2026_Overall_ADP_Rankings_Std.csv",
    "Half_PPR": "FantasyPros_2026_Overall_ADP_Rankings_Half_PPR.csv",
    "PPR": "FantasyPros_2026_Overall_ADP_Rankings_PPR.csv",
}

# "Name  TEAM (bye)" -- 2+ spaces before the team abbreviation
_PLAYER_RE = r"^(?<name>.*?)\s{2,}(?<team>[A-Za-z]{2,3})\s*\((?<bye>\d+)\)\s*$"
# "Full Team Name DST   (bye)"
_DST_RE = r"^(?<name>.*?)\s+DST\s*\((?<bye>\d+)\)\s*$"


def load_adp(
    data_dir: str | Path = "fantasyprosdp",
    scoring: str = "Std",
) -> pl.DataFrame:
    """Return tidy ADP: player, team, bye, position, pos_rank, adp, adp_overall_rank.

    scoring: one of "Std", "Half_PPR", "PPR" (or a bare filename in data_dir).
    """
    data_dir = Path(data_dir)
    filename = SCORING_FILES.get(scoring, scoring)
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No ADP file at {path}. Available: {sorted(p.name for p in data_dir.glob('*.csv'))}"
        )

    raw = pl.read_csv(path, infer_schema_length=0)  # all strings; cast what we need
    missing = {"Rank", "Player (Bye)", "POS", "AVG"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {sorted(missing)}")

    player_raw = pl.col("Player (Bye)").str.strip_chars()
    df = raw.select(
        pl.col("Rank").cast(pl.Int64, strict=False).alias("adp_overall_rank"),
        player_raw.alias("player_raw"),
        pl.col("POS").str.strip_chars().alias("pos_raw"),
        pl.col("AVG").cast(pl.Float64, strict=False).alias("adp"),
    ).filter(pl.col("player_raw").is_not_null() & (pl.col("player_raw").str.len_chars() > 0))

    df = df.with_columns(
        # DST first: "Houston Texans DST (8)" would otherwise mis-parse as team="DST"
        pl.coalesce(
            pl.col("player_raw").str.extract(_DST_RE, 1),
            pl.col("player_raw").str.extract(_PLAYER_RE, 1),
            pl.col("player_raw"),  # bare name, no team/bye
        )
        .str.strip_chars()
        .alias("player"),
        pl.col("player_raw").str.extract(_PLAYER_RE, 2).alias("team"),
        pl.coalesce(
            pl.col("player_raw").str.extract(_DST_RE, 2),
            pl.col("player_raw").str.extract(_PLAYER_RE, 3),
        )
        .cast(pl.Int64, strict=False)
        .alias("bye"),
        # "RB1" -> position "RB", positional rank 1
        pl.col("pos_raw").str.extract(r"^([A-Za-z]+)", 1).alias("position"),
        pl.col("pos_raw").str.extract(r"(\d+)$", 1).cast(pl.Int64, strict=False).alias("pos_rank"),
    )

    df = df.with_columns(_norm_expr("player"))
    return df.select(
        ["player", "team", "bye", "position", "pos_rank", "adp", "adp_overall_rank", "norm_name"]
    )


def attach_adp(proj: pl.DataFrame, adp: pl.DataFrame) -> pl.DataFrame:
    """Left-join ADP onto a projections frame on (norm_name, position).

    Players with no ADP row (undrafted / deep bench) keep a null adp rather than being
    dropped -- they still belong on the PAR board.
    """
    cols = adp.select(["norm_name", "position", "adp", "adp_overall_rank", "bye"])
    # guard against duplicate ADP rows creating fan-out on the join
    cols = cols.unique(subset=["norm_name", "position"], keep="first")
    return proj.join(cols, on=["norm_name", "position"], how="left")
