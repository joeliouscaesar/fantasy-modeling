"""Replacement levels and the final PAR table.

Replacement level is the per-game production of the *last startable* player at a
position given league size and roster settings -- not the positional average. FLEX
slots are distributed across RB/WR/TE by a configurable share. PAR is then:

    PAR = (points_per_game - replacement_points_per_game[pos]) * expected_games_played
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

FLEX_POSITIONS = ("RB", "WR", "TE")


@dataclass
class LeagueConfig:
    n_teams: int = 12
    starters: dict[str, int] = field(
        default_factory=lambda: {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}
    )
    n_flex: int = 1  # flex slots per team, drawn from RB/WR/TE
    flex_shares: dict[str, float] = field(
        default_factory=lambda: {"RB": 0.4, "WR": 0.5, "TE": 0.1}
    )
    # positions we don't model availability for; assume a full season
    default_games: float = 17.0

    def replacement_rank(self) -> dict[str, int]:
        """Number of rostered starters league-wide at each position (the cutoff rank)."""
        ranks: dict[str, int] = {}
        for pos, base in self.starters.items():
            rank = base * self.n_teams
            if pos in FLEX_POSITIONS:
                rank += round(self.flex_shares.get(pos, 0.0) * self.n_flex * self.n_teams)
            ranks[pos] = max(int(round(rank)), 1)
        return ranks


def replacement_levels(proj: pl.DataFrame, league: LeagueConfig) -> dict[str, float]:
    """Per-game points of the replacement-rank player at each position."""
    ranks = league.replacement_rank()
    levels: dict[str, float] = {}
    for pos, rank in ranks.items():
        pool = (
            proj.filter(pl.col("position") == pos)
            .sort("points_per_game", descending=True)
            .get_column("points_per_game")
        )
        if pool.len() == 0:
            levels[pos] = 0.0
        else:
            idx = min(rank, pool.len()) - 1  # clamp if the pool is short
            levels[pos] = float(pool[idx])
    return levels


def build_par_table(
    proj: pl.DataFrame,
    expected_games: pl.DataFrame,
    league: LeagueConfig | None = None,
) -> pl.DataFrame:
    """Assemble the final PAR table.

    proj: from projections.load_projections (needs player, team, position,
          points_per_game, gsis_id).
    expected_games: (gsis_id, expected_games) from the games model.
    """
    league = league or LeagueConfig()
    levels = replacement_levels(proj, league)

    repl_df = pl.DataFrame(
        {"position": list(levels.keys()), "replacement_ppg": list(levels.values())}
    )

    out = (
        proj.join(expected_games, on="gsis_id", how="left")
        .join(repl_df, on="position", how="left")
        .with_columns(
            # K/DST and any unmatched player fall back to a full season
            pl.col("expected_games").fill_null(league.default_games),
            pl.col("replacement_ppg").fill_null(0.0),
        )
        .with_columns(
            (pl.col("points_per_game") - pl.col("replacement_ppg")).alias("par_per_game")
        )
        .with_columns((pl.col("par_per_game") * pl.col("expected_games")).alias("PAR"))
        .rename({"points_per_game": "expected_points_per_game", "expected_games": "expected_games_played"})
        .select(
            [
                "player",
                "team",
                "position",
                "expected_points_per_game",
                "expected_games_played",
                "replacement_ppg",
                "par_per_game",
                "PAR",
            ]
        )
        .sort("PAR", descending=True)
    )
    return out
