"""Historical games-played and player features for the games model.

`player_stats` has no games column, so games played in a season = number of REG
weekly rows for that player. Season length is 16 (<=2020) or 17 (>=2021); we model
an *availability rate* = games / season_len so the target is comparable across the
16->17 game change, then rescale predictions to a 17-game season.

Caveat: a player who misses an entire season produces no rows and is absent here, so
this trains availability *conditional on being an active/rostered player*. That
slightly biases predictions upward for chronically injured players; good enough for a
draft-time prior, and easy to revisit with roster data later.
"""
from __future__ import annotations

import nflreadpy as nfl
import polars as pl

MODERN_SEASON_LEN = 17
OLD_SEASON_LEN = 16
MODERN_FROM = 2021

FEATURE_COLS = [
    "age",
    "experience",
    "draft_round",
    "prior_rate",
    "prior2_rate",
    "career_avg_rate",
    "prior_games",
    "position",
]


def _season_len_expr() -> pl.Expr:
    return (
        pl.when(pl.col("season") >= MODERN_FROM)
        .then(pl.lit(MODERN_SEASON_LEN))
        .otherwise(pl.lit(OLD_SEASON_LEN))
        .alias("season_len")
    )


def season_games(seasons: list[int]) -> pl.DataFrame:
    """Per (gsis_id, season) games played and availability rate (REG only)."""
    ps = nfl.load_player_stats(seasons).filter(pl.col("season_type") == "REG")
    base = (
        ps.group_by(["player_id", "season"])
        .agg(pl.col("week").n_unique().alias("games"))
        .rename({"player_id": "gsis_id"})
        .with_columns(_season_len_expr())
        .with_columns(
            (pl.col("games") / pl.col("season_len")).clip(0.0, 1.0).alias("availability_rate")
        )
    )
    return base


def player_meta() -> pl.DataFrame:
    """Static per-player attributes: birth year, rookie season, draft round, position."""
    p = nfl.load_players().select(
        ["gsis_id", "birth_date", "rookie_season", "draft_round", "position"]
    )
    return p.with_columns(
        pl.col("birth_date").cast(pl.Date, strict=False).dt.year().alias("birth_year"),
        pl.col("draft_round").cast(pl.Float64, strict=False).alias("draft_round"),
    ).select(["gsis_id", "birth_year", "rookie_season", "draft_round", "position"])


def _lag(base: pl.DataFrame, offset: int, cols: dict[str, str]) -> pl.DataFrame:
    """Self-join base onto `season - offset` to pull prior-season columns.

    Matching on an explicit season number (not row order) so a fully missed season
    yields a null prior rather than reaching further back.
    """
    lagged = base.select(
        pl.col("gsis_id"),
        (pl.col("season") + offset).alias("season"),
        *[pl.col(src).alias(dst) for src, dst in cols.items()],
    )
    return lagged


def _career_avg(base: pl.DataFrame) -> pl.DataFrame:
    """Expanding mean availability over strictly-prior seasons, per player."""
    a = base.select(["gsis_id", "season", "availability_rate"])
    b = a.rename({"season": "season_prior", "availability_rate": "rate_prior"})
    joined = a.join(b, on="gsis_id", how="left").filter(
        pl.col("season_prior") < pl.col("season")
    )
    agg = joined.group_by(["gsis_id", "season"]).agg(
        pl.col("rate_prior").mean().alias("career_avg_rate")
    )
    return agg


def assemble_features(
    base: pl.DataFrame,
    meta: pl.DataFrame,
    season: int,
    anchor_ids: list[str] | pl.Series | None = None,
) -> pl.DataFrame:
    """Feature rows for players relevant to `season`.

    Works for both a completed season (target availability_rate is present) and a
    future target season (target absent).

    anchor_ids=None (training): one row per player who has a prior-season row.
    anchor_ids given (prediction): one row per requested gsis_id -- including rookies
    with no history, whose prior_* features come through null and get imputed, so they
    still receive a modeled prediction from age/experience/draft/position rather than
    a hardcoded default.
    """
    prior1 = _lag(base, 1, {"availability_rate": "prior_rate", "games": "prior_games"})
    prior2 = _lag(base, 2, {"availability_rate": "prior2_rate"})
    career = _career_avg(base)

    if anchor_ids is None:
        anchor = prior1.filter(pl.col("season") == season)
    else:
        anchor = pl.DataFrame({"gsis_id": pl.Series(anchor_ids, dtype=pl.Utf8)}).with_columns(
            pl.lit(season).alias("season")
        )

    anchor = (
        anchor.join(prior1.filter(pl.col("season") == season), on=["gsis_id", "season"], how="left")
        .join(prior2.filter(pl.col("season") == season), on=["gsis_id", "season"], how="left")
        .join(career.filter(pl.col("season") == season), on=["gsis_id", "season"], how="left")
        .join(meta, on="gsis_id", how="left")
    )
    feats = anchor.with_columns(
        (pl.lit(season) - pl.col("birth_year")).alias("age"),
        (pl.lit(season) - pl.col("rookie_season")).clip(0, None).alias("experience"),
    )
    return feats


def build_training_frame(
    first_target: int, last_target: int, history_start: int = 2008
) -> pl.DataFrame:
    """Training rows across target seasons in [first_target, last_target].

    Target = availability_rate that season (joined from base). Features use only
    prior-season information, so there is no leakage.
    """
    seasons = list(range(history_start, last_target + 1))
    base = season_games(seasons)
    meta = player_meta()

    target = base.select(
        ["gsis_id", "season", "availability_rate", "games"]
    ).rename({"availability_rate": "target_rate", "games": "target_games"})

    frames = []
    for s in range(first_target, last_target + 1):
        feats = assemble_features(base, meta, s)
        feats = feats.join(
            target.filter(pl.col("season") == s), on=["gsis_id", "season"], how="inner"
        )
        frames.append(feats)
    return pl.concat(frames)


def build_target_frame(
    target_season: int,
    anchor_ids: list[str] | pl.Series | None = None,
    history_start: int = 2008,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (feature rows for target_season, base) for prediction.

    Pass anchor_ids (e.g. the projected players' gsis_ids) so every projected player
    -- rookies included -- gets a feature row and a modeled prediction.
    """
    seasons = list(range(history_start, target_season))
    base = season_games(seasons)
    meta = player_meta()
    feats = assemble_features(base, meta, target_season, anchor_ids=anchor_ids)
    return feats, base
