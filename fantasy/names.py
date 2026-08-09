"""Name normalization + FantasyPros <-> gsis_id bridging.

FantasyPros CSV exports only give a display name, so we normalize names and match
against nflreadpy's `load_ff_playerids()` table (which carries both a name and the
`gsis_id` used everywhere else) to attach player history.
"""
from __future__ import annotations

import re

import nflreadpy as nfl
import polars as pl

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """Lowercase, drop punctuation and generational suffixes, collapse whitespace."""
    if name is None:
        return ""
    s = name.lower()
    s = s.replace(".", " ").replace("'", "").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    tokens = [t for t in s.split() if t and t not in _SUFFIXES]
    return " ".join(tokens)


def _norm_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.replace_all(r"[.'\-]", " ")
        .str.replace_all(r"[^a-z0-9 ]", " ")
        .str.replace_all(r"\b(jr|sr|ii|iii|iv|v)\b", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("norm_name")
    )


def load_id_bridge() -> pl.DataFrame:
    """Return a (norm_name, position, gsis_id) lookup from ff_playerids.

    Deduplicated to one gsis_id per (norm_name, position); ambiguous names that map
    to multiple gsis_ids are dropped so we never silently attach the wrong history.
    """
    ids = nfl.load_ff_playerids().select(["name", "position", "gsis_id"])
    ids = (
        ids.filter(pl.col("gsis_id").is_not_null())
        .with_columns(_norm_expr("name"), pl.col("position").alias("pos"))
        .select(["norm_name", "pos", "gsis_id"])
        .unique()
    )
    # keep only names that resolve unambiguously within a position
    counts = ids.group_by(["norm_name", "pos"]).agg(pl.len().alias("n"))
    unique_keys = counts.filter(pl.col("n") == 1).select(["norm_name", "pos"])
    return ids.join(unique_keys, on=["norm_name", "pos"], how="inner")
