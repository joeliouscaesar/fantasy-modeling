
from dataclasses import FrozenInstanceError

import polars as pl
import pytest

from fantasy.draft import (
    NUM_TEAMS,
    ROSTER_SIZE,
    WEEKS,
    Player,
    Roster,
    add_picks_to_roster,
    best_draft_sequence,
    get_draft_combos,
    get_draftable_positions,
    get_flex_players,
    get_player,
    get_starting_games,
    get_team_remaining_picks,
    STARTING_POSITIONS,
    get_position_priority,
    make_draft_position_pick,
    modified_par,
    roster_par,
    set_player_draft_status,
    subset_partable_not_on_roster,
)

#############################################################
# Tests for get_starting_games
#############################################################

def make_player(
    par_per_game: float = 10.0,
    expected_games_played: float = 17.0,
    par_per_game_flex: float = 8.0,
    bye: int = 7,
    name: str = "Player",
    position: str = "TE",
) -> Player:
    """Build a Player with sensible defaults so tests only state what they care about."""
    return Player(
        par_per_game,
        expected_games_played,
        par_per_game_flex,
        bye,
        name,
        position,
    )


def test_only_player_at_position_can_start_every_week():
    player = make_player()
    assert get_starting_games(player, [], 1) == player.expected_games_played


def test_backup_only_starts_during_the_starters_bye():
    # One starting spot, already held by a higher-PAR player who plays all but
    # their bye week. The backup should inherit exactly that one week.
    starter = make_player(par_per_game=20.0, expected_games_played=17.0, bye=7, name="Starter")
    backup = make_player(par_per_game=10.0, expected_games_played=17.0, bye=12, name="Backup")

    assert get_starting_games(backup, [starter], 1) == 1

def test_no_available_but_different_bye():
    # from gross standpoint there are no available spots, but all share the same bye so a player 
    # has an available start
    drafted_players = [make_player(par_per_game=11,bye=7), make_player(par_per_game=11,bye=7), make_player(par_per_game=11, bye=7)]
    new_player = make_player(par_per_game=8, bye=12)
    assert get_starting_games(new_player, drafted_players, 2) == 1

def test_better_player_starts_fully():
    drafted_players = [make_player(par_per_game=1, bye=8), make_player(par_per_game=2, bye=12)]
    new_player = make_player(par_per_game=10, expected_games_played=4)
    assert get_starting_games(new_player, drafted_players, 1) == 4

def test_actually_two_available():
    # have 3 available spots from gross standpoint but only one bc share same bye
    drafted_players = [
        make_player(par_per_game=8, expected_games_played=11),
        make_player(par_per_game=4, expected_games_played=11),
        make_player(par_per_game=2, expected_games_played=11)
    ]
    new_player = make_player(par_per_game=1, bye=8)
    assert get_starting_games(new_player, drafted_players, 2) == 2

def test_actually_one_available():
    # have no spots from gross standpoint, but they all share same bye
    drafted_players = [
        make_player(par_per_game=8, expected_games_played=11),
        make_player(par_per_game=4, expected_games_played=11),
        make_player(par_per_game=2, expected_games_played=11),
        make_player(par_per_game=1.5, expected_games_played=11)
    ]
    new_player = make_player(par_per_game=1, bye=8)
    assert get_starting_games(new_player, drafted_players, 2) == 1

def test_skips_worse_players():
    # have no spots from gross standpoint, but they all share same bye
    drafted_players = [
        make_player(par_per_game=8, expected_games_played=11),
        make_player(par_per_game=4, expected_games_played=11),
        make_player(par_per_game=2, expected_games_played=11),
        make_player(par_per_game=1.5, expected_games_played=11)
    ]
    new_player = make_player(par_per_game=3, bye=8, expected_games_played=9)
    assert get_starting_games(new_player, drafted_players, 2) == 9

def test_skips_worse_players2():
    # have no spots from gross standpoint, but they all share same bye
    drafted_players = [
        make_player(par_per_game=8, expected_games_played=10),
        make_player(par_per_game=4, expected_games_played=10),
        make_player(par_per_game=3.5, expected_games_played=10),
        make_player(par_per_game=1.5, expected_games_played=11)
    ]
    new_player = make_player(par_per_game=3, bye=8, expected_games_played=9)
    assert get_starting_games(new_player, drafted_players, 2) == 5

def test_skips_worse_players3():
    # have no spots from gross standpoint, but they all share same bye
    drafted_players = [
        make_player(par_per_game=8, expected_games_played=10),
        make_player(par_per_game=4, expected_games_played=10),
        make_player(par_per_game=3.5, expected_games_played=10, bye=8),
        make_player(par_per_game=1.5, expected_games_played=11)
    ]
    new_player = make_player(par_per_game=3, bye=8, expected_games_played=9)
    assert get_starting_games(new_player, drafted_players, 2) == 6

#############################################################
# Tests for get_flex_players
#############################################################

def make_roster(
    starting_positions={"QB":1,"RB":2,"WR":2,"FLEX":1,"TE":1,"DEF":1,"K":1},
    roster_size=15
) -> Roster:
    """Build a Roster with sensible defaults so tests only state what they care about."""
    return Roster(
        starting_positions,
        roster_size=roster_size
    )

def test_empty_roster():
    roster = make_roster()
    assert get_flex_players(roster) == []

def test_all_rbs():
    # test that only rb3 is a flex player, their par_per_game and expected_games_played are adjusted
    roster = make_roster()
    rb1 = make_player(par_per_game=10, position="RB", bye=1)
    rb2 = make_player(par_per_game=9, position="RB", bye=2)
    rb3 = make_player(par_per_game=8, par_per_game_flex=6, position="RB", bye=3)
    roster.drafted["RB"] = [rb1, rb2, rb3]
    roster.drafted["RB"].sort(reverse=True)
    rb3_flex = make_player(par_per_game=6, par_per_game_flex=6, position="FLEX", bye=3,expected_games_played=15)
    assert get_flex_players(roster) == [rb3_flex]

def test_all_multiple_pos():
    # test that rb3 + wr3 are flex players
    roster = make_roster()
    rb1 = make_player(par_per_game=10, position="RB", bye=1)
    rb2 = make_player(par_per_game=9, position="RB", bye=2)
    rb3 = make_player(par_per_game=8, par_per_game_flex=6, position="RB", bye=3, name="flex1")
    wr1 = make_player(par_per_game=10, position="WR", bye=1)
    wr2 = make_player(par_per_game=9, position="WR", bye=2)
    wr3 = make_player(par_per_game=7, par_per_game_flex=5, position="WR", bye=3, name="flex2")
    roster.drafted["RB"] = [rb1, rb2, rb3]
    roster.drafted["RB"].sort(reverse=True)
    roster.drafted["WR"] = [wr1, wr2, wr3]
    roster.drafted["WR"].sort(reverse=True)
    rb3_flex = make_player(par_per_game=6, par_per_game_flex=6, position="FLEX", bye=3,expected_games_played=15, name="flex1")
    wr3_flex = make_player(par_per_game=5, par_per_game_flex=5, position="FLEX", bye=3,expected_games_played=15, name="flex2")
    assert get_flex_players(roster) == [rb3_flex, wr3_flex]

def test_all_multiple_pos2():
    # same as above but change output order
    roster = make_roster()
    rb1 = make_player(par_per_game=10, position="RB", bye=1)
    rb2 = make_player(par_per_game=9, position="RB", bye=2)
    rb3 = make_player(par_per_game=7, par_per_game_flex=5, position="RB", bye=3, name="flex1")
    wr1 = make_player(par_per_game=10, position="WR", bye=1)
    wr2 = make_player(par_per_game=9, position="WR", bye=2)
    wr3 = make_player(par_per_game=8, par_per_game_flex=6, position="WR", bye=3, name="flex2")
    roster.drafted["RB"] = [rb1, rb2, rb3]
    roster.drafted["RB"].sort(reverse=True)
    roster.drafted["WR"] = [wr1, wr2, wr3]
    roster.drafted["WR"].sort(reverse=True)
    rb3_flex = make_player(par_per_game=5, par_per_game_flex=5, position="FLEX", bye=3,expected_games_played=15, name="flex1")
    wr3_flex = make_player(par_per_game=6, par_per_game_flex=6, position="FLEX", bye=3,expected_games_played=15, name="flex2")
    assert get_flex_players(roster) == [wr3_flex, rb3_flex]

def test_other_pos():
    # tests qb, def, k not added
    roster = make_roster()
    for pos in ["QB","DEF","K"]:
        roster.drafted[pos] = [make_player(position=pos), make_player(position=pos), make_player(position=pos)]
    assert get_flex_players(roster) == []


#############################################################
# Fixtures for the partable-driven functions
#############################################################

def make_raw_row(
    player: str,
    position: str,
    expected_points_per_game: float,
    replacement_ppg: float = 10.0,
    expected_games_played: float = 16.0,
    bye: int = 7,
    adp_overall_rank: int | None = 100,
) -> dict:
    """One row shaped like outputs/par_table.csv, before the draft-time prep."""
    par_per_game = expected_points_per_game - replacement_ppg
    return {
        "player": player,
        "team": "XXX",
        "position": position,
        "expected_points_per_game": float(expected_points_per_game),
        "expected_games_played": float(expected_games_played),
        "replacement_ppg": float(replacement_ppg),
        "par_per_game": float(par_per_game),
        "PAR": float(par_per_game * expected_games_played),
        "par_rank": 0,
        "adp": None if adp_overall_rank is None else float(adp_overall_rank),
        "adp_overall_rank": adp_overall_rank,
        "adp_delta": 0.0,
        "bye": bye,
    }


def prepare_partable(raw: pl.DataFrame) -> pl.DataFrame:
    """Mirror of the partable prep sketched at the bottom of fantasy/draft.py.

    Adds flex_par_per_game, renames DST -> DEF, and derives drafted_after from ADP.
    """
    is_flex = pl.col("position").is_in(["RB", "WR", "TE"])
    flex_replacement_value = raw.filter(is_flex).select("replacement_ppg").max()
    return raw.with_columns(
        pl.when(is_flex)
        .then(pl.col("expected_points_per_game") - flex_replacement_value["replacement_ppg"])
        .otherwise(pl.lit(None))
        .alias("flex_par_per_game"),
        pl.when(pl.col("position") == "DST")
        .then(pl.lit("DEF"))
        .otherwise(pl.col("position"))
        .alias("position"),
        (pl.col("adp_overall_rank").fill_null(NUM_TEAMS * ROSTER_SIZE + 1) - 1).alias("drafted_after"),
    )


def make_partable(rows: list[dict]) -> pl.DataFrame:
    return prepare_partable(pl.DataFrame(rows))


def simple_partable() -> pl.DataFrame:
    """4 RBs and 4 WRs, strictly decreasing PAR, all available at every pick.

    RB1 > WR1 > RB2 > WR2 > RB3 > WR3 > RB4 > WR4 by PAR.
    """
    rows = []
    for i in range(4):
        rows.append(make_raw_row(f"RB{i + 1}", "RB", 20 - i, bye=i + 1))
        rows.append(make_raw_row(f"WR{i + 1}", "WR", 19 - i, bye=i + 1))
    return make_partable(rows)


def roster_with(*players: Player) -> Roster:
    """Roster holding the given players, keyed by their position."""
    roster = make_roster()
    for player in players:
        roster.drafted[player.position].append(player)
    return roster


# A player who is not in any test partable. Needed because
# subset_partable_not_on_roster raises on a genuinely empty roster (see the xfail
# test below), and most tests here care about other behaviour.
SENTINEL = Player(0.0, 16.0, 0.0, 1, "NotInTable", "QB")


def drafted_frame(partable: pl.DataFrame) -> pl.DataFrame:
    """partable with the `drafted` flag get_player/set_player_draft_status expect."""
    return partable.with_columns(pl.lit(False).alias("drafted"))


#############################################################
# Tests for subset_partable_not_on_roster
#############################################################

def test_subset_removes_rostered_players():
    partable = simple_partable()
    roster = roster_with(Player(10.0, 16.0, 5.0, 1, "RB1", "RB"))

    remaining = subset_partable_not_on_roster(partable, roster)

    assert "RB1" not in remaining["player"].to_list()
    assert remaining.height == partable.height - 1


def test_subset_keeps_partable_columns_and_drops_join_flag():
    partable = simple_partable()
    roster = roster_with(Player(10.0, 16.0, 5.0, 1, "RB1", "RB"))

    remaining = subset_partable_not_on_roster(partable, roster)

    assert remaining.columns == partable.columns
    assert "drafted" not in remaining.columns


def test_subset_matches_on_player_and_position_together():
    # Same name at two positions: only the rostered (name, position) pair goes.
    partable = make_partable([
        make_raw_row("Ambiguous", "RB", 20),
        make_raw_row("Ambiguous", "WR", 18),
    ])
    roster = roster_with(Player(10.0, 16.0, 5.0, 1, "Ambiguous", "RB"))

    remaining = subset_partable_not_on_roster(partable, roster)

    assert remaining.height == 1
    assert remaining["position"].to_list() == ["WR"]


def test_subset_does_not_mutate_the_input():
    partable = simple_partable()
    roster = roster_with(Player(10.0, 16.0, 5.0, 1, "RB1", "RB"))
    before = partable.height

    subset_partable_not_on_roster(partable, roster)

    assert partable.height == before


def test_subset_with_empty_roster_returns_everything():
    partable = simple_partable()
    remaining = subset_partable_not_on_roster(partable, make_roster())
    assert remaining.height == partable.height


#############################################################
# Tests for get_draftable_positions
#############################################################

ALL_POSITIONS = ["DEF", "K", "QB", "RB", "TE", "WR"]


def test_draftable_positions_defaults_to_every_position():
    assert get_draftable_positions([0, 1, 2], None) == [ALL_POSITIONS] * 3


def test_draftable_positions_returns_one_entry_per_pick():
    assert len(get_draftable_positions([4, 9, 14, 19], {"K": 1})) == 4


def test_draftable_positions_excludes_ignored_positions_then_restores_them():
    # The value is a count of picks: {"K": 2} suppresses K at the first two picks
    # (indices 0 and 1) and allows it from the third pick on.
    draftable = get_draftable_positions([0, 1, 2, 3], {"K": 2})

    assert draftable[0] == ["DEF", "QB", "RB", "TE", "WR"]
    assert draftable[1] == ["DEF", "QB", "RB", "TE", "WR"]
    assert draftable[2] == ALL_POSITIONS
    assert draftable[3] == ALL_POSITIONS


def test_draftable_positions_can_ignore_several_positions():
    draftable = get_draftable_positions([0, 1], {"K": 5, "DEF": 5})
    assert draftable[0] == ["QB", "RB", "TE", "WR"]


def test_draftable_positions_zero_suppresses_nothing():
    draftable = get_draftable_positions([0, 1], {"K": 0})
    assert draftable == [ALL_POSITIONS, ALL_POSITIONS]


def test_draftable_positions_one_suppresses_only_the_first_pick():
    draftable = get_draftable_positions([0, 1], {"K": 1})
    assert "K" not in draftable[0]
    assert "K" in draftable[1]


#############################################################
# Tests for get_player
#############################################################

def test_get_player_returns_highest_par_at_position():
    frame = drafted_frame(simple_partable())
    assert get_player(0, "RB", 0, frame).name == "RB1"
    assert get_player(0, "WR", 0, frame).name == "WR1"


def test_get_player_pos_num_walks_down_the_position_ranking():
    frame = drafted_frame(simple_partable())
    assert [get_player(0, "RB", i, frame).name for i in range(4)] == ["RB1", "RB2", "RB3", "RB4"]


def test_get_player_skips_players_gone_before_this_pick():
    # RB1 is expected to be drafted by pick 2, so he is unavailable at pick 3.
    frame = drafted_frame(simple_partable()).with_columns(
        pl.when(pl.col("player") == "RB1").then(2).otherwise(pl.col("drafted_after")).alias("drafted_after")
    )
    assert get_player(0, "RB", 0, frame).name == "RB1"
    assert get_player(3, "RB", 0, frame).name == "RB2"


def test_get_player_skips_already_drafted_players():
    frame = drafted_frame(simple_partable()).with_columns(
        pl.when(pl.col("player") == "RB1").then(True).otherwise(pl.col("drafted")).alias("drafted")
    )
    assert get_player(0, "RB", 0, frame).name == "RB2"


def test_get_player_maps_partable_columns_onto_player_fields():
    partable = make_partable([
        make_raw_row("Target", "RB", 18.0, replacement_ppg=10.0, expected_games_played=15.0, bye=9)
    ])
    player = get_player(0, "RB", 0, drafted_frame(partable))
    row = partable.row(0, named=True)

    assert player.name == "Target"
    assert player.position == "RB"
    assert player.par_per_game == row["par_per_game"]
    assert player.expected_games_played == row["expected_games_played"]
    assert player.par_per_game_flex == row["flex_par_per_game"]
    assert player.bye == row["bye"]


def test_get_player_raises_when_no_player_is_available():
    frame = drafted_frame(simple_partable())
    with pytest.raises(Exception):
        get_player(0, "RB", 99, frame)


#############################################################
# Tests for set_player_draft_status
#############################################################

def test_set_player_draft_status_marks_only_the_target_player():
    frame = drafted_frame(simple_partable())
    target = get_player(0, "RB", 0, frame)

    updated = set_player_draft_status(target, True, frame)

    drafted = updated.filter(pl.col("drafted"))["player"].to_list()
    assert drafted == ["RB1"]
    assert updated.height == frame.height


def test_set_player_draft_status_round_trips():
    frame = drafted_frame(simple_partable())
    target = get_player(0, "RB", 0, frame)

    marked = set_player_draft_status(target, True, frame)
    released = set_player_draft_status(target, False, marked)

    assert released.filter(pl.col("drafted")).height == 0


def test_set_player_draft_status_distinguishes_same_name_at_two_positions():
    partable = make_partable([
        make_raw_row("Ambiguous", "RB", 20),
        make_raw_row("Ambiguous", "WR", 18),
    ])
    frame = drafted_frame(partable)
    target = get_player(0, "RB", 0, frame)

    updated = set_player_draft_status(target, True, frame)

    assert updated.filter(pl.col("drafted"))["position"].to_list() == ["RB"]


def test_set_player_draft_status_does_not_mutate_the_input():
    frame = drafted_frame(simple_partable())
    target = get_player(0, "RB", 0, frame)

    set_player_draft_status(target, True, frame)

    assert frame.filter(pl.col("drafted")).height == 0


#############################################################
# Tests for get_draft_combos
#############################################################

# Leaves RB and WR as the only draftable positions.
ONLY_RB_WR = {"K": 99, "DEF": 99, "TE": 99, "QB": 99}


def collect_combos(generator, limit: int = 200) -> list[tuple[str, ...]]:
    """Drain get_draft_combos into plain tuples of player names."""
    combos: list[tuple[str, ...]] = []
    for i, combo in enumerate(generator):
        combos.append(tuple(player.name for player in combo))
        if i >= limit:
            raise AssertionError(f"generator produced more than {limit} combos")
    return combos


def test_draft_combos_enumerates_top_player_per_position_pair():
    combos = collect_combos(
        get_draft_combos([0, 1], roster_with(SENTINEL), simple_partable(),
                         consider_per_position=1, position_ignore=ONLY_RB_WR)
    )
    assert combos == [("RB1", "RB2"), ("RB1", "WR1"), ("WR1", "RB1"), ("WR1", "WR2")]


def test_draft_combos_never_repeats_a_player_within_a_combo():
    combos = collect_combos(
        get_draft_combos([0, 1, 2], roster_with(SENTINEL), simple_partable(),
                         consider_per_position=2, position_ignore=ONLY_RB_WR)
    )
    assert combos, "expected at least one combo"
    for combo in combos:
        assert len(set(combo)) == len(combo)


def test_draft_combos_considers_more_players_as_consider_per_position_rises():
    # consider_per_position=k walks ranks 0..k-1, i.e. exactly k players per position.
    # With 2 positions and 2 picks that is (2k)^2 -> 4 combos at k=1, 16 at k=2.
    narrow = collect_combos(
        get_draft_combos([0, 1], roster_with(SENTINEL), simple_partable(),
                         consider_per_position=1, position_ignore=ONLY_RB_WR)
    )
    wide = collect_combos(
        get_draft_combos([0, 1], roster_with(SENTINEL), simple_partable(),
                         consider_per_position=2, position_ignore=ONLY_RB_WR)
    )
    assert len(narrow) == 4
    assert len(wide) == 16


def test_draft_combos_only_drafts_allowed_positions():
    partable = make_partable([
        make_raw_row("RB1", "RB", 20), make_raw_row("RB2", "RB", 19),
        make_raw_row("K1", "K", 12), make_raw_row("DEF1", "DST", 11),
    ])
    combos = collect_combos(
        get_draft_combos([0, 1], roster_with(SENTINEL), partable,
                         consider_per_position=1, position_ignore={"K": 99, "DEF": 99, "TE": 99, "QB": 99, "WR": 99})
    )
    drafted_names = {name for combo in combos for name in combo}
    assert drafted_names <= {"RB1", "RB2"}


def test_draft_combos_excludes_players_already_on_the_roster():
    roster = roster_with(SENTINEL, Player(10.0, 16.0, 5.0, 1, "RB1", "RB"))
    combos = collect_combos(
        get_draft_combos([0, 1], roster, simple_partable(),
                         consider_per_position=1, position_ignore=ONLY_RB_WR)
    )
    drafted_names = {name for combo in combos for name in combo}
    assert "RB1" not in drafted_names


def test_draft_combos_respects_drafted_after():
    # RB1 and WR1 are both gone before pick 5, so combos start from RB2 / WR2.
    partable = simple_partable().with_columns(
        pl.when(pl.col("player").is_in(["RB1", "WR1"])).then(2)
        .otherwise(pl.col("drafted_after")).alias("drafted_after")
    )
    combos = collect_combos(
        get_draft_combos([5, 6], roster_with(SENTINEL), partable,
                         consider_per_position=1, position_ignore=ONLY_RB_WR)
    )
    drafted_names = {name for combo in combos for name in combo}
    assert not ({"RB1", "WR1"} & drafted_names)


def test_draft_combos_yields_one_player_per_remaining_pick():
    combos = collect_combos(
        get_draft_combos([0, 1, 2], roster_with(SENTINEL), simple_partable(),
                         consider_per_position=2, position_ignore=ONLY_RB_WR)
    )
    assert combos, "expected at least one combo"
    assert all(len(combo) == 3 for combo in combos)


def test_draft_combos_terminates_cleanly():
    list(
        get_draft_combos([0, 1], roster_with(SENTINEL), simple_partable(),
                         consider_per_position=1, position_ignore=ONLY_RB_WR)
    )


def test_draft_combos_yields_independent_combos():
    # Holding on to a combo (to track the best roster) must not see it mutate.
    generator = get_draft_combos([0, 1], roster_with(SENTINEL), simple_partable(),
                                 consider_per_position=1, position_ignore=ONLY_RB_WR)
    first = next(generator)
    snapshot = list(first)
    next(generator)

    assert list(first) == snapshot


def test_draft_combos_keeps_every_combo_distinct_when_collected():
    # The whole point of yielding copies: collecting them keeps real alternatives.
    combos = list(
        get_draft_combos([0, 1], roster_with(SENTINEL), simple_partable(),
                         consider_per_position=1, position_ignore=ONLY_RB_WR)
    )
    as_names = [tuple(p.name for p in combo) for combo in combos]
    assert len(set(as_names)) == len(as_names)


def test_draft_combos_can_track_the_best_combo_by_par():
    # A caller should be able to keep the running best without copying defensively.
    combos = get_draft_combos([0, 1], roster_with(SENTINEL), simple_partable(),
                              consider_per_position=1, position_ignore=ONLY_RB_WR)
    best = max(combos, key=lambda combo: sum(p.par_per_game * p.expected_games_played for p in combo))
    assert [p.name for p in best] == ["RB1", "RB2"]


def test_draft_combos_falls_back_when_a_position_is_exhausted():
    # Only one RB exists, so the second pick has to fall through to WR.
    partable = make_partable([make_raw_row("RB1", "RB", 20), make_raw_row("WR1", "WR", 19)])
    combos = collect_combos(
        get_draft_combos([0, 1], roster_with(SENTINEL), partable,
                         consider_per_position=1, position_ignore=ONLY_RB_WR)
    )
    assert ("RB1", "WR1") in combos


def test_draft_combos_rejects_consider_per_position_below_one():
    with pytest.raises(ValueError):
        next(get_draft_combos([0, 1], roster_with(SENTINEL), simple_partable(),
                              consider_per_position=0))


def test_draft_combos_with_no_remaining_picks_yields_nothing():
    assert list(get_draft_combos([], roster_with(SENTINEL), simple_partable())) == []


#############################################################
# Tests for Roster.copy and Player immutability
#############################################################

def test_roster_copy_does_not_share_position_lists():
    # The reason copy() exists: branching the draft search must not leak picks back
    # into the roster we branched from.
    roster = make_roster()
    roster.drafted["RB"].append(make_player(position="RB", name="Bijan"))

    branch = roster.copy()
    branch.drafted["RB"].append(make_player(position="RB", name="Gibbs"))

    assert [p.name for p in roster.drafted["RB"]] == ["Bijan"]
    assert [p.name for p in branch.drafted["RB"]] == ["Bijan", "Gibbs"]


def test_roster_copy_covers_every_position():
    # A missed position would share its list and silently corrupt the original.
    roster = make_roster()
    branch = roster.copy()
    for position in roster.drafted:
        assert branch.drafted[position] is not roster.drafted[position]


def test_roster_copy_shares_player_objects():
    # Deliberate: Players are frozen, so copying them would be pure overhead.
    roster = make_roster()
    roster.drafted["RB"].append(make_player(position="RB", name="Bijan"))

    branch = roster.copy()

    assert branch.drafted["RB"][0] is roster.drafted["RB"][0]


def test_roster_copy_carries_over_config():
    roster = make_roster()
    branch = roster.copy()
    assert branch.starting_positions == roster.starting_positions
    assert branch.roster_size == roster.roster_size


def test_player_cannot_be_mutated():
    # What makes sharing Players between roster copies safe.
    player = make_player()
    with pytest.raises(FrozenInstanceError):
        player.par_per_game = 99.0


def test_player_is_hashable():
    # Falls out of frozen=True, and lets callers dedupe combos with a set.
    player = make_player(name="Bijan", position="RB")
    same = make_player(name="Bijan", position="RB")
    assert len({player, same}) == 1

#############################################################
# Tests for best_draft_sequence
#############################################################

# Two-team snake over two rounds: team 0 picks at 0 and 3, team 1 at 1 and 2.
SNAKE_ORDER = [0, 1, 1, 0]


def rb_wr_partable() -> pl.DataFrame:
    """Four players where the best WR out-scores the best RB.

    PAR: WR1 176 > RB1 160 > WR2 144 > RB2 128, so a sequence that just took the
    alphabetically-first position would pick the wrong player.
    """
    return make_partable([
        make_raw_row("RB1", "RB", 20, bye=1), make_raw_row("RB2", "RB", 18, bye=2),
        make_raw_row("WR1", "WR", 21, bye=3), make_raw_row("WR2", "WR", 19, bye=4),
    ])


def score(combo: list[Player], roster: Roster, bench_penalty: float = 0.5) -> float:
    """PAR of `roster` once `combo` is added -- the quantity being maximised."""
    return roster_par(add_picks_to_roster(combo, roster), bench_penalty=bench_penalty)


def test_best_draft_sequence_takes_the_highest_par_player_available():
    # One pick, so the best sequence is simply the best player on the board.
    best = best_draft_sequence(0, SNAKE_ORDER, make_roster(), rb_wr_partable(),
                               through_round=1, position_ignore=ONLY_RB_WR)
    assert [player.name for player in best] == ["WR1"]


def test_best_draft_sequence_maximises_roster_par():
    roster, partable = make_roster(), rb_wr_partable()
    picks = get_team_remaining_picks(0, SNAKE_ORDER)[:2]
    every_combo = list(get_draft_combos(picks, roster, partable, 1, ONLY_RB_WR))

    best = best_draft_sequence(0, SNAKE_ORDER, roster, partable,
                               through_round=2, position_ignore=ONLY_RB_WR)

    assert score(best, roster) == max(score(combo, roster) for combo in every_combo)


def test_best_draft_sequence_returns_one_player_per_planned_pick():
    best = best_draft_sequence(0, SNAKE_ORDER, make_roster(), rb_wr_partable(),
                               through_round=2, position_ignore=ONLY_RB_WR)
    assert len(best) == 2


def test_best_draft_sequence_plans_fewer_picks_as_through_round_shrinks():
    partable = rb_wr_partable()
    one = best_draft_sequence(0, SNAKE_ORDER, make_roster(), partable,
                              through_round=1, position_ignore=ONLY_RB_WR)
    two = best_draft_sequence(0, SNAKE_ORDER, make_roster(), partable,
                              through_round=2, position_ignore=ONLY_RB_WR)
    assert (len(one), len(two)) == (1, 2)


def test_best_draft_sequence_counts_rostered_players_towards_through_round():
    # One player already drafted, so through_round=2 leaves room for one more pick.
    roster = roster_with(SENTINEL)
    best = best_draft_sequence(0, SNAKE_ORDER, roster, rb_wr_partable(),
                               through_round=2, position_ignore=ONLY_RB_WR)
    assert len(best) == 1


def test_best_draft_sequence_does_not_mutate_the_roster():
    roster = make_roster()
    before = {position: list(players) for (position, players) in roster.drafted.items()}

    best_draft_sequence(0, SNAKE_ORDER, roster, rb_wr_partable(),
                        through_round=2, position_ignore=ONLY_RB_WR)

    assert roster.drafted == before


def test_best_draft_sequence_does_not_redraft_a_rostered_player():
    rb1 = Player(10.0, 16.0, 10.0, 1, "RB1", "RB")
    roster = roster_with(SENTINEL, rb1)

    best = best_draft_sequence(0, SNAKE_ORDER, roster, rb_wr_partable(),
                               through_round=4, position_ignore=ONLY_RB_WR)

    assert "RB1" not in [player.name for player in best]


def test_best_draft_sequence_respects_position_ignore():
    best = best_draft_sequence(0, SNAKE_ORDER, make_roster(), rb_wr_partable(),
                               through_round=2,
                               position_ignore={"WR": 99, "QB": 99, "TE": 99, "K": 99, "DEF": 99})
    assert {player.position for player in best} == {"RB"}


def test_best_draft_sequence_skips_players_gone_before_the_pick():
    # WR1 is the best player, but is projected off the board before pick 0.
    partable = make_partable([
        make_raw_row("RB1", "RB", 20, bye=1),
        make_raw_row("WR1", "WR", 21, bye=3, adp_overall_rank=0),
    ])
    best = best_draft_sequence(0, SNAKE_ORDER, make_roster(), partable,
                               through_round=1, position_ignore=ONLY_RB_WR)
    assert [player.name for player in best] == ["RB1"]


def test_best_draft_sequence_widening_the_search_never_scores_worse():
    # consider_per_position=2 enumerates a superset of the k=1 combos.
    roster, partable = make_roster(), simple_partable()
    narrow = best_draft_sequence(0, SNAKE_ORDER, roster, partable,
                                 consider_per_position=1, through_round=2,
                                 position_ignore=ONLY_RB_WR)
    wide = best_draft_sequence(0, SNAKE_ORDER, roster, partable,
                               consider_per_position=2, through_round=2,
                               position_ignore=ONLY_RB_WR)
    assert score(wide, roster) >= score(narrow, roster)


def test_best_draft_sequence_bench_penalty_changes_the_pick():
    """A bench-bound star beats a modest starter only when benches count fully.

    RB3 has the higher PAR but no route into the lineup: RB and FLEX are already
    taken. TE1 scores less but starts every week. At bench_penalty=0.5 the starter
    wins; at 1.0 the bench points are worth enough to flip it.
    """
    starting_positions = {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1}
    partable = make_partable([
        make_raw_row("RB3", "RB", 25, bye=3),  # PAR 240, but bench-bound
        make_raw_row("TE1", "TE", 20, bye=5),  # PAR 160, starts outright
    ])
    ignore = {"QB": 99, "WR": 99, "K": 99, "DEF": 99}

    def best_at(bench_penalty: float) -> list[str]:
        roster = Roster(starting_positions, 15)
        roster.drafted["RB"] = [Player(20.0, 16.0, 20.0, 1, "RBstar", "RB"),
                                Player(16.0, 16.0, 16.0, 2, "RBflex", "RB")]
        roster.drafted["WR"] = [Player(18.0, 16.0, 18.0, 4, "WRstar", "WR")]
        best = best_draft_sequence(0, SNAKE_ORDER, roster, partable, through_round=4,
                                   bench_penalty=bench_penalty, position_ignore=ignore)
        return [player.name for player in best]

    assert best_at(0.5) == ["TE1"]
    assert best_at(1.0) == ["RB3"]


def test_best_draft_sequence_with_no_rounds_left_returns_no_picks():
    roster = roster_with(SENTINEL, Player(10.0, 16.0, 10.0, 1, "Other", "RB"))
    assert best_draft_sequence(0, SNAKE_ORDER, roster, rb_wr_partable(),
                               through_round=2, position_ignore=ONLY_RB_WR) == []


def test_best_draft_sequence_past_through_round_returns_no_picks():
    roster = roster_with(SENTINEL,
                         Player(10.0, 16.0, 10.0, 1, "Other", "RB"),
                         Player(9.0, 16.0, 9.0, 2, "Another", "RB"))
    assert best_draft_sequence(0, SNAKE_ORDER, roster, rb_wr_partable(),
                               through_round=2, position_ignore=ONLY_RB_WR) == []


#############################################################
# Tests for get_position_priority
#############################################################

# get_position_priority reads the module-level STARTING_POSITIONS rather than the
# roster's own, so these rosters are built with that config to stay meaningful.
# STARTING_POSITIONS is {"QB":1,"WR":3,"RB":2,"TE":1,"FLEX":1,"K":1,"DEF":1}.

def priority_roster(**counts: int) -> Roster:
    """Roster holding `counts` filler players per position, e.g. priority_roster(RB=3)."""
    roster = Roster(STARTING_POSITIONS, 15)
    for position, n in counts.items():
        roster.drafted[position] = [
            Player(10.0 - i, 16.0, 10.0 - i, i + 1, f"{position}{i}", position)
            for i in range(n)
        ]
    return roster


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        # drafted 0, 1, 2, ... at that position, nothing drafted elsewhere
        ("QB", [0, 1, 1, 2]),
        ("TE", [0, 0, 1, 2]),
        ("RB", [0, 0, 0, 1, 2, 2]),
        ("WR", [0, 0, 0, 0, 1, 2]),
        ("K", [0, 2, 2]),
        ("DEF", [0, 2, 2]),
    ],
)
def test_position_priority_progression_as_a_position_fills_up(position, expected):
    actual = [
        get_position_priority(position, priority_roster(**{position: n}))
        for n in range(len(expected))
    ]
    assert actual == expected


def test_position_priority_is_zero_while_a_starting_slot_is_open():
    assert get_position_priority("WR", priority_roster(WR=2)) == 0
    assert get_position_priority("RB", priority_roster(RB=1)) == 0
    assert get_position_priority("QB", priority_roster()) == 0


def test_position_priority_stays_zero_while_the_flex_can_absorb_a_player():
    # Starters are full at each of these, but the FLEX slot is still open.
    assert get_position_priority("RB", priority_roster(RB=2)) == 0
    assert get_position_priority("WR", priority_roster(WR=3)) == 0
    assert get_position_priority("TE", priority_roster(TE=1)) == 0


def test_position_priority_drops_once_another_position_has_taken_the_flex():
    # A 4th WR occupies the FLEX, so a 3rd RB is now bench rather than a starter.
    assert get_position_priority("RB", priority_roster(RB=2)) == 0
    assert get_position_priority("RB", priority_roster(RB=2, WR=4)) == 1


def test_position_priority_flex_credit_only_applies_to_flex_positions():
    # The FLEX is wide open, but a second QB still cannot start.
    assert get_position_priority("QB", priority_roster(QB=1)) == 1


def test_position_priority_sends_kicker_and_defense_straight_to_the_bottom():
    # No bench tier for K/DEF: once the starter is there, extras are lowest priority.
    assert get_position_priority("K", priority_roster(K=1)) == 2
    assert get_position_priority("DEF", priority_roster(DEF=1)) == 2


def test_position_priority_bottoms_out_at_two():
    assert get_position_priority("RB", priority_roster(RB=5)) == 2
    assert get_position_priority("WR", priority_roster(WR=6)) == 2
    assert get_position_priority("QB", priority_roster(QB=4)) == 2


@pytest.mark.xfail(
    strict=True,
    reason="Reads the module-level STARTING_POSITIONS instead of roster.starting_positions, "
           "so a roster configured for a different league is ignored. get_starting_games and "
           "roster_par do honour the roster's own config, so the two disagree.",
)
def test_position_priority_uses_the_rosters_own_starting_positions():
    roster = Roster({"QB": 3, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1}, 15)
    roster.drafted["QB"] = [Player(10.0, 16.0, 10.0, 1, "Q1", "QB")]
    # this league starts 3 QBs, so a second QB would still be a starter
    assert get_position_priority("QB", roster) == 0


@pytest.mark.xfail(
    strict=True,
    reason="Priority 1 is documented as 'would fill first bench spot', but "
           "drafted_less_starting <= 1 makes both the first and the second extra player "
           "score 1, so two QBs deep still reads as the first bench spot.",
)
def test_position_priority_marks_only_one_tier_as_the_first_bench_spot():
    # QB=1 -> the next QB is the first bench QB (1). QB=2 -> the next is the second (2).
    assert get_position_priority("QB", priority_roster(QB=1)) == 1
    assert get_position_priority("QB", priority_roster(QB=2)) == 2


#############################################################
# Tests for make_draft_position_pick
#############################################################

def adp_partable() -> pl.DataFrame:
    """One player per position, with distinct expected draft positions."""
    return make_partable([
        make_raw_row("RB_early", "RB", 20, adp_overall_rank=1),
        make_raw_row("WR_mid", "WR", 21, adp_overall_rank=5),
        make_raw_row("TE_late", "TE", 15, adp_overall_rank=20),
        make_raw_row("K_last", "K", 8, adp_overall_rank=40),
    ])


def test_draft_position_pick_takes_the_earliest_adp_among_top_priority_positions():
    # Empty roster: every position is priority 0, so the earliest ADP wins outright.
    assert make_draft_position_pick(priority_roster(), adp_partable(), kicker_min_round=10) \
        == ("RB_early", "RB")


def test_draft_position_pick_restricts_itself_to_the_highest_priority_positions():
    # Everything except TE is filled past its starters, so TE is the only priority 0.
    roster = priority_roster(QB=1, RB=3, WR=4, DEF=1, K=1)
    priorities = {pos: get_position_priority(pos, roster) for pos in ["QB", "RB", "WR", "TE", "DEF", "K"]}
    assert priorities["TE"] == 0 and min(priorities.values()) == 0
    assert [pos for (pos, p) in priorities.items() if p == 0] == ["TE"]

    # ...so it takes the TE even though RB_early has a much earlier ADP.
    assert make_draft_position_pick(roster, adp_partable(), kicker_min_round=10) \
        == ("TE_late", "TE")


def test_draft_position_pick_returns_a_name_and_position_pair():
    pick = make_draft_position_pick(priority_roster(), adp_partable(), kicker_min_round=10)
    assert isinstance(pick, tuple) and len(pick) == 2
    name, position = pick
    assert isinstance(name, str) and position in ["QB", "RB", "WR", "TE", "DEF", "K"]


@pytest.mark.xfail(
    strict=True,
    reason="BUG: the partable is filtered by position only, never against the roster, so a "
           "player already drafted is handed back again. subset_partable_not_on_roster "
           "already does this filtering for the combo search.",
)
def test_draft_position_pick_skips_players_already_on_the_roster():
    roster = priority_roster()
    roster.drafted["RB"] = [Player(10.0, 16.0, 10.0, 1, "RB_early", "RB")]
    assert make_draft_position_pick(roster, adp_partable(), kicker_min_round=10)[0] != "RB_early"


@pytest.mark.xfail(
    strict=True,
    reason="BUG: kicker_min_round is never read. `round` is computed and then unused, so a "
           "kicker can be taken before the round the caller asked to hold off until.",
)
def test_draft_position_pick_respects_kicker_min_round():
    # Everything but K is filled, making K the sole priority-0 position at round 12.
    roster = priority_roster(QB=1, RB=3, WR=4, TE=2, DEF=1)
    assert get_position_priority("K", roster) == 0
    assert make_draft_position_pick(roster, adp_partable(), kicker_min_round=99)[1] != "K"
