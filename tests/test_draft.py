
import polars as pl
import pytest

from fantasy.draft import (
    NUM_TEAMS,
    ROSTER_SIZE,
    WEEKS,
    Player,
    Roster,
    get_draft_combos,
    get_draftable_positions,
    get_flex_players,
    get_player,
    get_starting_games,
    modified_par,
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