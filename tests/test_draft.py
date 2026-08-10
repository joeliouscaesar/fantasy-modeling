
from fantasy.draft import WEEKS, Player, Roster, get_starting_games, get_flex_players, modified_par

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