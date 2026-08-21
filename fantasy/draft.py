# TODO

# file to optimize draft pick
# Want to do the following
# - For a given team at a given pick, want to consider players available at all future picks (including current)
# - Construct possible combinations of future picks
# - Assess these combinations by maximizing the teams overall PAR
# - Make the pick that maximizes this
#
# Function should take in as inputs
# - Current roster
# - Current pick
# - Number of picks in the future to consider
# - Position exlcusions (K/DEF), could be dict with # of picks to ignore them for

# current active_roster
# -> maybe an iterator that returns a sequence of players to be drafted?
# -> given that players, add to roster
# -> assess roster value
# -> track best roster
# -> return best roster


from collections import defaultdict
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Self

import polars as pl


@dataclass(slots=True, order=True, frozen=True)
class Player:
    par_per_game: float
    expected_games_played: float
    par_per_game_flex: float
    bye: int
    name: str
    position: str


class Roster:
    def __init__(self, starting_positions: dict[str, int], roster_size: int):
        self.drafted = {
            "QB": [],
            "RB": [],
            "WR": [],
            "TE": [],
            "K": [],
            "DEF": [],
            "FLEX": [],
        }
        assert set(starting_positions.keys()) == set(self.drafted.keys()), (
            "starting positions incorrectly formatted"
        )
        self.starting_positions = starting_positions
        self.roster_size = roster_size

    def get_players(self) -> list[Player]:
        return (
            self.drafted["QB"]
            + self.drafted["RB"]
            + self.drafted["WR"]
            + self.drafted["TE"]
            + self.drafted["K"]
            + self.drafted["DEF"]
        )

    def copy(self) -> Self:
        """
        Returns a roster that can be drafted to without affecting this one.

        Each position list is copied, but the Players in them are shared rather than
        duplicated. That is safe because Player is frozen, and it keeps branching the
        draft search cheap enough to do per candidate combination.
        """
        new_roster = type(self)(self.starting_positions, self.roster_size)
        new_roster.drafted = {
            pos: players.copy() for (pos, players) in self.drafted.items()
        }
        return new_roster

    def add_player(self, player: Player):
        self.drafted[player.position].append(player)
        self.drafted[player.position].sort(reverse=True)


def get_starting_games(
    player: Player, drafted_list: list[Player], num_starters: int, weeks: int = 18
) -> float:
    """
    Function that gets the number of games a player can start at a specified position.

    Assumes player lists are sorted by PAR per game

    The bye logic is complicated but the thought is the following:
    - the number of spots needed to be filled in a week is (# starters - (# drafted - # with bye that week))
    - if a player does not have a bye that week, that week can count for exactly one starting spot, the accumulation of these
        weeks is a lower bound on available starting games for that player
    - on the other hand, if a player also has that bye, they can't fill any of those open spots, or if there is more than one
        open spot, they can only fill one. therefore we have to have a downward correction on available games by

        (# starters - (# drafted - # with bye that week)) - (players bye is different week)
    """
    # iterate over players drafted ahead of this player
    # determine gross number of available games, then correct for bye weeks
    bye_counts: dict[int, int] = {}
    games_covered = 0.0
    better_drafted_count = 0
    for drafted_player in drafted_list:
        if player > drafted_player:
            break
        better_drafted_count += 1
        bye_counts[drafted_player.bye] = bye_counts.get(drafted_player.bye, 0) + 1
        games_covered += drafted_player.expected_games_played
    # calculate available games
    available_games = num_starters * weeks - games_covered
    spots_from_bye_overlap = 0
    excess_spots_same_week = 0
    for bye, count in bye_counts.items():
        open_spots_week = num_starters - (better_drafted_count - count)
        # if there are no available games, if the byes have too many players then may be some, so positive correction
        if (bye != player.bye) and (open_spots_week > 0):
            spots_from_bye_overlap += 1
        if open_spots_week > (player.bye != bye):
            excess_spots_same_week += open_spots_week - (player.bye != bye)
    available_games = available_games - excess_spots_same_week
    available_games = max(available_games, spots_from_bye_overlap)
    expected_start = min(available_games, player.expected_games_played)
    return expected_start


def modified_par(
    player: Player,
    roster: Roster,
    bench_penalty: float,
    position: str | None = None,
    weeks: int = 18,
):
    """
    Calculates the modifed par for adding this player. Applies a simple bench_penalty
    if we don't project this player to be a starter. Accounts for bye's of players
    who are already on the roster at this position.
    """
    position = player.position if position is None else position
    # return their modified par
    expected_start = get_starting_games(
        player, roster.drafted[position], roster.starting_positions[position], weeks
    )
    starting_value = player.par_per_game * expected_start
    if player.par_per_game >= 0:
        bench_value = (
            (player.expected_games_played - expected_start)
            * bench_penalty
            * player.par_per_game
        )
    else:
        bench_value = (
            player.expected_games_played - expected_start
        ) * player.par_per_game
    return starting_value + bench_value


def get_flex_players(roster: Roster, weeks: int = 18) -> list[Player]:
    """
    Function that given a roster, determines which players
    are likely to be flex players. Used to assess flex player
    modified PAR
    """
    positions = ["RB", "WR", "TE"]
    flex_player_list = []
    for pos in positions:
        drafted_players: list[Player] = roster.drafted[pos]
        players_iterated_over = []
        num_starters = roster.starting_positions[pos]
        for player in drafted_players:
            starting_games = get_starting_games(
                player, players_iterated_over, num_starters, weeks
            )
            if starting_games < player.expected_games_played:
                # in this case add to flex player list
                player_flex = Player(
                    player.par_per_game_flex,  # par_per_game is relative to flex players
                    player.expected_games_played
                    - starting_games,  # reduce number of expected games available for flex
                    player.par_per_game_flex,
                    player.bye,
                    player.name,
                    "FLEX",  # change position to FLEX
                )
                flex_player_list.append(player_flex)
            players_iterated_over.append(player)
    flex_player_list.sort(reverse=True)
    return flex_player_list


def make_draft_order(num_teams: int = 12, roster_size: int = 15) -> list[int]:
    """
    Returns a snake pick order as a list
    """
    draft_order = []
    teams = [team for team in range(num_teams)]
    reverse_teams = teams.copy()
    reverse_teams.reverse()
    for round in range(roster_size):
        if round % 2 == 0:
            draft_order += teams
        else:
            draft_order += reverse_teams
    return draft_order


def get_team_remaining_picks(pick: int, draft_order: list[int]) -> list[int]:
    # returns list of draft picks remaining for the team picking now
    team_id = draft_order[pick]
    team_picks = [
        i + pick for (i, team) in enumerate(draft_order[pick:]) if team == team_id
    ]
    return team_picks


# def get_player(partable, position_rank, roster):


def subset_partable_not_on_roster(
    partable: pl.DataFrame, roster: Roster
) -> pl.DataFrame:
    subpartable = partable.clone()
    players_on_roster = roster.get_players()
    if len(players_on_roster) == 0:
        return subpartable
    drafted = pl.DataFrame(
        [
            {"player": player.name, "position": player.position, "drafted": 1}
            for player in roster.get_players()
        ]
    )
    not_drafted = (
        subpartable.join(drafted, on=["player", "position"], how="left")
        .filter(pl.col("drafted").is_null())
        .drop("drafted")
    )
    return not_drafted


def get_draftable_positions(team_picks, position_ignore) -> list[list[str]]:
    """
    List of draftable positions for each of the team's picks per position ignore

    position_ignore values are a count of picks, so {"K":2} suppresses K at the
    team's first two picks (indices 0 and 1) and allows it from the third on.
    """
    positions = {"QB", "RB", "WR", "TE", "DEF", "K"}
    if position_ignore is None:
        return [sorted(positions) for _ in team_picks]
    draftable = []
    for pick_num in range(len(team_picks)):
        ignored: set = {
            pos
            for (pos, ignore_for) in position_ignore.items()
            if ignore_for > pick_num
        }
        draftable.append(sorted(positions.difference(ignored)))
    return draftable


def get_available_players(overall_pick_num, position, not_drafted_df) -> pl.DataFrame:
    """
    Players at a position still available at this pick, best PAR first
    """
    return not_drafted_df.filter(
        (pl.col("drafted_after") >= overall_pick_num)
        & (pl.col("position") == position)
        & (~pl.col("drafted"))
    ).sort(by="PAR", descending=True)


def _player_from_row(row: dict) -> Player:
    return Player(
        row["par_per_game"],
        row["expected_games_played"],
        row["flex_par_per_game"],
        row["bye"],
        row["player"],
        row["position"],
    )


def get_player(overall_pick_num, position, pos_num, not_drafted_df) -> Player:
    """
    Gets player at position, pos num, who hasn't been drafted yet
    - overall pick num is the pick number in the draft
    - position is the str position
    - pos_num is the rank at that position by PAR
    - not drafted df is the polars dataframe with par info, etc

    Raises if there is no player at that rank. Use get_player_or_none to search
    across positions without having to catch that.
    """
    available = get_available_players(overall_pick_num, position, not_drafted_df)
    return _player_from_row(available.row(pos_num, named=True))


def get_player_or_none(
    overall_pick_num, position, pos_num, not_drafted_df
) -> Player | None:
    """
    Same as get_player but returns None when the position is too thin to have a
    player at pos_num, rather than raising
    """
    available = get_available_players(overall_pick_num, position, not_drafted_df)
    if pos_num >= available.height:
        return None
    return _player_from_row(available.row(pos_num, named=True))


def set_player_draft_status(
    player: Player, status: bool, not_drafted_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Sets players drafted status for not_drafted_df
    """
    # mark player as drafted in not_drafted_df
    player_pred = (pl.col("player") == player.name) & (
        pl.col("position") == player.position
    )
    return not_drafted_df.with_columns(
        pl.when(player_pred).then(status).otherwise(pl.col("drafted")).alias("drafted")
    )


def _next_selection(
    team_pick_num, pos_id, pos_rank, team_picks, draftable_positions, not_drafted_df
):
    """
    Finds the next available player for a pick, starting the search at
    (pos_id, pos_rank) and walking forward through the position list.

    Positions too thin to have a player at that rank are skipped rather than
    raising, which happens once the good players at a position are used up.

    Returns (player, pos_id, pos_rank), or None if this pick has no options left.
    """
    overall_pick_num = team_picks[team_pick_num]
    positions = draftable_positions[team_pick_num]
    while pos_id < len(positions):
        player = get_player_or_none(
            overall_pick_num, positions[pos_id], pos_rank, not_drafted_df
        )
        if player is not None:
            return (player, pos_id, pos_rank)
        # not enough players left at this position, so try the next one
        pos_id += 1
        pos_rank = 0
    return None


def get_draft_combos(
    team_picks, roster, partable, consider_per_position=1, position_ignore=None
) -> Generator[list[Player]]:
    """
    Iterator that returns a set of possible next picks
    - team_picks is list of remaining picks for to optimize for the team
    - roster is current roster
    - partable is the polars dataframe with player scoring/edp data
    - consider_per_position is number of players to consider per position (sorted by PAR),
        so 1 considers only the best available player at each position
    - position_ignore has a dictionary of positions to ignore
        key: position
        value: # of team picks to avoid positions for

    Each combo is yielded as its own list, so callers can hold on to one (to track
    the best roster so far) without it changing as the search continues.
    """
    if consider_per_position < 1:
        raise ValueError("consider_per_position must be at least 1")
    if len(team_picks) == 0:
        return
    not_drafted_df = subset_partable_not_on_roster(partable, roster).with_columns(
        pl.lit(False).alias("drafted")
    )
    selected_player_ranks: list[
        tuple[int, int]
    ] = []  # first is pos #, second is k/# considered
    selected_players = []
    draftable_positions = get_draftable_positions(team_picks, position_ignore)
    while True:
        while len(selected_players) < len(team_picks):
            # this block adds selections to selected_players
            team_pick_num = len(selected_players)
            selection = _next_selection(
                team_pick_num, 0, 0, team_picks, draftable_positions, not_drafted_df
            )
            if selection is None:
                # no player can fill this pick, so back up and try another branch
                break
            (top_player, pos_id, pos_rank) = selection
            selected_player_ranks.append((pos_id, pos_rank))
            selected_players.append(top_player)
            # sets their status to drafted
            not_drafted_df = set_player_draft_status(top_player, True, not_drafted_df)

        if len(selected_players) == len(team_picks):
            # copy so the caller keeps this combo even as we mutate ours
            yield list(selected_players)

        while True:
            if len(selected_players) == 0:
                # no more players to iterate for the first pick, so we're done
                return
            last_player: Player = selected_players.pop()
            # mark them undrafted
            not_drafted_df = set_player_draft_status(last_player, False, not_drafted_df)
            (pos_id, pos_rank) = selected_player_ranks.pop()
            team_pick_num = len(selected_players)
            pos_rank += 1
            if pos_rank >= consider_per_position:
                # considered enough at this position, so move to the next one
                pos_id += 1
                pos_rank = 0
            selection = _next_selection(
                team_pick_num,
                pos_id,
                pos_rank,
                team_picks,
                draftable_positions,
                not_drafted_df,
            )
            if selection is None:
                # exhausted the positions to draft at this pick, so back up further
                continue
            # otherwise we add next player to pick
            (top_player, pos_id, pos_rank) = selection
            selected_player_ranks.append((pos_id, pos_rank))
            selected_players.append(top_player)
            # set status to drafted
            not_drafted_df = set_player_draft_status(top_player, True, not_drafted_df)
            break


def add_picks_to_roster(draft_combos: list[Player], roster: Roster) -> Roster:
    """
    Returns a new roster with players added
    """
    roster = roster.copy()
    for player in draft_combos:
        roster.drafted[player.position].append(player)
        roster.drafted[player.position].sort(reverse=True)
    return roster


def roster_par(roster: Roster, bench_penalty=0.5, weeks: int = 18) -> float:
    total_par = 0.0
    flex_players: list[Player] = []
    for pos in ["QB", "RB", "WR", "TE", "DEF", "K"]:
        pos_players: list[Player] = roster.drafted[pos]
        for i, player in enumerate(pos_players):
            other_drafted = pos_players[:i] + pos_players[i + 1 :]
            expected_start = get_starting_games(
                player, other_drafted, roster.starting_positions[player.position], weeks
            )
            total_par += (
                player.par_per_game * expected_start
            )  # add starting value for this player at position
            if pos in ["RB", "WR", "TE"] and (
                expected_start < player.expected_games_played
            ):
                flex_players.append(
                    Player(
                        player.par_per_game,
                        player.expected_games_played - expected_start,
                        player.par_per_game_flex,
                        player.bye,
                        player.name,
                        "FLEX",
                    )
                )
            else:
                total_par += (
                    (player.expected_games_played - expected_start)
                    * player.par_per_game
                    * bench_penalty
                )
    # now go through flex players
    flex_players.sort(reverse=True)
    for i, player in enumerate(flex_players):
        other_drafted = flex_players[:i] + flex_players[i + 1 :]
        expected_start = get_starting_games(
            player, other_drafted, roster.starting_positions["FLEX"], weeks
        )
        # add their flex starting/bench values
        total_par += (
            player.par_per_game_flex * expected_start
        )  # add starting value for this player at FLEX
        total_par += (
            (player.expected_games_played - expected_start)
            * player.par_per_game_flex
            * bench_penalty
        )
    return total_par


def best_draft_sequence(
    pick: int,
    draft_order: list[int],
    roster: Roster,
    partable: pl.DataFrame,
    consider_per_position=1,
    bench_penalty=0.5,
    through_round=10,
    position_ignore: dict[str, int] | None = None,
    weeks: int = 18,
) -> list[Player]:
    """
    Best expected sequence for remaining picks

    Returns an empty list when the roster has already reached through_round.
    """
    if position_ignore is None:
        position_ignore = {"K": 10, "DEF": 10}
    # get remaining picks for this team
    team_picks = get_team_remaining_picks(pick, draft_order)
    # clamp at 0 so a roster already past through_round plans nothing. a negative
    # rounds_left would slice picks off the end of the list instead of emptying it
    rounds_left = max(through_round - len(roster.get_players()), 0)
    team_picks = team_picks[:rounds_left]
    # default covers there being no picks left to plan, where max would otherwise raise
    return max(
        get_draft_combos(
            team_picks, roster, partable, consider_per_position, position_ignore
        ),
        key=lambda dc: roster_par(
            add_picks_to_roster(dc, roster), bench_penalty=bench_penalty, weeks=weeks
        ),
        default=[],
    )


def get_position_priority(position: str, roster: Roster) -> int:
    """
    Returns the priority for drafting a player at a given position
    - 0 is high priority, player could start
    - 1 is lower, would fill first bench spot (except kickers/def)
    - 2 is lowest, second bench spot and beyond
    """
    drafted_less_starting = (
        len(roster.drafted[position]) - roster.starting_positions[position]
    )
    is_flex = position in ["RB", "WR", "TE"]
    if drafted_less_starting < 0:
        # highest priority
        return 0
    # if we have an open flex spot, priority 0
    flexes_open = roster.starting_positions["FLEX"]
    if is_flex:
        for pos in ["RB", "WR", "TE"]:
            flexes_open = flexes_open - max(
                0, len(roster.drafted[pos]) - roster.starting_positions[pos]
            )
        if flexes_open > 0:
            return 0
    if (position not in ["K", "DEF"]) and (drafted_less_starting == 0):
        # first bench priority 1
        return 1
    # elif is_flex and (flexes_open == 0):
    #     # backup flex we value at 1
    #     return 1
    else:
        return 2


def make_priority_based_pick(
    roster: Roster,
    partable: pl.DataFrame,
    criteria_col: str,
    criteria_sort_descending: bool,
    kicker_min_round: int,
    def_min_round: int,
) -> tuple[str, str] | None:
    """
    Function which picks a player based on their expected draft position and
    the position priority based on roster construction. Returns (player name, position) pair.

    Prioritize higher priority position (lowest int), and then expected draft position
    for equal priorities
    """
    if criteria_col not in partable.columns:
        raise ValueError(f"criteria_col {criteria_col} not found in partable")

    round = len(roster.get_players()) + 1
    pos_priorities: dict[int, list[str]] = defaultdict(list)
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_priority = get_position_priority(pos, roster)
        pos_priorities[pos_priority].append(pos)

    min_priority = min(pos_priorities.keys())
    max_priority = max(pos_priorities.keys())

    def_priority = (
        get_position_priority("DEF", roster)
        if round >= def_min_round
        else max_priority + 1
    )
    pos_priorities[def_priority].append("DEF")

    k_priority = (
        get_position_priority("K", roster)
        if round >= kicker_min_round
        else max_priority + 1
    )
    pos_priorities[k_priority].append("K")

    for priority in range(min_priority, max_priority + 2):
        highest_priority_pos = pos_priorities.get(priority, None)
        if highest_priority_pos is None:
            continue
        lowest_adp_player: pl.DataFrame = (
            partable.filter(pl.col("position").is_in(highest_priority_pos))
            .sort(by=[criteria_col], descending=criteria_sort_descending)
            .head(1)
        )

        # check if players
        if lowest_adp_player.shape[0] == 0:
            continue

        lowest_adp_player_d: dict = lowest_adp_player.row(0, named=True)
        return (lowest_adp_player_d["player"], lowest_adp_player_d["position"])

    return None


def full_draft(
    partable: pl.DataFrame,
    strategies: dict[int, Callable],
    num_teams: int = 12,
    roster_size: int = 15,
    starting_positions: dict[str, int] | None = None,
) -> pl.DataFrame:
    """
    runs through the draft, for each team either drafting per the best_draft_sequence
    or a simplified process using highest expected draft position of players available

    Returns a dataframe with a columns for the player picked, position, pick number, and team id

    Params:
    - draft_order is the full list of draft picks with the values being which team is picking there
    - partable is the dataframe with player information
    - strategies
       dictionary of functions used by each team for picking players.
       all strategy functions should map from

       (
           pick:int,
           team_id:int,
           draft_order:list[int],
           remaining_players:pl.DataFrame,
           rosters:list[Roster]
       ) -> tuple[player, position]

    """
    if starting_positions is None:
        starting_positions = {
            "QB": 1,
            "WR": 3,
            "RB": 2,
            "TE": 1,
            "FLEX": 1,
            "K": 1,
            "DEF": 1,
        }

    remaining_players = partable.clone()
    rosters = [Roster(starting_positions, roster_size) for _ in range(num_teams)]
    draft_order = make_draft_order(num_teams, roster_size)
    drafted_tuples: list[tuple[str, str, int, int]] = []
    for pick, team_id in enumerate(draft_order):
        strategy = strategies[team_id]
        (player, position) = strategy(
            pick, team_id, draft_order, remaining_players, rosters
        )
        drafted_tuples.append((player, position, pick + 1, team_id))

        # get drafted player
        drafted_player_df: pl.DataFrame = remaining_players.filter(
            (pl.col("player") == player) & (pl.col("position") == position)
        )
        assert drafted_player_df.shape[0] == 1
        drafted_player: Player = _player_from_row(drafted_player_df.row(0, named=True))
        rosters[team_id].add_player(drafted_player)

        # remove from remaining players
        remaining_players = remaining_players.filter(
            ~((pl.col("player") == player) & (pl.col("position") == position))
        )
    # prepare output df
    outputdf = pl.DataFrame(
        drafted_tuples,
        schema={
            "player": pl.String,
            "position": pl.String,
            "pick_number": pl.Int64,
            "team_id": pl.Int64,
        },
        orient="row",
    )
    return outputdf


# import polars as pl
# partable = pl.read_csv("outputs/par_table.csv")
# k = 5
# pos = "QB"


# is_flex = pl.col("position").is_in(["RB","WR","TE"])
# flex_replacement_value = partable.filter(
#     is_flex
# ).select(
#     "replacement_ppg"
# ).max()

# partable = partable.with_columns(
#     pl.when(is_flex).then(
#         pl.col("expected_points_per_game") - flex_replacement_value["replacement_ppg"]
#     ).otherwise(
#         pl.lit(None)
#     ).alias("flex_par_per_game"),

#     pl.when(pl.col("position") == "DST").then(
#         pl.lit("DEF")
#     ).otherwise(
#         pl.col("position")
#     ).alias("position"),

#     (pl.col("adp_overall_rank").fill_null(NUM_TEAMS*ROSTER_SIZE + 1) - 1).alias("drafted_after")
# )

# assert partable[["player","position"]].n_unique() == partable.shape[0]
