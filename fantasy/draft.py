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



import polars as pl
from dataclasses import dataclass
from typing import Self
from collections.abc import Generator

NUM_TEAMS = 12
ROSTER_SIZE = 15
WEEKS = 18
STARTING_POSITIONS = {"QB":1,"WR":3,"RB":2,"TE":1,"FLEX":1,"K":1,"DEF":1}

@dataclass(slots=True,order=True)
class Player():
    par_per_game:float
    expected_games_played:float
    par_per_game_flex:float
    bye:int
    name:str
    position:str


class Roster():
    def __init__(self, starting_positions:dict[str, int], roster_size:int):
        self.drafted = {
            "QB":[],
            "RB":[],
            "WR":[],
            "TE":[],
            "K":[],
            "DEF":[],
            "FLEX":[]
        } 
        assert set(starting_positions.keys()) == set(self.drafted.keys()), "starting positions incorrectly formatted"
        self.starting_positions = starting_positions
        self.roster_size = roster_size
    def get_players(self) -> list[Player]:
        return self.drafted["QB"] + self.drafted["RB"] + self.drafted["WR"] + self.drafted["TE"] + self.drafted["K"] + self.drafted["DEF"]


def get_starting_games(player:Player, drafted_list:list[Player], num_starters:int) -> float:
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
    bye_counts:dict[int, int] = {}
    games_covered = 0.0
    better_drafted_count = 0
    for drafted_player in drafted_list:
        if player > drafted_player:
            break
        better_drafted_count += 1
        bye_counts[drafted_player.bye] = bye_counts.get(drafted_player.bye, 0) + 1
        games_covered += drafted_player.expected_games_played
    # calculate available games
    available_games = num_starters * WEEKS - games_covered
    spots_from_bye_overlap = 0
    excess_spots_same_week = 0
    for (bye, count) in bye_counts.items():
        open_spots_week = num_starters - (better_drafted_count - count)
        # if there are no available games, if the byes have too many players then may be some, so positive correction
        if (bye != player.bye) and (open_spots_week > 0):
            spots_from_bye_overlap += 1
        if open_spots_week > (player.bye != bye):
            excess_spots_same_week += (open_spots_week - (player.bye != bye))
    available_games = available_games - excess_spots_same_week
    available_games = max(available_games, spots_from_bye_overlap)
    expected_start = min(available_games, player.expected_games_played)
    return expected_start

def modified_par(player:Player, roster:Roster, bench_penalty:float, position:str|None=None):
    """
    Calculates the modifed par for adding this player. Applies a simple bench_penalty
    if we don't project this player to be a starter. Accounts for bye's of players 
    who are already on the roster at this position.
    """
    position = player.position if position is None else position
    # return their modified par
    expected_start = get_starting_games(player, roster.drafted[position], roster.starting_positions[position])
    starting_value = player.par_per_game * expected_start
    if player.par_per_game >= 0:
        bench_value = (player.expected_games_played - expected_start) * bench_penalty * player.par_per_game
    else:
        bench_value = (player.expected_games_played - expected_start) * player.par_per_game
    return (starting_value + bench_value)

def get_flex_players(roster:Roster) -> list[Player]:
    """
    Function that given a roster, determines which players 
    are likely to be flex players. Used to assess flex player 
    modified PAR
    """
    positions = ["RB","WR","TE"]
    flex_player_list = []
    for pos in positions:
        drafted_players:list[Player] = roster.drafted[pos]
        players_iterated_over = []
        num_starters = roster.starting_positions[pos]
        for player in drafted_players:
            starting_games = get_starting_games(player, players_iterated_over, num_starters)
            if starting_games < player.expected_games_played:
                # in this case add to flex player list
                player_flex = Player(
                    player.par_per_game_flex, # par_per_game is relative to flex players
                    player.expected_games_played - starting_games, # reduce number of expected games available for flex
                    player.par_per_game_flex,
                    player.bye, 
                    player.name,
                    "FLEX" # change position to FLEX
                )
                flex_player_list.append(player_flex)
            players_iterated_over.append(player)
    flex_player_list.sort(reverse=True)
    return flex_player_list



def make_draft_order() -> list[int]:
    """
    Returns a snake pick order as a list
    """
    draft_order = []
    teams = [team for team in range(NUM_TEAMS)]
    reverse_teams = teams.copy()
    reverse_teams.reverse()
    for round in range(ROSTER_SIZE):
        if round % 2 == 0:
            draft_order += teams
        else:
            draft_order += reverse_teams
    return draft_order

def get_team_remaining_picks(pick:int, draft_order:list[int]) -> list[int]:
    # returns list of draft picks remaining for the team picking now
    team_id = draft_order[pick]
    team_picks = [i+pick for (i, team) in enumerate(draft_order[pick:]) if team == team_id]
    return team_picks

# def get_player(partable, position_rank, roster):

def subset_partable_not_on_roster(partable:pl.DataFrame, roster:Roster):
    subpartable = partable.clone()
    drafted = pl.DataFrame(
        [
            {"player":player.name, "position":player.position, "drafted":1} for player in roster.get_players()
        ]
    )
    not_drafted = subpartable.join(drafted, on=["player","position"], how="left").filter(pl.col("drafted").is_null()).drop("drafted")
    return not_drafted

def get_draftable_positions(team_picks, position_ignore) -> list[list[str]]:
    """
    List of draftable positions for each of the team's picks per position ignore
    """
    positions = {"QB","RB","WR","TE","DEF","K"}
    if position_ignore is None:
        return [sorted(positions) for _ in team_picks]
    draftable = []
    for pick_num in range(len(team_picks)):
        ignored:set = {pos for (pos, ignore_for) in position_ignore.items() if ignore_for >= pick_num}    
        draftable.append(
            sorted(positions.difference(ignored))
        )
    return draftable

def get_player(overall_pick_num, position, pos_num, not_drafted_df) -> Player:
    """
    Gets player at position, pos num, who hasn't been drafted yet
    - overall pick num is the pick number in the draft
    - position is the str position
    - pos_num is the rank at that position by PAR
    - not drafted df is the polars dataframe with par info, etc
    """
    top_player_dict = not_drafted_df.filter(
        (pl.col("drafted_after") >= overall_pick_num) & (pl.col("position") == position) & (~pl.col("drafted"))
    ).sort(
        by="PAR", descending=True
    ).row(pos_num, named=True)

    player = Player(
        top_player_dict["par_per_game"],
        top_player_dict["expected_games_played"],
        top_player_dict["flex_par_per_game"],
        top_player_dict["bye"],
        top_player_dict["player"],
        top_player_dict["position"]
    )

    return player

def set_player_draft_status(player:Player, status:bool, not_drafted_df:pl.DataFrame) -> pl.DataFrame:
    """
    Sets players drafted status for not_drafted_df
    """
    # mark player as drafted in not_drafted_df
    player_pred = (pl.col("player") == player.name) & (pl.col("position") == player.position) 
    return not_drafted_df.with_columns(
        pl.when(player_pred).then(status).otherwise(pl.col("drafted")).alias("drafted")
    )

def get_draft_combos(team_picks, roster, partable, consider_per_position=1, position_ignore=None) -> Generator[list[Player]]:
    """
    Iterator that returns a set of possible next picks 
    - team_picks is list of remaining picks for to optimize for the team
    - roster is current roster
    - partable is the polars dataframe with player scoring/edp data
    - consider_per_position is number of players to consider per position (sorted by PAR)
    - position_ignore has a dictionary of positions to ignore
        key: position
        value: # of team picks to avoid positions for
    """
    not_drafted_df = subset_partable_not_on_roster(partable, roster).with_columns(pl.lit(False).alias("drafted"))
    selected_player_ranks:list[tuple[int,int]] = [] # first is pos #, second is k/# considered
    selected_players = []
    draftable_positions = get_draftable_positions(team_picks, position_ignore)
    while True:
        while len(selected_players) < len(team_picks):
            # this block adds selections to selected_players
            selected_player_ranks.append((0, 0))
            team_pick_num = len(selected_players)
            overall_pick_num = team_picks[team_pick_num]
            pos_to_draft = draftable_positions[team_pick_num][0]
            # gets top player
            top_player = get_player(overall_pick_num, pos_to_draft, 0, not_drafted_df)
            selected_players.append(top_player)
            # sets their status to drafted
            not_drafted_df = set_player_draft_status(top_player, True, not_drafted_df)

        yield selected_players

        while True:
            last_player:Player = selected_players.pop()
            # mark them undrafted
            not_drafted_df = set_player_draft_status(last_player, False, not_drafted_df)
            (pos_id, pos_rank) = selected_player_ranks.pop()
            team_pick_num = len(selected_players)
            if pos_rank >= consider_per_position:
                # in this case we move to next position
                pos_id += 1
                pos_rank = 0
            else:
                pos_rank += 1
            if pos_id >= len(draftable_positions[team_pick_num]):
                # in this case we've exhausted positions to draft at this pick
                continue
            # otherwise we add next player to pick
            selected_player_ranks.append((pos_id, pos_rank))
            overall_pick_num = team_picks[team_pick_num]
            pos_to_draft = draftable_positions[team_pick_num][pos_id]
            top_player = get_player(overall_pick_num, pos_to_draft, pos_rank, not_drafted_df)
            selected_players.append(top_player)
            # set status to drafted
            not_drafted_df = set_player_draft_status(top_player, True, not_drafted_df)
            break
                
        # if there are no more players to iterate for first pick, exit
        if len(selected_players) == 0:
            break


def best_draft_sequence(
        pick:int, 
        draft_order:list[int], 
        roster:Roster,
        partable:pl.DataFrame, 
        k=1,
        bench_penalty=0.5,
        through_round=10,
        position_ignore={"K":10,"DEF":10}):
    """
    Best expected sequence for remaining picks
    """
    # get remaining picks for this team
    # team_picks = get_team_remaining_picks(pick, draft_order)
    # best_player_combo = []
    # for player_combos in get_draft_combos(team_picks, roster, partable, consider_per_position=k, through_round=through_round, position_ignore=position_ignore):
        # add players to roster
        # assess value
        # if best, save as best_player_combo

    # # for each pick assess top k players in each position group for modified par
    # for remaining_pick in team_picks:
    #     for pos in ["QB","RB","WR","TE","DEF","K"]:
    #         is_flex = pos in ["RB","WR","TE"]
    #         pos_and_available = (pl.col("position") == pos) & (pl.col("drafted_after") >= remaining_pick)
    #         # select top k players by par 
    #         pos_df:pl.DataFrame = partable.filter(
    #             pos_and_available
    #         ).limit(
    #             k
    #         ).sort(
    #             by="PAR",
    #             descending=True
    #         ).select(
    #             [
    #                 "par_per_game",
    #                 "expected_games_played",
    #                 "flex_par_per_game",
    #                 "bye",
    #                 "player",
    #                 "position"
    #             ]
    #         )
    #         pos_to_consider = [Player(*row) for row in pos_df.iter_rows()]
    #         mpar = [modified_par(player, roster, bench_penalty, pos) for player in pos_to_consider]
    #         if is_flex:
    #             # consider their flex value
    #             mpar_flex = [modified_par(player, roster, bench_penalty, "FLEX") for player in pos_to_consider]
    #             mpar = [max(mp, mpf) for (mp, mpf) in zip(mpar, mpar_flex)]
    #         best_index = mpar.index(max(mpar))
    #         best_by_position.append(pos_to_consider[best_index])            
    pass




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

