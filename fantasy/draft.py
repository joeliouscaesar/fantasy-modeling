# TODO
# - modified_par_flex

# file to optimize draft pick

# Algorithm will (hopefully) work by repeatedly calculating the order in which players 
# should be drafted, which could possibly converge (at least for the next several picks)

# First round
# For all picks: 
# - Given a roster, and series of future picks, calculate the modified value add for top k players for each of those series of picks.
# - Pick player from highest scoring sequence of picks.
# 
# nth round, for n > 1
# - Update draft positions of players based on the n-1th round
# - Repeat selection steps as in the first round for all picks
# - If the selections are identical or we hit a maxiter, terminate

import polars as pl
from dataclasses import dataclass


NUM_TEAMS = 12
ROSTER_SIZE = 15
WEEKS = 18
STARTING_POSITIONS = {"QB":1,"WR":3,"RB":2,"TE":1,"FLEX":1,"K":1,"DEF":1}


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

@dataclass(slots=True,order=True)
class Player():
    par_per_game:float
    expected_games_played:float
    par_per_game_flex:float
    bye:int
    name:str
    position:str

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


def remaining_draft_picks(pick:int, draft_order:list[int], rosters:dict[int, Roster], partable, k=5, bench_penalty=0.5, max_iter=10):
    """
    Returns sequence of remaining picks
    """
    # calculate modified PAR for top 5 players in each position group
    # sorted by PAR
    pass

  
# def main():
#     partable = pl.read_csv("outputs/par_table.csv")
#     is_flex = pl.col("position").is_in(["RB","WR","TE"])
#     flex_replacement_value = partable.filter(
#         is_flex
#     ).select(
#         "replacement_ppg"    
#     ).max()

#     partable = partable.with_columns(
#         pl.when(is_flex).then(
#             pl.col("expected_points_per_game") - flex_replacement_value["replacement_ppg"]
#         ).otherwise(
#             pl.lit(None)
#         ).alias("flex_par_per_game")
#     )
