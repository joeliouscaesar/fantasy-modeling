## Quick World

For draft optimization, thinking about the value add for starting roles throughout the year

QB:  X X X X ... 
RB1: X X X X ...
RB2: X X X X ...
...
DEF: X X X X ... 

High level, if we have $n$ available starting weeks for a given position, for available players at that position we cap their PAR at 

$ \min (Par, n \times ParPG)$

This is fine and well, but we have to consider 

1. How to penalize open starting spots in the algorithm so that points are optimized
2. How to value players when there aren't starting spots.

The first I think is pretty simple, we assign values a penalty $\alpha $ for a given slot, which could start at zero (e.g. we fill with a replacement player) or negative (worse than replacement player).

Second point is more tricky. There's definitely a drop off in value if we are filling with players who won't start. Maybe we do the following:
- if their ppg projection is higher than enough players on roster such that they would start some games, we can give them credit for those games
- we do a quick injury adjustment, so we get injury probability $p$ by position, for players with higher ppg than them. Then we can get the expected number of available games counting injuries, maybe we assume players play half the expected games if they get hurt. So say we have 2 players drafted at a position with $n$ spots, and they are expected to take up $m_1, m_2 $ spots respectively, then we can do something like 
$$n - (1-p)^2(m_1+m_2)- p(1-p)(m_1\times 3/2+m_2\times 3/2) - p^2 (m_1+m_2)/2$$


And just making sure those terms we multiply times the probabilities are never bigger than n, so that there is at least some small possibility for value. So like

$$n - (1-p)^2\min(m_1+m_2, n) ... $$

At this point it definitely undervalues players who are close in expected points per game played but below those currently on the roster.


## (More) Ideal World
We have distributions for games played and points per game. We draw the games played and points per game for each player, and go through the analysis. as in part 1. Problem is mostly computational feasibility. Just feels like a lot more work, especially for something real time. 

## Algorithm 
For a potential pick, we calculate their modified PAR. Then, we assume this player was selected, and move on to the next pick, doing the same optimization for them. We continue through all teams until the team from the original pick as made all their picks. This allows us to calculate the teams overall score for picks. 

The biggest issue with this is it's a lot of combos and which grows exponentially. 

12 teams, 15 rounds, 7 position groups

Yeah lmao this is way to big of a number. We need a simplification.


## New Plan
At a given point in the draft, we have a pool of players who are available, and we have expected draft orders for those players (from websites). For each of a team's picks, we can construct a pool of players who we expect to be available. Then we can pick an optimal path through those players. Biggest issue is it disregards other teams picking.

Would love if there was some kind of convergence. Okay what about this. 

First Round: We use our estimates, we simulate the whole draft. 

Second Round: Instead of our estimates of where people will be drafted, we use where they get drafted in the first round. 

Third Round: Etc. Etc. 

So we slowly introduce team's picking optimally into the process. Good too because it shouldn't take too long? 

## Notes
- Something to account for injury status for the games played projection (Tucker Kraft)
- Build in projections for different leagues, maybe add more of those details to the league configs
- Think about replacement defn, probably want it to be relative to the top free agents, not top starters.
- In the draft optimization could re-calculate PAR after each round? True sense of who will be available. 

1. Look through Claude code, add initial overall rankings to that par_table.csv.
2. I'm envision a roster class, 


