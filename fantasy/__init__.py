"""Fantasy football PAR (points above replacement) modeling.

Pipeline
--------
1. projections.load_projections  -> tidy FantasyPros season projections + per-game rate
2. history.build_player_history   -> historical games-played + features (keyed by gsis_id)
3. games_model                    -> pluggable models predicting expected games played
4. par.build_par_table            -> replacement levels + final PAR table

See build_par.py for the end-to-end entrypoint.
"""
