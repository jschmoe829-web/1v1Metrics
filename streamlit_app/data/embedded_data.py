"""
Embedded data module - loads full CSV data into memory for use in the app.
Data is kept in memory only and not exposed to users.
"""

import pandas as pd
import os

# Global DataFrame - loaded once, kept in memory
_df = None

def get_data():
    """Get the full DataFrame, loading it if necessary."""
    global _df
    if _df is None:
        load_data()
    return _df

def load_data(csv_path=None):
    """Load the CSV data into memory."""
    global _df
    
    if csv_path is None:
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), '1v1me_stakes_20260227_204857.csv'),
            os.path.join(os.path.dirname(__file__), '..', '1v1me_stakes_20260227_204857.csv'),
            'streamlit_app/1v1me_stakes_20260227_204857.csv',
            '1v1me_stakes_20260227_204857.csv',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
    
    if csv_path and os.path.exists(csv_path):
        _df = pd.read_csv(csv_path, low_memory=False)
        return _df
    else:
        return pd.DataFrame()

def reload_data():
    """Force reload of data."""
    global _df
    _df = None
    return get_data()

# Legacy exports for compatibility
GAME_DISTRIBUTION = None
WIN_DISTRIBUTION = None
TOP_WINNERS = None
SEASON_DISTRIBUTION = None
CHARACTER_DISTRIBUTION = None
TOP_TEAMS_BY_WIN_RATE = None
EARNINGS_PERCENTILES = None
TOTAL_MATCHES = None
TOP_PLAYERS_TEAMS = None
GAME_OPTIONS = None
HEAD_TO_HEAD_HISTORY = None

def compute_aggregates():
    """Compute all aggregates from the data (run after loading)."""
    global GAME_DISTRIBUTION, WIN_DISTRIBUTION, TOP_WINNERS, SEASON_DISTRIBUTION
    global CHARACTER_DISTRIBUTION, TOP_TEAMS_BY_WIN_RATE, EARNINGS_PERCENTILES
    global TOTAL_MATCHES, TOP_PLAYERS_TEAMS, GAME_OPTIONS, HEAD_TO_HEAD_HISTORY
    
    df = get_data()
    if df is None or df.empty:
        return
    
    # Game distribution
    GAME_DISTRIBUTION = df['game_name'].value_counts().head(15).to_dict()
    
    # Win distribution
    team1_wins = int((df['team1_placement'] == 1).sum())
    team2_wins = int((df['team2_placement'] == 1).sum())
    WIN_DISTRIBUTION = {"Team 1": team1_wins, "Team 2": team2_wins}
    
    # Top winners
    TOP_WINNERS = df[df['winner_name'].notna()]['winner_name'].value_counts().head(20).to_dict()
    
    # Season distribution
    if 'season_state_title_compact' in df.columns:
        SEASON_DISTRIBUTION = df['season_state_title_compact'].value_counts().to_dict()
    
    # Character distribution
    if 'team1_character_tag' in df.columns:
        CHARACTER_DISTRIBUTION = df['team1_character_tag'].value_counts().head(20).to_dict()
    
    # Team win rates
    team_stats = {}
    for idx, row in df.iterrows():
        for team_col, wins_col, completed_col in [
            ('team1_name', 'team1_wins', 'team1_completed'),
            ('team2_name', 'team2_wins', 'team2_completed')
        ]:
            team = row.get(team_col)
            if pd.notna(team) and team:
                if team not in team_stats:
                    team_stats[team] = {'wins': 0, 'completed': 0}
                team_stats[team]['wins'] += row.get(wins_col, 0)
                team_stats[team]['completed'] += row.get(completed_col, 0)
    
    win_rates = []
    for team, stats in team_stats.items():
        if stats['completed'] > 30:  # Min 30 games
            win_rates.append({
                'team': team,
                'win_rate': stats['wins']/stats['completed']*100,
                'games': stats['completed']
            })
    win_rates.sort(key=lambda x: x['win_rate'], reverse=True)
    TOP_TEAMS_BY_WIN_RATE = win_rates[:20]
    
    # Earnings percentiles
    earnings = df['team1p1_total_earnings'].dropna()
    earnings = earnings[earnings > 0]
    if len(earnings) > 0:
        EARNINGS_PERCENTILES = {
            "p25": float(earnings.quantile(0.25)),
            "p50": float(earnings.quantile(0.50)),
            "p75": float(earnings.quantile(0.75)),
            "p95": float(earnings.quantile(0.95))
        }
    
    # Total matches
    TOTAL_MATCHES = len(df)
    
    # Top players/teams for dropdowns
    all_names = set()
    for col in ['team1_name', 'team2_name', 'team1p1_username', 'team2p1_username']:
        if col in df.columns:
            all_names.update(df[col].dropna().unique())
    TOP_PLAYERS_TEAMS = sorted(list(all_names))[:200]
    
    # Game options
    GAME_OPTIONS = ["All Games"] + sorted(df['game_name'].unique().tolist())
    
    # Head-to-head history - compute all matchups
    h2h = {}
    for idx, row in df.iterrows():
        team1 = row.get('team1_name')
        team2 = row.get('team2_name')
        winner = row.get('winner_name')
        
        if pd.notna(team1) and pd.notna(team2) and pd.notna(winner):
            key = tuple(sorted([team1, team2]))
            if key not in h2h:
                h2h[key] = {'wins': {}, 'total': 0}
            h2h[key]['wins'][winner] = h2h[key]['wins'].get(winner, 0) + 1
            h2h[key]['total'] += 1
    
    HEAD_TO_HEAD_HISTORY = {}
    for key, data in h2h.items():
        if data['total'] >= 3:  # At least 3 matches
            wins = data['wins']
            if len(wins) == 2:
                team1, team2 = key
                HEAD_TO_HEAD_HISTORY[key] = {
                    team1: wins.get(team1, 0),
                    team2: wins.get(team2, 0),
                    'total': data['total']
                }

# Initialize on import
try:
    load_data()
    compute_aggregates()
except:
    pass
