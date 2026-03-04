"""
Analysis Module - uses full data from embedded_data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.embedded_data import HEAD_TO_HEAD_HISTORY, TOP_PLAYERS_TEAMS, get_data


def analyze_matchup(player1: str, player2: str):
    """Analyze head-to-head matchup between two players using full data."""
    df = get_data()
    
    if df is None or df.empty:
        return {
            "player1": player1, "player2": player2,
            "player1_wins": 0, "player2_wins": 0,
            "total_matches": 0, "player1_win_rate": 0,
            "player2_win_rate": 0, "leader": "N/A",
            "margin": 0, "has_history": False,
            "matches": []
        }
    
    # Find matches between these two players
    mask = (
        ((df['team1_name'] == player1) & (df['team2_name'] == player2)) |
        ((df['team1_name'] == player2) & (df['team2_name'] == player1))
    )
    matches = df[mask]
    
    if matches.empty:
        # Try with usernames too
        mask = (
            ((df['team1p1_username'] == player1) & (df['team2p1_username'] == player2)) |
            ((df['team1p1_username'] == player2) & (df['team2p1_username'] == player1))
        )
        matches = df[mask]
    
    if matches.empty:
        return {
            "player1": player1, "player2": player2,
            "player1_wins": 0, "player2_wins": 0,
            "total_matches": 0, "player1_win_rate": 0,
            "player2_win_rate": 0, "leader": "N/A",
            "margin": 0, "has_history": False,
            "matches": []
        }
    
    # Count wins
    player1_wins = 0
    player2_wins = 0
    total = len(matches)
    
    for _, match in matches.iterrows():
        winner = match.get('winner_name')
        if winner == player1:
            player1_wins += 1
        elif winner == player2:
            player2_wins += 1
    
    p1_win_rate = (player1_wins / total) * 100 if total > 0 else 0
    p2_win_rate = (player2_wins / total) * 100 if total > 0 else 0
    
    if player1_wins > player2_wins:
        leader = player1
        margin = player1_wins - player2_wins
    elif player2_wins > player1_wins:
        leader = player2
        margin = player2_wins - player1_wins
    else:
        leader = "Tied"
        margin = 0
    
    return {
        "player1": player1,
        "player2": player2,
        "player1_wins": player1_wins,
        "player2_wins": player2_wins,
        "total_matches": total,
        "player1_win_rate": p1_win_rate,
        "player2_win_rate": p2_win_rate,
        "leader": leader,
        "margin": margin,
        "has_history": total > 0,
        "matches": matches.to_dict('records')
    }


def get_available_players():
    """Get all players from the full dataset."""
    return TOP_PLAYERS_TEAMS


def get_player_stats(player_name: str):
    """Get player stats from the full dataset."""
    df = get_data()
    
    if df is None or df.empty:
        return {
            "rank": 15, "win_rate": 0.50, "last5": 0.50,
            "experience": 100, "earnings": 500000, "followers": 10000
        }
    
    # Find all matches for this player
    player_matches = df[
        (df['team1_name'] == player_name) |
        (df['team2_name'] == player_name) |
        (df['team1p1_username'] == player_name) |
        (df['team2p1_username'] == player_name)
    ]
    
    if player_matches.empty:
        return {
            "rank": 15, "win_rate": 0.50, "last5": 0.50,
            "experience": 100, "earnings": 500000, "followers": 10000
        }
    
    # Calculate stats
    wins = 0
    losses = 0
    
    for _, match in player_matches.iterrows():
        winner = match.get('winner_name')
        if winner == player_name:
            wins += 1
        else:
            losses += 1
    
    total = wins + losses
    win_rate = wins / total if total > 0 else 0.5
    
    # Get rank
    if player_name in df['team1_name'].values:
        rank_row = df[df['team1_name'] == player_name].iloc[0]
        rank = rank_row.get('team1_rank', 15)
    elif player_name in df['team2_name'].values:
        rank_row = df[df['team2_name'] == player_name].iloc[0]
        rank = rank_row.get('team2_rank', 15)
    else:
        rank = 15
    
    # Get experience (count of matches player appears in)
    exp = len(player_matches)
    
    # Get earnings
    earnings = 0
    if player_name in df['team1p1_username'].values:
        earnings = df[df['team1p1_username'] == player_name]['team1p1_total_earnings'].iloc[0]
    elif player_name in df['team2p1_username'].values:
        earnings = df[df['team2p1_username'] == player_name]['team2p1_total_earnings'].iloc[0]
    
    # Get followers
    followers = 0
    if player_name in df['team1p1_username'].values:
        followers = df[df['team1p1_username'] == player_name]['team1p1_followers'].iloc[0]
    elif player_name in df['team2p1_username'].values:
        followers = df[df['team2p1_username'] == player_name]['team2p1_followers'].iloc[0]
    
    return {
        "rank": int(rank) if pd.notna(rank) else 15,
        "win_rate": win_rate,
        "last5": win_rate,  # Approximate
        "experience": int(exp) if pd.notna(exp) else total,
        "earnings": float(earnings) if pd.notna(earnings) else 500000,
        "followers": int(followers) if pd.notna(followers) else 10000
    }

import pandas as pd
