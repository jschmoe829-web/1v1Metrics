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
        winner = match.get('winner')
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
    
    match_details = []
    for _, match in matches.iterrows():
        t1_name = match.get('team1_name', '')
        t2_name = match.get('team2_name', '')
        t1_char = match.get('team1_players', match.get('team1_character_tag', ''))
        t2_char = match.get('team2_players', match.get('team2_character_tag', ''))
        t1_score = match.get('team1_scores', '')
        t2_score = match.get('team2_scores', '')
        winner = match.get('winner', '')
        
        match_details.append({
            "team1": t1_name,
            "team2": t2_name,
            "team1_character": t1_char,
            "team2_character": t2_char,
            "team1_score": t1_score,
            "team2_score": t2_score,
            "winner": winner
        })
    
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
        "matches": match_details
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
    mask = (df['team1_name'] == player_name) | (df['team2_name'] == player_name)
    if 'team1p1_username' in df.columns:
        mask = mask | (df['team1p1_username'] == player_name) | (df['team2p1_username'] == player_name)
    player_matches = df[mask]
    
    if player_matches.empty:
        return {
            "rank": 15, "win_rate": 0.50, "last5": 0.50,
            "experience": 100, "earnings": 500000, "followers": 10000
        }
    
    # Calculate stats
    wins = 0
    losses = 0
    
    for _, match in player_matches.iterrows():
        winner = match.get('winner')
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
    if 'team1p1_total_earnings' in df.columns and 'team1p1_username' in df.columns:
        if player_name in df['team1p1_username'].values:
            earnings = df[df['team1p1_username'] == player_name]['team1p1_total_earnings'].iloc[0]
        elif player_name in df['team2p1_username'].values:
            earnings = df[df['team2p1_username'] == player_name]['team2p1_total_earnings'].iloc[0]
    
    # Get followers
    followers = 0
    if 'team1p1_followers' in df.columns and 'team1p1_username' in df.columns:
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

def analyze_character_matchup(character1: str, character2: str):
    """Analyze matchup between two characters/teams."""
    df = get_data()
    
    if df is None or df.empty:
        return {
            "character1": character1, "character2": character2,
            "character1_wins": 0, "character2_wins": 0,
            "total_matches": 0, "character1_win_rate": 0,
            "character2_win_rate": 0, "leader": "N/A",
            "margin": 0, "has_history": False,
            "matches": []
        }
    
    t1_char_col = 'team1_character_tag' if 'team1_character_tag' in df.columns else 'team1_players'
    t2_char_col = 'team2_character_tag' if 'team2_character_tag' in df.columns else 'team2_players'
    
    if t1_char_col not in df.columns:
        return {
            "character1": character1, "character2": character2,
            "character1_wins": 0, "character2_wins": 0,
            "total_matches": 0, "character1_win_rate": 0,
            "character2_win_rate": 0, "leader": "N/A",
            "margin": 0, "has_history": False,
            "matches": []
        }
    
    mask = (
        ((df[t1_char_col] == character1) & (df[t2_char_col] == character2)) |
        ((df[t1_char_col] == character2) & (df[t2_char_col] == character1))
    )
    matches = df[mask]
    
    if matches.empty:
        return {
            "character1": character1, "character2": character2,
            "character1_wins": 0, "character2_wins": 0,
            "total_matches": 0, "character1_win_rate": 0,
            "character2_win_rate": 0, "leader": "N/A",
            "margin": 0, "has_history": False,
            "matches": []
        }
    
    char1_wins = 0
    char2_wins = 0
    total = len(matches)
    
    for _, match in matches.iterrows():
        winner = match.get('winner')
        if match.get(t1_char_col) == character1:
            if winner == match.get('team1_name'):
                char1_wins += 1
            else:
                char2_wins += 1
        else:
            if winner == match.get('team1_name'):
                char2_wins += 1
            else:
                char1_wins += 1
    
    c1_win_rate = (char1_wins / total) * 100 if total > 0 else 0
    c2_win_rate = (char2_wins / total) * 100 if total > 0 else 0
    
    if char1_wins > char2_wins:
        leader = character1
        margin = char1_wins - char2_wins
    elif char2_wins > char1_wins:
        leader = character2
        margin = char2_wins - char1_wins
    else:
        leader = "Tied"
        margin = 0
    
    match_details = []
    for _, match in matches.iterrows():
        t1_name = match.get('team1_name', '')
        t2_name = match.get('team2_name', '')
        t1_char = match.get('team1_players', match.get('team1_character_tag', ''))
        t2_char = match.get('team2_players', match.get('team2_character_tag', ''))
        t1_score = match.get('team1_scores', '')
        t2_score = match.get('team2_scores', '')
        winner = match.get('winner', '')
        
        match_details.append({
            "team1": t1_name,
            "team2": t2_name,
            "team1_character": t1_char,
            "team2_character": t2_char,
            "team1_score": t1_score,
            "team2_score": t2_score,
            "winner": winner
        })
    
    return {
        "character1": character1,
        "character2": character2,
        "character1_wins": char1_wins,
        "character2_wins": char2_wins,
        "total_matches": total,
        "character1_win_rate": c1_win_rate,
        "character2_win_rate": c2_win_rate,
        "leader": leader,
        "margin": margin,
        "has_history": total > 0,
        "matches": match_details
    }


def get_all_characters():
    """Get all unique characters from the dataset."""
    df = get_data()
    if df is None or df.empty:
        return []
    chars = set()
    for col in ['team1_character_tag', 'team2_character_tag', 'team1_players', 'team2_players']:
        if col in df.columns:
            chars.update(df[col].dropna().unique())
    return sorted([c for c in chars if c])


import pandas as pd
