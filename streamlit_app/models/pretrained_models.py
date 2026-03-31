"""
Pre-trained ML models for match prediction.
Loads a pre-trained model for match predictions.
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streamlit_app.data.embedded_data import get_data


class PretrainedPredictor:
    """Loads pre-trained model for match predictions."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained model and scaler."""
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, 'trained_model.pkl')
        scaler_path = os.path.join(model_dir, 'trained_scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            self.is_trained = False
            return
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            self.feature_names = [
                'team1_rank', 'team2_rank',
                'team1_win_rate', 'team2_win_rate',
                'team1_last5', 'team2_last5',
                'team1_exp', 'team2_exp',
                'win_rate_diff', 'rank_advantage'
            ]
        except Exception as e:
            self.is_trained = False
    
    def predict(self, player1_stats, player2_stats):
        """Predict match outcome between two players."""
        if not self.is_trained:
            return {"error": "No trained model available"}
        
        p1_rank = player1_stats.get('rank') or 15
        p2_rank = player2_stats.get('rank') or 15
        p1_wr = player1_stats.get('win_rate') or 0.5
        p2_wr = player2_stats.get('win_rate') or 0.5
        p1_last5 = player1_stats.get('last5') or 0.5
        p2_last5 = player2_stats.get('last5') or 0.5
        p1_exp = player1_stats.get('experience') or 100
        p2_exp = player2_stats.get('experience') or 100
        
        features = [
            p1_rank, p2_rank,
            p1_wr, p2_wr,
            p1_last5, p2_last5,
            p1_exp, p2_exp,
            p1_wr - p2_wr,
            1 if p1_rank < p2_rank else 0,
        ]
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        predicted_winner = "Player 1" if prediction == 1 else "Player 2"
        
        return {
            "prediction": predicted_winner,
            "confidence": float(max(probabilities)),
            "player1_probability": float(probabilities[1]),
            "player2_probability": float(probabilities[0]),
            "factors": self._get_prediction_factors(player1_stats, player2_stats)
        }
    
    def _get_prediction_factors(self, p1_stats, p2_stats):
        """Get key factors that influenced the prediction."""
        factors = []
        
        p1_wr = p1_stats.get('win_rate', 0.5)
        p2_wr = p2_stats.get('win_rate', 0.5)
        
        if p1_stats.get('rank', 15) < p2_stats.get('rank', 15):
            factors.append("Player 1 has better rank")
        elif p1_stats.get('rank', 15) > p2_stats.get('rank', 15):
            factors.append("Player 2 has better rank")
        
        if p1_wr > p2_wr + 0.1:
            factors.append("Player 1 has higher win rate")
        elif p2_wr > p1_wr + 0.1:
            factors.append("Player 2 has higher win rate")
        
        if p1_stats.get('last5', 0.5) > p2_stats.get('last5', 0.5) + 0.15:
            factors.append("Player 1 has better recent form")
        elif p2_stats.get('last5', 0.5) > p1_stats.get('last5', 0.5) + 0.15:
            factors.append("Player 2 has better recent form")
        
        return factors


def get_player_stats(player_name):
    """Get player stats from the full dataset."""
    df = get_data()
    
    if df is None or df.empty:
        return {
            "rank": 15, "win_rate": 0.50, "last5": 0.50,
            "experience": 100, "earnings": 500000, "followers": 10000
        }
    
    mask = (df['team1_name'] == player_name) | (df['team2_name'] == player_name)
    if 'team1p1_username' in df.columns:
        mask = mask | (df['team1p1_username'] == player_name) | (df['team2p1_username'] == player_name)
    player_matches = df[mask]
    
    if player_matches.empty:
        return {
            "rank": 15, "win_rate": 0.50, "last5": 0.50,
            "experience": 100, "earnings": 500000, "followers": 10000
        }
    
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
    
    rank = 15
    if player_name in df['team1_name'].values:
        ranks = df[df['team1_name'] == player_name]['team1_rank'].dropna()
        if not ranks.empty:
            rank = int(ranks.iloc[0])
    elif player_name in df['team2_name'].values:
        ranks = df[df['team2_name'] == player_name]['team2_rank'].dropna()
        if not ranks.empty:
            rank = int(ranks.iloc[0])
    
    exp = len(player_matches)
    
    earnings = 500000
    if 'team1p1_total_earnings' in df.columns and 'team1p1_username' in df.columns:
        if player_name in df['team1p1_username'].values:
            earnings_vals = df[df['team1p1_username'] == player_name]['team1p1_total_earnings'].dropna()
            if not earnings_vals.empty:
                earnings = float(earnings_vals.iloc[0])
        elif player_name in df['team2p1_username'].values:
            earnings_vals = df[df['team2p1_username'] == player_name]['team2p1_total_earnings'].dropna()
            if not earnings_vals.empty:
                earnings = float(earnings_vals.iloc[0])
    
    followers = 10000
    if 'team1p1_followers' in df.columns and 'team1p1_username' in df.columns:
        if player_name in df['team1p1_username'].values:
            follower_vals = df[df['team1p1_username'] == player_name]['team1p1_followers'].dropna()
            if not follower_vals.empty:
                followers = int(follower_vals.iloc[0])
        elif player_name in df['team2p1_username'].values:
            follower_vals = df[df['team2p1_username'] == player_name]['team2p1_followers'].dropna()
            if not follower_vals.empty:
                followers = int(follower_vals.iloc[0])
    
    return {
        "rank": rank,
        "win_rate": win_rate,
        "last5": win_rate,
        "experience": exp,
        "earnings": earnings,
        "followers": followers
    }


def predict_match(player1, player2):
    """Predict match between two players using trained model."""
    predictor = PretrainedPredictor()
    p1_stats = get_player_stats(player1)
    p2_stats = get_player_stats(player2)
    return predictor.predict(p1_stats, p2_stats)
