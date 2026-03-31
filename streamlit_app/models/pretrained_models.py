"""
Pre-trained ML models for match prediction.
Loads pre-trained models for winner, margin, and blowout predictions.
Supports game-specific models for better accuracy.
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streamlit_app.data.embedded_data import get_data


class PretrainedPredictor:
    """Loads pre-trained models for match predictions."""
    
    def __init__(self, game=None):
        self.models = None
        self.is_trained = False
        self.game = game
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained models."""
        model_dir = os.path.dirname(os.path.abspath(__file__))
        
        if self.game:
            model_path = os.path.join(model_dir, 'all_games_models.pkl')
        else:
            model_path = os.path.join(model_dir, 'trained_model.pkl')
        
        if not os.path.exists(model_path):
            self.is_trained = False
            return
        
        try:
            with open(model_path, 'rb') as f:
                all_models = pickle.load(f)
            
            if self.game and self.game in all_models:
                self.models = all_models[self.game]
            elif self.game is None and 'winner_model' in all_models:
                self.models = all_models
            else:
                self.models = all_models.get('Madden NFL', all_models.get('winner_model'))
            
            self.is_trained = True
            self.feature_names = self.models.get('feature_names', [])
        except Exception as e:
            self.is_trained = False
            print(f"Error loading model: {e}")
    
    def predict(self, player1_stats, player2_stats):
        """Predict match outcome between two players."""
        if not self.is_trained:
            return {"error": "No trained model available"}
        
        features = self._build_features(player1_stats, player2_stats)
        
        X = np.array([features])
        
        winner_model = self.models['winner_model']
        winner_scaler = self.models['winner_scaler']
        X_scaled = winner_scaler.transform(X)
        
        prediction = winner_model.predict(X_scaled)[0]
        probabilities = winner_model.predict_proba(X_scaled)[0]
        
        predicted_winner = "Player 1" if prediction == 1 else "Player 2"
        
        margin_model = self.models['margin_model']
        margin_scaler = self.models['margin_scaler']
        X_margin_scaled = margin_scaler.transform(X)
        predicted_margin = int(margin_model.predict(X_margin_scaled)[0])
        
        blowout_model = self.models['blowout_model']
        blowout_scaler = self.models['blowout_scaler']
        X_blowout_scaled = blowout_scaler.transform(X)
        blowout_prob = float(blowout_model.predict_proba(X_blowout_scaled)[0][1])
        
        return {
            "prediction": predicted_winner,
            "confidence": float(max(probabilities)),
            "player1_probability": float(probabilities[1]),
            "player2_probability": float(probabilities[0]),
            "predicted_margin": predicted_margin,
            "blowout_probability": blowout_prob,
            "is_blowout": blowout_prob > 0.5,
            "factors": self._get_prediction_factors(player1_stats, player2_stats)
        }
    
    def _build_features(self, p1_stats, p2_stats):
        p1_rank = p1_stats.get('rank') or 15
        p2_rank = p2_stats.get('rank') or 15
        p1_wr = p1_stats.get('win_rate') or 0.5
        p2_wr = p2_stats.get('win_rate') or 0.5
        p1_last5 = p1_stats.get('last5') or 0.5
        p2_last5 = p2_stats.get('last5') or 0.5
        p1_exp = p1_stats.get('experience') or 100
        p2_exp = p2_stats.get('experience') or 100
        
        earnings1 = p1_stats.get('earnings') or 500000
        earnings2 = p2_stats.get('earnings') or 500000
        total_earnings = earnings1 + earnings2 + 1
        earnings_ratio = earnings1 / total_earnings
        
        followers1 = p1_stats.get('followers') or 10000
        followers2 = p2_stats.get('followers') or 10000
        total_followers = followers1 + followers2 + 1
        followers_ratio = followers1 / total_followers
        
        stake_ratio1 = p1_stats.get('stake_ratio') or 1.0
        stake_ratio2 = p2_stats.get('stake_ratio') or 1.0
        
        is_partner1 = 1 if p1_stats.get('is_partner') else 0
        is_partner2 = 1 if p2_stats.get('is_partner') else 0
        
        h2h_wr = p1_stats.get('h2h_winrate', 0.5)
        h2h_games = p1_stats.get('h2h_games', 0)
        
        char_wr = p1_stats.get('char_matchup_wr', 0.5)
        char_games = p1_stats.get('char_matchup_games', 0)
        
        p1_momentum = p1_stats.get('momentum', 0.5)
        p2_momentum = p2_stats.get('momentum', 0.5)
        
        return [
            p1_rank, p2_rank,
            p1_wr, p2_wr,
            p1_last5, p2_last5,
            p1_exp, p2_exp,
            p1_wr - p2_wr,
            1 if p1_rank < p2_rank else 0,
            h2h_wr,
            h2h_games,
            char_wr,
            char_games,
            p1_momentum,
            p2_momentum,
            earnings_ratio,
            followers_ratio,
            stake_ratio1,
            stake_ratio2,
            is_partner1,
            is_partner2,
        ]
    
    def _get_prediction_factors(self, p1_stats, p2_stats):
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


def get_available_games():
    """Get list of games with trained models."""
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, 'all_games_models.pkl')
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                models = pickle.load(f)
            return models.get('__games__', [])
        except:
            pass
    return []


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


PLAYER_STATS_DATABASE = {
    "Bemo": {"rank": 2, "win_rate": 0.72, "last5": 0.75, "experience": 2800, "earnings": 4500000, "followers": 85000},
    "Probe": {"rank": 4, "win_rate": 0.65, "last5": 0.62, "experience": 3200, "earnings": 3800000, "followers": 72000},
    "NuckleDu": {"rank": 1, "win_rate": 0.88, "last5": 0.85, "experience": 3776, "earnings": 8500000, "followers": 150000},
    "PunkDaGod": {"rank": 3, "win_rate": 0.85, "last5": 0.80, "experience": 2592, "earnings": 6200000, "followers": 95000},
}


def predict_match(player1, player2, game=None):
    """Predict match between two players using game-specific model."""
    predictor = PretrainedPredictor(game=game)
    p1_stats = get_player_stats(player1)
    p2_stats = get_player_stats(player2)
    result = predictor.predict(p1_stats, p2_stats)
    result['game'] = game
    return result
