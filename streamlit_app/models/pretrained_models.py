"""
Pre-trained ML models for match prediction.
This module provides pre-trained model functionality without requiring users to train models.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app.data.embedded_data import get_data
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class PretrainedPredictor:
    """Pre-trained model predictor for match outcomes."""
    
    def __init__(self):
        self.model = None
        self.margin_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize and train a pre-trained model with synthetic data based on real patterns."""
        np.random.seed(42)
        
        n_samples = 5000
        
        team1_rank = np.random.randint(1, 32, n_samples)
        team2_rank = np.random.randint(1, 32, n_samples)
        team1_win_rate = np.random.beta(5, 5, n_samples)
        team2_win_rate = np.random.beta(5, 5, n_samples)
        team1_last5 = np.random.beta(3, 2, n_samples)
        team2_last5 = np.random.beta(3, 2, n_samples)
        team1_experience = np.random.exponential(100, n_samples)
        team2_experience = np.random.exponential(100, n_samples)
        team1_followers = np.random.exponential(50000, n_samples)
        team2_followers = np.random.exponential(50000, n_samples)
        
        X = np.column_stack([
            team1_rank,
            team2_rank,
            team1_win_rate,
            team2_win_rate,
            team1_last5,
            team2_last5,
            team1_experience,
            team2_experience,
            team1_followers / 10000,
            team2_followers / 10000,
            (team1_rank < team2_rank).astype(int),
            (team1_win_rate > team2_win_rate).astype(int),
            (team1_last5 > team2_last5).astype(int),
        ])
        
        rank_diff = team2_rank - team1_rank
        win_rate_diff = team1_win_rate - team2_win_rate
        form_diff = team1_last5 - team2_last5
        
        win_prob = (
            0.3 * (rank_diff > 0).astype(float) +
            0.35 * (win_rate_diff > 0).astype(float) +
            0.2 * (form_diff > 0).astype(float) +
            0.15 * np.random.random(n_samples)
        )
        
        y = (win_prob > 0.5).astype(int)
        
        noise = np.random.random(n_samples) < 0.1
        y = np.where(noise, 1 - y, y)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        
        skill_diff = (team1_win_rate - team2_win_rate) * 20 + (team1_last5 - team2_last5) * 15 + (32 - team1_rank - (32 - team2_rank)) * 0.1
        y_margin = skill_diff + np.random.normal(0, 3, n_samples)
        
        self.margin_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.margin_model.fit(X_scaled, y_margin)
        self.is_trained = True
        
        self.feature_names = [
            'team1_rank', 'team2_rank',
            'team1_win_rate', 'team2_win_rate',
            'team1_last5', 'team2_last5',
            'team1_experience', 'team2_experience',
            'team1_followers', 'team2_followers',
            'rank_advantage', 'win_rate_advantage', 'form_advantage'
        ]
    
    def predict(self, player1_stats, player2_stats):
        """
        Predict match outcome between two players/teams.
        
        Args:
            player1_stats: dict with keys like 'rank', 'win_rate', 'last5', 'experience', 'earnings', 'followers'
            player2_stats: dict with same keys
            
        Returns:
            dict with prediction and probability
        """
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        features = [
            player1_stats.get('rank', 15),
            player2_stats.get('rank', 15),
            player1_stats.get('win_rate', 0.5),
            player2_stats.get('win_rate', 0.5),
            player1_stats.get('last5', 0.5),
            player2_stats.get('last5', 0.5),
            player1_stats.get('experience', 100),
            player2_stats.get('experience', 100),
            player1_stats.get('followers', 10000) / 10000,
            player2_stats.get('followers', 10000) / 10000,
            1 if player1_stats.get('rank', 15) < player2_stats.get('rank', 15) else 0,
            1 if player1_stats.get('win_rate', 0.5) > player2_stats.get('win_rate', 0.5) else 0,
            1 if player1_stats.get('last5', 0.5) > player2_stats.get('last5', 0.5) else 0,
        ]
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        margin_prediction = self.margin_model.predict(X_scaled)[0]
        
        predicted_winner = "Player 1" if prediction == 1 else "Player 2"
        
        if predicted_winner == "Player 1":
            score_margin = abs(float(margin_prediction))
        else:
            score_margin = -abs(float(margin_prediction))
        
        return {
            "prediction": predicted_winner,
            "confidence": float(max(probabilities)),
            "player1_probability": float(probabilities[1]),
            "player2_probability": float(probabilities[0]),
            "score_margin": round(score_margin, 1),
            "factors": self._get_prediction_factors(player1_stats, player2_stats)
        }
    
    def _get_prediction_factors(self, p1_stats, p2_stats):
        """Get key factors that influenced the prediction."""
        factors = []
        
        if p1_stats.get('rank', 15) < p2_stats.get('rank', 15):
            factors.append("Player 1 has better rank")
        elif p1_stats.get('rank', 15) > p2_stats.get('rank', 15):
            factors.append("Player 2 has better rank")
        
        if p1_stats.get('win_rate', 0.5) > p2_stats.get('win_rate', 0.5) + 0.1:
            factors.append("Player 1 has higher win rate")
        elif p2_stats.get('win_rate', 0.5) > p1_stats.get('win_rate', 0.5) + 0.1:
            factors.append("Player 2 has higher win rate")
        
        if p1_stats.get('last5', 0.5) > p2_stats.get('last5', 0.5) + 0.15:
            factors.append("Player 1 has better recent form")
        elif p2_stats.get('last5', 0.5) > p1_stats.get('last5', 0.5) + 0.15:
            factors.append("Player 2 has better recent form")
        
        if p1_stats.get('experience', 100) > p2_stats.get('experience', 100) * 1.5:
            factors.append("Player 1 has more experience")
        elif p2_stats.get('experience', 100) > p1_stats.get('experience', 100) * 1.5:
            factors.append("Player 2 has more experience")
        
        return factors


PLAYER_STATS_DATABASE = {
    "Bemo": {"rank": 2, "win_rate": 0.72, "last5": 0.75, "experience": 2800, "earnings": 4500000, "followers": 85000},
    "Probe": {"rank": 4, "win_rate": 0.65, "last5": 0.62, "experience": 3200, "earnings": 3800000, "followers": 72000},
    "NuckleDu": {"rank": 1, "win_rate": 0.88, "last5": 0.85, "experience": 3776, "earnings": 8500000, "followers": 150000},
    "PunkDaGod": {"rank": 3, "win_rate": 0.85, "last5": 0.80, "experience": 2592, "earnings": 6200000, "followers": 95000},
    "Yikkers": {"rank": 8, "win_rate": 0.86, "last5": 0.78, "experience": 441, "earnings": 2100000, "followers": 45000},
    "Drini": {"rank": 12, "win_rate": 0.58, "last5": 0.55, "experience": 1850, "earnings": 1800000, "followers": 38000},
    "Cleff": {"rank": 15, "win_rate": 0.55, "last5": 0.52, "experience": 2100, "earnings": 1200000, "followers": 28000},
    "Boogz": {"rank": 10, "win_rate": 0.62, "last5": 0.58, "experience": 1650, "earnings": 2100000, "followers": 52000},
    "Wesley": {"rank": 18, "win_rate": 0.52, "last5": 0.48, "experience": 980, "earnings": 750000, "followers": 22000},
    "tigerwoodsville": {"rank": 6, "win_rate": 0.70, "last5": 0.72, "experience": 1950, "earnings": 2800000, "followers": 48000},
    "ken99ovr": {"rank": 9, "win_rate": 0.68, "last5": 0.65, "experience": 1720, "earnings": 1950000, "followers": 35000},
    "Dcroft": {"rank": 14, "win_rate": 0.60, "last5": 0.58, "experience": 1450, "earnings": 1100000, "followers": 26000},
    "Soku": {"rank": 20, "win_rate": 0.50, "last5": 0.45, "experience": 850, "earnings": 650000, "followers": 18000},
    "OG2K": {"rank": 22, "win_rate": 0.48, "last5": 0.42, "experience": 720, "earnings": 420000, "followers": 12000},
    "Kiv": {"rank": 25, "win_rate": 0.45, "last5": 0.40, "experience": 580, "earnings": 280000, "followers": 8500},
    "YungLerFlex": {"rank": 11, "win_rate": 0.58, "last5": 0.52, "experience": 1380, "earnings": 950000, "followers": 22000},
    "Rayyblood": {"rank": 13, "win_rate": 0.55, "last5": 0.50, "experience": 1150, "earnings": 820000, "followers": 19000},
    "itsreallycruz": {"rank": 16, "win_rate": 0.53, "last5": 0.48, "experience": 980, "earnings": 580000, "followers": 15000},
    "leakqn": {"rank": 19, "win_rate": 0.51, "last5": 0.46, "experience": 750, "earnings": 420000, "followers": 11000},
    "AGRADEAP": {"rank": 17, "win_rate": 0.54, "last5": 0.50, "experience": 890, "earnings": 520000, "followers": 13500},
    "NinjaKiller": {"rank": 5, "win_rate": 1.0, "last5": 1.0, "experience": 99, "earnings": 3200000, "followers": 68000},
    "Team vKreuger": {"rank": 7, "win_rate": 1.0, "last5": 1.0, "experience": 52, "earnings": 1800000, "followers": 32000},
    "Team GetPicked": {"rank": 21, "win_rate": 1.0, "last5": 1.0, "experience": 32, "earnings": 450000, "followers": 8500},
    "streakdv": {"rank": 23, "win_rate": 0.83, "last5": 0.80, "experience": 157, "earnings": 850000, "followers": 18000},
    "EthxnH": {"rank": 24, "win_rate": 0.82, "last5": 0.78, "experience": 1258, "earnings": 2200000, "followers": 42000},
    "Team Jared": {"rank": 26, "win_rate": 0.82, "last5": 0.75, "experience": 1277, "earnings": 1950000, "followers": 38000},
    "GenghisD0n": {"rank": 27, "win_rate": 0.83, "last5": 0.82, "experience": 72, "earnings": 520000, "followers": 12000},
}


def get_player_stats(player_name):
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
    
    # Calculate win rate
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
    rank = 15
    if player_name in df['team1_name'].values:
        ranks = df[df['team1_name'] == player_name]['team1_rank'].dropna()
        if not ranks.empty:
            rank = int(ranks.iloc[0])
    elif player_name in df['team2_name'].values:
        ranks = df[df['team2_name'] == player_name]['team2_rank'].dropna()
        if not ranks.empty:
            rank = int(ranks.iloc[0])
    
    # Get experience (count of matches player appears in)
    exp = len(player_matches)
    
    # Get earnings
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
    
    # Get followers
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
        "last5": win_rate,  # Approximation
        "experience": exp,
        "earnings": earnings,
        "followers": followers
    }


def predict_match(player1, player2):
    """Predict match between two players using real data."""
    predictor = PretrainedPredictor()
    p1_stats = get_player_stats(player1)
    p2_stats = get_player_stats(player2)
    return predictor.predict(p1_stats, p2_stats)
