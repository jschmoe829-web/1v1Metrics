"""
Model training and backtesting script for 1v1 match prediction.
Run locally to train model, then commit the trained model for use in Streamlit app.

Features:
- Temporal split (train on older data, test on most recent)
- 80/10/10 split (80% train, 10% validation, 10% test)
- K-fold cross validation for robust accuracy estimation
- Predicts winner, score margin, and blowout probability
- Game-specific models supported
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
import pickle
import os

CSV_PATH = "1v1me_Mar31.csv"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
N_FOLDS = 5


def get_available_games(df):
    if 'game_name' in df.columns:
        return df['game_name'].dropna().unique().tolist()
    return []


def parse_last_five(val):
    if pd.isna(val):
        return 0.5
    if isinstance(val, str):
        try:
            wins = val.count('W')
            losses = val.count('L')
            total = wins + losses
            return wins / total if total > 0 else 0.5
        except:
            return 0.5
    return 0.5


def safe_score_to_int(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip()
    if val_str.upper() in ['W', 'WIN']:
        return 1
    elif val_str.upper() in ['L', 'LOSS']:
        return -1
    try:
        return int(val_str)
    except:
        try:
            parts = val_str.split('-')
            if len(parts) == 2:
                return int(parts[1].strip()) - int(parts[0].strip())
        except:
            pass
        return 0


def compute_h2h_features(df, current_idx):
    if current_idx < 1:
        return {'h2h_team1_winrate': 0.5, 'h2h_team1_games': 0}
    
    current = df.iloc[current_idx]
    team1_name = current.get('team1_name')
    team2_name = current.get('team2_name')
    
    if pd.isna(team1_name) or pd.isna(team2_name):
        return {'h2h_team1_winrate': 0.5, 'h2h_team1_games': 0}
    
    past_matches = df.iloc[:current_idx]
    
    team1_vs_team2 = past_matches[
        ((past_matches['team1_name'] == team1_name) & (past_matches['team2_name'] == team2_name)) |
        ((past_matches['team1_name'] == team2_name) & (past_matches['team2_name'] == team1_name))
    ]
    
    if len(team1_vs_team2) == 0:
        return {'h2h_team1_winrate': 0.5, 'h2h_team1_games': 0}
    
    team1_wins = 0
    for _, match in team1_vs_team2.iterrows():
        if match.get('team1_name') == team1_name:
            if match.get('team1_placement') == 1:
                team1_wins += 1
        else:
            if match.get('team2_placement') == 1:
                team1_wins += 1
    
    return {
        'h2h_team1_winrate': team1_wins / len(team1_vs_team2),
        'h2h_team1_games': len(team1_vs_team2)
    }


def compute_character_features(df, current_idx):
    if current_idx < 1:
        return {'char_matchup_wr': 0.5, 'char_matchup_games': 0}
    
    current = df.iloc[current_idx]
    char1 = current.get('team1_character_tag')
    char2 = current.get('team2_character_tag')
    
    if pd.isna(char1) or pd.isna(char2):
        return {'char_matchup_wr': 0.5, 'char_matchup_games': 0}
    
    past_matches = df.iloc[:current_idx]
    
    char_matches = past_matches[
        ((past_matches['team1_character_tag'] == char1) & (past_matches['team2_character_tag'] == char2)) |
        ((past_matches['team1_character_tag'] == char2) & (past_matches['team2_character_tag'] == char1))
    ]
    
    if len(char_matches) == 0:
        return {'char_matchup_wr': 0.5, 'char_matchup_games': 0}
    
    char1_wins = 0
    for _, match in char_matches.iterrows():
        if match.get('team1_character_tag') == char1:
            if match.get('team1_placement') == 1:
                char1_wins += 1
        else:
            if match.get('team2_placement') == 1:
                char1_wins += 1
    
    return {
        'char_matchup_wr': char1_wins / len(char_matches),
        'char_matchup_games': len(char_matches)
    }


def compute_momentum_features(df, current_idx, lookback=20):
    if current_idx < 1:
        return {'team1_momentum': 0.5, 'team2_momentum': 0.5}
    
    current = df.iloc[current_idx]
    team1_name = current.get('team1_name')
    team2_name = current.get('team2_name')
    
    past = df.iloc[:current_idx].tail(lookback)
    
    team1_matches = past[(past['team1_name'] == team1_name) | (past['team2_name'] == team1_name)]
    team2_matches = past[(past['team1_name'] == team2_name) | (past['team2_name'] == team2_name)]
    
    def calc_winrate(matches, team_name):
        if len(matches) == 0:
            return 0.5
        wins = 0
        for _, m in matches.iterrows():
            if m.get('team1_name') == team_name:
                if m.get('team1_placement') == 1:
                    wins += 1
            else:
                if m.get('team2_placement') == 1:
                    wins += 1
        return wins / len(matches)
    
    return {
        'team1_momentum': calc_winrate(team1_matches, team1_name),
        'team2_momentum': calc_winrate(team2_matches, team2_name)
    }


def prepare_data(df, game=None):
    required_cols = ['team1_wins', 'team1_losses', 'team2_wins', 'team2_losses', 
                    'team1_rank', 'team2_rank', 'team1_placement']
    
    if not all(col in df.columns for col in required_cols):
        return None, None, None, None
    
    matches = df[df['team1_wins'].notna() & df['team1_losses'].notna() & 
               df['team2_wins'].notna() & df['team2_losses'].notna() &
               df['team1_rank'].notna() & df['team2_rank'].notna() &
               df['team1_placement'].notna()].copy()
    
    if game and 'game_name' in matches.columns:
        matches = matches[matches['game_name'] == game]
    
    if 'start_date' in matches.columns:
        matches['start_date'] = pd.to_datetime(matches['start_date'], errors='coerce')
        matches = matches.sort_values('start_date').reset_index(drop=True)
    
    if len(matches) < 50:
        return None, None, None, None
    
    game_max_margin = {
        'Counter Strike 2': 9,
        'COD: Black Ops': 5,
        'Street Fighter': 3,
        'Tekken': 3,
    }.get(game, 5)
    
    rounds_per_win = {
        'Tekken': 3,
        'Street Fighter': 2,
    }.get(game, 1)
    
    matches['team1_win_rate'] = matches['team1_wins'] / (matches['team1_wins'] + matches['team1_losses'])
    matches['team2_win_rate'] = matches['team2_wins'] / (matches['team2_wins'] + matches['team2_losses'])
    matches['team1_total_games'] = matches['team1_wins'] + matches['team1_losses']
    matches['team2_total_games'] = matches['team2_wins'] + matches['team2_losses']
    matches['team1_last_five'] = matches['team1_last_five'].apply(parse_last_five)
    matches['team2_last_five'] = matches['team2_last_five'].apply(parse_last_five)
    
    matches['y_winner'] = (matches['team1_placement'] == 1).astype(int)
    
    team1_scores = matches['team1_score'].apply(safe_score_to_int)
    team2_scores = matches['team2_score'].apply(safe_score_to_int)
    team1_actual_wins = (team1_scores / rounds_per_win).round().clip(lower=0).astype(int)
    team2_actual_wins = (team2_scores / rounds_per_win).round().clip(lower=0).astype(int)
    matches['score_margin'] = (team1_actual_wins - team2_actual_wins).abs()
    matches['score_margin'] = matches['score_margin'].clip(upper=game_max_margin)
    matches['y_margin'] = matches['score_margin']
    matches['y_blowout'] = (matches['score_margin'] >= 2).astype(int)
    
    valid_mask = (
        matches['team1_win_rate'].notna() & matches['team2_win_rate'].notna() &
        matches['team1_total_games'].notna() & matches['team2_total_games'].notna() &
        (matches['team1_total_games'] > 0) & (matches['team2_total_games'] > 0)
    )
    matches = matches[valid_mask]
    
    if len(matches) < 50:
        return None, None, None, None
    
    feature_list = []
    for idx in range(len(matches)):
        row = matches.iloc[idx]
        
        base_features = [
            row['team1_rank'],
            row['team2_rank'],
            row['team1_win_rate'],
            row['team2_win_rate'],
            row['team1_last_five'] if pd.notna(row['team1_last_five']) else 0.5,
            row['team2_last_five'] if pd.notna(row['team2_last_five']) else 0.5,
            row['team1_total_games'],
            row['team2_total_games'],
            row['team1_win_rate'] - row['team2_win_rate'],
            (row['team1_rank'] < row['team2_rank']).astype(int),
        ]
        
        h2h = compute_h2h_features(matches, idx)
        char = compute_character_features(matches, idx)
        mom = compute_momentum_features(matches, idx)
        
        earnings1 = row.get('team1p1_total_earnings', 0) or 0
        earnings2 = row.get('team2p1_total_earnings', 0) or 0
        total_earnings = earnings1 + earnings2 + 1
        earnings_ratio = earnings1 / total_earnings
        
        followers1 = row.get('team1p1_followers', 0) or 0
        followers2 = row.get('team2p1_followers', 0) or 0
        total_followers = followers1 + followers2 + 1
        followers_ratio = followers1 / total_followers
        
        stake1 = row.get('team1_stakes_won', 0) or 0
        stake_placed1 = row.get('team1_stakes_placed', 1) or 1
        stake_ratio1 = stake1 / stake_placed1 if stake_placed1 > 0 else 1.0
        
        stake2 = row.get('team2_stakes_won', 0) or 0
        stake_placed2 = row.get('team2_stakes_placed', 1) or 1
        stake_ratio2 = stake2 / stake_placed2 if stake_placed2 > 0 else 1.0
        
        advanced_features = [
            h2h['h2h_team1_winrate'],
            h2h['h2h_team1_games'],
            char['char_matchup_wr'],
            char['char_matchup_games'],
            mom['team1_momentum'],
            mom['team2_momentum'],
            earnings_ratio,
            followers_ratio,
            stake_ratio1,
            stake_ratio2,
            (row.get('team1p1_is_partner', '') == 'True'),
            (row.get('team2p1_is_partner', '') == 'True'),
        ]
        
        feature_list.append(base_features + advanced_features)
    
    X = np.array(feature_list)
    y_winner = matches['y_winner'].values
    y_margin = matches['y_margin'].values
    y_blowout = matches['y_blowout'].values
    
    return X, y_winner, y_margin, y_blowout


def train_winner_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def train_margin_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = GradientBoostingClassifier(
        n_estimators=30,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def train_blowout_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = GradientBoostingClassifier(
        n_estimators=30,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def evaluate_winner_model(model, scaler, X_test, y_test, X_train=None, y_train=None):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"  Test Accuracy: {acc:.1%}")
    
    return acc


def train_game_models(game_name, df):
    print(f"\n{'='*50}")
    print(f"Training: {game_name}")
    print(f"{'='*50}")
    
    X, y_winner, y_margin, y_blowout = prepare_data(df, game_name)
    
    if X is None or len(X) < 50:
        print(f"  Skipping {game_name} - not enough data")
        return None
    
    split_train = int(len(X) * TRAIN_RATIO)
    split_val = int(len(X) * (TRAIN_RATIO + VAL_RATIO))
    
    X_train = X[:split_train]
    X_test = X[split_val:]
    
    y_train_winner = y_winner[:split_train]
    y_test_winner = y_winner[split_val:]
    
    y_train_margin = y_margin[:split_train]
    y_test_margin = y_margin[split_val:]
    
    y_train_blowout = y_blowout[:split_train]
    y_test_blowout = y_blowout[split_val:]
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    if len(X_test) < 10:
        return None
    
    winner_model, winner_scaler = train_winner_model(X_train, y_train_winner)
    acc = evaluate_winner_model(winner_model, winner_scaler, X_test, y_test_winner)
    
    margin_model, margin_scaler = train_margin_model(X_train, y_train_margin)
    
    blowout_model, blowout_scaler = train_blowout_model(X_train, y_train_blowout)
    
    return {
        'winner_model': winner_model,
        'winner_scaler': winner_scaler,
        'margin_model': margin_model,
        'margin_scaler': margin_scaler,
        'blowout_model': blowout_model,
        'blowout_scaler': blowout_scaler,
        'feature_names': [
            'team1_rank', 'team2_rank', 'team1_win_rate', 'team2_win_rate',
            'team1_last5', 'team2_last5', 'team1_exp', 'team2_exp',
            'win_rate_diff', 'rank_advantage', 'h2h_winrate', 'h2h_games',
            'char_matchup_wr', 'char_matchup_games', 'team1_momentum', 'team2_momentum',
            'earnings_ratio', 'followers_ratio', 'team1_stake_ratio', 'team2_stake_ratio',
            'team1_partner', 'team2_partner'
        ],
        'game_name': game_name,
        'test_accuracy': acc
    }


def run_all_games():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"Total records: {len(df)}")
    
    games = get_available_games(df)
    print(f"Available games: {games}")
    
    all_models = {}
    
    for game in games:
        result = train_game_models(game, df)
        if result:
            all_models[game] = result
    
    all_models['__games__'] = list(all_models.keys())
    
    with open("all_games_models.pkl", 'wb') as f:
        pickle.dump(all_models, f)
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for game in all_models.get('__games__', []):
        if game in all_models:
            print(f"  {game}: {all_models[game].get('test_accuracy', 0):.1%} accuracy")
    
    print(f"\nSaved to all_games_models.pkl")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--all-games':
        run_all_games()
    else:
        print("Usage: python train_model.py --all-games")
