"""
Model training and backtesting script for 1v1 match prediction.
Run locally to train model, then commit the trained model for use in Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle
import os
import sys

CSV_PATH = "1v1me_Mar31.csv"
MODEL_PATH = "trained_model.pkl"
SCALER_PATH = "trained_scaler.pkl"
TRAIN_RATIO = 0.8


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


def prepare_data(df):
    """Prepare features from raw data."""
    required_cols = ['team1_wins', 'team1_losses', 'team2_wins', 'team2_losses', 
                    'team1_rank', 'team2_rank', 'team1_placement']
    
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns. Available: {df.columns.tolist()}")
        return None, None
    
    matches = df[df['team1_wins'].notna() & df['team1_losses'].notna() & 
               df['team2_wins'].notna() & df['team2_losses'].notna() &
               df['team1_rank'].notna() & df['team2_rank'].notna() &
               df['team1_placement'].notna()].copy()
    
    print(f"Found {len(matches)} matches with valid win/loss data")
    
    if len(matches) < 100:
        print("Insufficient data for training")
        return None, None
    
    matches['team1_win_rate'] = matches['team1_wins'] / (matches['team1_wins'] + matches['team1_losses'])
    matches['team2_win_rate'] = matches['team2_wins'] / (matches['team2_wins'] + matches['team2_losses'])
    matches['team1_total_games'] = matches['team1_wins'] + matches['team1_losses']
    matches['team2_total_games'] = matches['team2_wins'] + matches['team2_losses']
    matches['team1_last_five'] = matches['team1_last_five'].apply(parse_last_five)
    matches['team2_last_five'] = matches['team2_last_five'].apply(parse_last_five)
    
    matches['y'] = (matches['team1_placement'] == 1).astype(int)
    
    valid_mask = (
        matches['team1_win_rate'].notna() & matches['team2_win_rate'].notna() &
        matches['team1_total_games'].notna() & matches['team2_total_games'].notna() &
        (matches['team1_total_games'] > 0) & (matches['team2_total_games'] > 0)
    )
    matches = matches[valid_mask]
    
    print(f"After filtering valid records: {len(matches)} matches")
    
    if len(matches) < 100:
        print("Insufficient valid data for training")
        return None, None
    
    X = np.column_stack([
        matches['team1_rank'].values,
        matches['team2_rank'].values,
        matches['team1_win_rate'].values,
        matches['team2_win_rate'].values,
        matches['team1_last_five'].fillna(0.5).values,
        matches['team2_last_five'].fillna(0.5).values,
        matches['team1_total_games'].values,
        matches['team2_total_games'].values,
        (matches['team1_win_rate'] - matches['team2_win_rate']).values,
        (matches['team1_rank'] < matches['team2_rank']).astype(int).values,
    ])
    
    y = matches['y'].values
    
    return X, y


def train_model(X_train, y_train):
    """Train the prediction model."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test, model_name="Model"):
    """Evaluate model on test set."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*50}")
    print(f"Test set size: {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return accuracy_score(y_test, y_pred)


def save_model(model, scaler, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Save trained model and scaler."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


def load_model(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Load trained model and scaler."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def run_backtest():
    """Main backtesting workflow."""
    print("Loading data...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"Total records: {len(df)}")
    
    X, y = prepare_data(df)
    if X is None:
        return
    
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print("\nTraining model...")
    model, scaler = train_model(X_train, y_train)
    
    evaluate_model(model, scaler, X_test, y_test, "RandomForest")
    
    print("\n" + "="*50)
    print("Testing with GradientBoosting...")
    print("="*50)
    
    scaler2 = StandardScaler()
    X_train_scaled = scaler2.fit_transform(X_train)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        min_samples_split=5,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    
    evaluate_model(gb_model, scaler2, X_test, y_test, "GradientBoosting")
    
    save_model(model, scaler)
    
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    feature_names = [
        'team1_rank', 'team2_rank',
        'team1_win_rate', 'team2_win_rate',
        'team1_last5', 'team2_last5',
        'team1_exp', 'team2_exp',
        'win_rate_diff', 'rank_advantage'
    ]
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"{name}: {imp:.4f}")


if __name__ == "__main__":
    run_backtest()
