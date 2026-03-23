"""
Visualization Module - uses full data from embedded_data
"""

import plotly.graph_objects as go
import plotly.express as px
from data.embedded_data import get_data


def get_viz_options():
    """Get list of visualization options."""
    return [
        "Win Distribution",
        "Game Popularity",
        "Season Performance",
        "Team Win Rates",
        "Character/Team Popularity",
        "Top Winners"
    ]


def plot_win_distribution():
    """Plot win distribution between Team 1 and Team 2."""
    df = get_data()
    
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data loaded", x=0.5, y=0.5)
        return fig
    
    team1_wins = int((df['team1_placement'] == 1).sum())
    team2_wins = int((df['team2_placement'] == 1).sum())
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Team 1', 'Team 2'],
            y=[team1_wins, team2_wins],
            marker_color=['#4CAF50', '#2196F3'],
            text=[team1_wins, team2_wins],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Win Distribution: Team 1 vs Team 2',
        xaxis_title='Team',
        yaxis_title='Number of Wins',
        template='plotly_dark',
        height=400
    )
    
    return fig


def plot_game_popularity():
    """Plot game popularity as a pie chart."""
    df = get_data()
    
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data loaded", x=0.5, y=0.5)
        return fig
    
    game_dist = df['game_name'].value_counts().head(10)
    
    fig = go.Figure(data=[go.Pie(
        labels=game_dist.index,
        values=game_dist.values,
        hole=0.4,
        textinfo='label+percent'
    )])
    
    fig.update_layout(
        title='Game Distribution',
        template='plotly_dark',
        height=450
    )
    
    return fig


def plot_season_performance():
    """Plot matches by season."""
    df = get_data()
    
    if df is None or df.empty or 'season_state_title_compact' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No season data", x=0.5, y=0.5)
        return fig
    
    season_dist = df['season_state_title_compact'].value_counts().head(20)
    
    fig = go.Figure(data=[
        go.Bar(
            x=season_dist.index,
            y=season_dist.values,
            marker_color='#58A6FF'
        )
    ])
    
    fig.update_layout(
        title='Matches by Season (Top 20)',
        xaxis_title='Season',
        yaxis_title='Number of Matches',
        template='plotly_dark',
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig


def plot_team_win_rates():
    """Plot top teams by win rate."""
    df = get_data()
    
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data loaded", x=0.5, y=0.5)
        return fig
    
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
                team_stats[team]['wins'] += row.get(wins_col, 0) or 0
                team_stats[team]['completed'] += row.get(completed_col, 0) or 0
    
    win_rates = []
    for team, stats in team_stats.items():
        if stats['completed'] > 30:
            win_rates.append({
                'team': team[:20],
                'win_rate': stats['wins']/stats['completed']*100,
                'games': stats['completed']
            })
    
    win_rates.sort(key=lambda x: x['win_rate'], reverse=True)
    win_rates = win_rates[:15]
    
    if not win_rates:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data", x=0.5, y=0.5)
        return fig
    
    teams = [t['team'] for t in win_rates]
    rates = [t['win_rate'] for t in win_rates]
    games = [t['games'] for t in win_rates]
    
    colors = ['#3FB950' if r >= 80 else '#D29922' if r >= 60 else '#F85149' for r in rates]
    
    fig = go.Figure(data=[
        go.Bar(
            x=rates,
            y=teams,
            orientation='h',
            marker_color=colors,
            text=[f'{r:.1f}% ({g})' for r, g in zip(rates, games)],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Top Teams by Win Rate (min 30 games)',
        xaxis_title='Win Rate (%)',
        yaxis_title='Team',
        template='plotly_dark',
        height=450,
        xaxis_range=[0, 110]
    )
    
    return fig


def plot_character_popularity():
    """Plot most popular characters/teams."""
    df = get_data()
    
    char_col = 'team1_character_tag' if 'team1_character_tag' in df.columns else 'team1_players'
    
    if df is None or df.empty or char_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No character data", x=0.5, y=0.5)
        return fig
    
    char_dist = df[char_col].value_counts().head(15)
    
    fig = go.Figure(data=[
        go.Bar(
            y=char_dist.index[::-1],
            x=char_dist.values[::-1],
            orientation='h',
            marker_color='#F78166',
            text=char_dist.values[::-1],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Most Popular Characters/Teams',
        xaxis_title='Times Selected',
        yaxis_title='Character/Team',
        template='plotly_dark',
        height=450
    )
    
    return fig


def plot_top_winners():
    """Plot top winners by total wins."""
    df = get_data()
    
    if df is None or df.empty or 'winner' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No winner data", x=0.5, y=0.5)
        return fig
    
    winners = df['winner'].value_counts().head(15)
    
    fig = go.Figure(data=[
        go.Bar(
            y=winners.index[::-1],
            x=winners.values[::-1],
            orientation='h',
            marker_color='#A371F7',
            text=winners.values[::-1],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Top Players by Total Wins',
        xaxis_title='Number of Wins',
        yaxis_title='Player',
        template='plotly_dark',
        height=450
    )
    
    return fig


def plot_earnings_distribution():
    """Plot earnings distribution."""
    df = get_data()
    
    if df is None or df.empty or 'team1p1_total_earnings' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No earnings data", x=0.5, y=0.5)
        return fig
    
    earnings = df['team1p1_total_earnings'].dropna()
    earnings = earnings[earnings > 0]
    
    if len(earnings) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No earnings data", x=0.5, y=0.5)
        return fig
    
    p25 = earnings.quantile(0.25)
    p50 = earnings.quantile(0.50)
    p75 = earnings.quantile(0.75)
    p95 = earnings.quantile(0.95)
    
    categories = ['0-25%', '25-50%', '50-75%', '75-95%', '95-100%']
    values = [p25, p50-p25, p75-p50, p95-p75, earnings.max()-p95]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color='#79C0FF'
        )
    ])
    
    fig.update_layout(
        title='Earnings Distribution by Percentile',
        xaxis_title='Earnings Percentile',
        yaxis_title='Earnings ($)',
        template='plotly_dark',
        height=400
    )
    
    return fig


def plot_h2h_comparison(player1_wins, player2_wins, player1, player2):
    """Plot head-to-head comparison."""
    fig = go.Figure(data=[
        go.Bar(
            x=[player1, player2],
            y=[player1_wins, player2_wins],
            marker_color=['#4CAF50', '#2196F3'],
            text=[player1_wins, player2_wins],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f'Head-to-Head: {player1} vs {player2}',
        xaxis_title='Player',
        yaxis_title='Wins',
        template='plotly_dark',
        height=350
    )
    
    return fig


def get_visualization(viz_type):
    """Get the requested visualization."""
    viz_functions = {
        "Win Distribution": plot_win_distribution,
        "Game Popularity": plot_game_popularity,
        "Season Performance": plot_season_performance,
        "Team Win Rates": plot_team_win_rates,
        "Character/Team Popularity": plot_character_popularity,
        "Top Winners": plot_top_winners
    }
    
    if viz_type in viz_functions:
        return viz_functions[viz_type]()
    else:
        return plot_win_distribution()


def get_total_matches():
    """Get total matches from data."""
    df = get_data()
    if df is None or df.empty:
        return 0
    return len(df)

import pandas as pd
