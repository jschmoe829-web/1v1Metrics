"""
1v1me Esports Analytics Streamlit Application
A browser-based analytics tool for 1v1.me esports data.
Uses full CSV data loaded in memory (not exposed to users).
"""

import streamlit as st
import pandas as pd
from streamlit_app.data.embedded_data import get_data, TOTAL_MATCHES, GAME_OPTIONS
from streamlit_app.utils.analysis import analyze_matchup, get_available_players, get_player_stats, analyze_character_matchup, get_all_characters, get_player_team_stats
from streamlit_app.utils.visualization import get_viz_options, get_visualization, plot_h2h_comparison
from streamlit_app.utils.card_generator import create_prediction_card
from streamlit_app.models.pretrained_models import predict_match, PLAYER_STATS_DATABASE


st.set_page_config(
    page_title="1v1Metrics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)


def apply_custom_styles():
    """Apply custom CSS styles."""
    st.markdown("""
        <style>
        .main {
            background-color: #0E1117;
        }
        .stApp {
            background-color: #0E1117;
        }
        .css-1d391kg {
            padding-top: 1rem;
        }
        div.stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
        }
        div.stButton > button:hover {
            background-color: #45a049;
        }
        .metric-card {
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)


def show_data_tab():
    """Show Data Management tab."""
    df = get_data()
    
    st.title("📊 Data Management")
    
    if df is None or df.empty:
        st.error("No data loaded! Please ensure the CSV file is in the correct location.")
        return
    
    total = len(df)
    team1_wins = int((df['team1_placement'] == 1).sum())
    team2_wins = int((df['team2_placement'] == 1).sum())
    
    # Get scrape date for banner
    scrape_date = "Unknown"
    date_col = None
    for col in ['end_date', 'scrape_timestamp', 'start_date']:
        if col in df.columns:
            date_col = col
            break
    if date_col:
        try:
            scrape_date = pd.to_datetime(df[date_col].iloc[0]).strftime('%B %d, %Y')
        except:
            pass
    
    st.markdown(f"""
    <div style="background-color: #262730; padding: 10px; border-radius: 5px; margin-bottom: 20px; text-align: center;">
        <strong>📊 Data Last Updated:</strong> {scrape_date}
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", f"{total:,}")
    with col2:
        st.metric("Unique Games", df['game_name'].nunique())
    with col3:
        st.metric("Team 1 Wins", f"{team1_wins:,}")
    with col4:
        st.metric("Team 2 Wins", f"{team2_wins:,}")
    
    st.divider()
    
    st.subheader("📈 Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Game Distribution")
        game_df = df['game_name'].value_counts().reset_index()
        game_df.columns = ['Game', 'Matches']
        st.dataframe(game_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.write("#### Top 15 Winners")
        winners = df['winner_name'].value_counts().head(15).reset_index()
        winners.columns = ['Player', 'Wins']
        st.dataframe(winners, hide_index=True, use_container_width=True)
    
    st.divider()
    
    with st.expander("📋 Data Preview (First 10 rows)"):
        # Only show specific columns
        cols_to_show = [
            'game_name', 'game_mode_title', 'season_state_title_compact', 
            'commentator_name', 'team1_name', 'team1_character_tag', 
            'team1p1_username', 'team1p2_username', 'team2_name', 
            'team2_character_tag', 'team2p1_username', 'team2p2_username', 
            'winner', 'score_summary'
        ]
        # Filter to only existing columns
        cols_to_show = [c for c in cols_to_show if c in df.columns]
        preview_df = df.head(10)[cols_to_show]
        st.dataframe(preview_df, use_container_width=True)


def show_h2h_tab():
    """Show Head-to-Head Analysis tab."""
    st.title("⚔️ Head-to-Head Analysis")
    
    st.markdown("Analyze historical matchups between players to see who has the edge.")
    
    df = get_data()
    
    if df is None or df.empty:
        st.error("No data loaded!")
        return
    
    # Get all unique players from the data
    all_players = set()
    for col in ['team1_name', 'team2_name', 'team1p1_username', 'team2p1_username']:
        if col in df.columns:
            all_players.update(df[col].dropna().unique())
    available_players = sorted(list(all_players))
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.selectbox(
            "Select Player/Team 1",
            available_players,
            index=0,
            key="h2h_player1"
        )
    
    with col2:
        player2 = st.selectbox(
            "Select Player/Team 2",
            available_players,
            index=min(1, len(available_players)-1),
            key="h2h_player2"
        )
    
    if st.button("Analyze Matchup", type="primary"):
        if player1 == player2:
            st.error("Please select different players!")
        else:
            analysis = analyze_matchup(player1, player2)
            
            st.divider()
            
            if analysis['has_history']:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(f"{player1} Wins", analysis['player1_wins'], f"{analysis['player1_win_rate']:.1f}%")
                with col2:
                    st.metric(f"{player2} Wins", analysis['player2_wins'], f"{analysis['player2_win_rate']:.1f}%")
                with col3:
                    st.metric("Total Matches", analysis['total_matches'])
                
                st.divider()
                
                fig = plot_h2h_comparison(
                    analysis['player1_wins'],
                    analysis['player2_wins'],
                    player1,
                    player2
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                if analysis['leader'] == "Tied":
                    st.info("🏆 The series is tied!")
                else:
                    st.success(f"🏆 **{analysis['leader']}** leads the series by {analysis['margin']} match(es)")
                
                st.divider()
                
                if analysis['matches']:
                    st.subheader("📋 Match Details")
                    
                    match_data = []
                    for m in analysis['matches']:
                        match_data.append({
                            "Date": m['date'],
                            "Team 1": m['team1'],
                            "Team 1 Char": m['team1_character'],
                            "Score": f"{m['team1_score']} - {m['team2_score']}",
                            "Team 2 Char": m['team2_character'],
                            "Team 2": m['team2'],
                            "Winner": m['winner']
                        })
                    
                    match_df = pd.DataFrame(match_data)
                    st.dataframe(match_df, use_container_width=True, hide_index=True)
                    
            else:
                st.warning(f"No historical match data found between {player1} and {player2}.")
    
    st.divider()
    
    with st.expander("📋 Player Statistics"):
        st.write("#### Select a player to view stats:")
        selected_player = st.selectbox("Player", available_players, key="stats_player")
        
        if selected_player:
            stats = get_player_stats(selected_player)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rank", stats['rank'])
                st.metric("Win Rate", f"{stats['win_rate']*100:.1f}%")
            with col2:
                st.metric("Experience (matches)", stats['experience'])
                st.metric("Recent Form", f"{stats['last5']*100:.1f}%")
            with col3:
                st.metric("Followers", f"{stats['followers']:,}")


def show_visualization_tab():
    """Show Visualization tab."""
    st.title("📈 Visualizations")
    
    st.markdown("Explore the full dataset through interactive charts.")
    
    df = get_data()
    
    if df is None or df.empty:
        st.error("No data loaded!")
        return
    
    viz_options = get_viz_options()
    
    selected_viz = st.selectbox(
        "Select Visualization Type",
        viz_options,
        key="viz_selector"
    )
    
    if st.button("Generate Visualization", type="primary"):
        with st.spinner("Generating chart..."):
            fig = get_visualization(selected_viz)
            st.plotly_chart(fig, use_container_width=True)


def show_prediction_tab():
    """Show Match Prediction tab."""
    st.title("🔮 Match Prediction")
    
    st.markdown("Get match predictions based on statistical analysis and machine learning models trained on historical match data.")
    
    df = get_data()
    
    if df is None or df.empty:
        st.error("No data loaded!")
        return
    
    # Get available games
    game_options = ["All Games"]
    if 'game_name' in df.columns:
        game_options.extend(df['game_name'].dropna().unique().tolist())
    
    # Get players with stats from the dataset
    all_players = set()
    for col in ['team1_name', 'team2_name', 'team1p1_username', 'team2p1_username']:
        if col in df.columns:
            all_players.update(df[col].dropna().unique())
    available_players = sorted(list(all_players))
    
    # Add known players from model
    for p in PLAYER_STATS_DATABASE.keys():
        if p not in available_players:
            available_players.append(p)
    available_players = sorted(available_players)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_game = st.selectbox(
            "Select Game",
            game_options,
            index=0,
            key="pred_game"
        )
        game = None if selected_game == "All Games" else selected_game
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.selectbox(
            "Select Player/Team 1",
            available_players,
            index=0,
            key="pred_player1"
        )
    
    with col2:
        player2 = st.selectbox(
            "Select Player/Team 2",
            available_players,
            index=min(1, len(available_players)-1),
            key="pred_player2"
        )
    
    # Show player stats
    p1_stats = get_player_stats(player1)
    p2_stats = get_player_stats(player2)
    
    st.caption(f"**{player1}**: Rank #{p1_stats['rank']} | Win Rate: {p1_stats['win_rate']*100:.1f}% | Games: {p1_stats['experience']}")
    st.caption(f"**{player2}**: Rank #{p2_stats['rank']} | Win Rate: {p2_stats['win_rate']*100:.1f}% | Games: {p2_stats['experience']}")
    
    st.divider()
    
    if st.button("Generate Prediction", type="primary", use_container_width=True):
        if player1 == player2:
            st.error("Please select different players!")
        else:
            with st.spinner("Running prediction model..."):
                result = predict_match(player1, player2, game=game)
                
                st.divider()
                
                col1, col2 = st.columns(2)
                
                winner = result['prediction']
                confidence = result['confidence'] * 100
                
                with col1:
                    if winner == "Player 1":
                        st.success(f"🏆 **{player1}** is predicted to win!")
                    else:
                        st.success(f"🏆 **{player2}** is predicted to win!")
                    
                    st.metric("Confidence", f"{confidence:.1f}%")
                    
                    predicted_margin = result.get('predicted_margin', 0)
                    blowout_prob = result.get('blowout_probability', 0) * 100
                    
                    score_label = "Rounds/Score"
                    if game in ['Madden NFL', 'NBA 2K', 'College Football']:
                        score_label = "Score"
                    
                    if winner == "Player 1":
                        margin_label = f"{player1} wins by ~{abs(predicted_margin):.0f}"
                    else:
                        margin_label = f"{player2} wins by ~{abs(predicted_margin):.0f}"
                    st.metric(f"Predicted Margin ({score_label})", f"~{predicted_margin:.0f}", margin_label)
                    st.metric("Blowout Probability", f"{blowout_prob:.1f}%", "≥3 difference" if blowout_prob > 50 else "<3 difference")
                
                with col2:
                    st.write("#### Win Probability")
                    p1_prob = result['player1_probability'] * 100
                    p2_prob = result['player2_probability'] * 100
                    
                    st.write(f"**{player1}**: {p1_prob:.1f}%")
                    st.progress(p1_prob / 100)
                    
                    st.write(f"**{player2}**: {p2_prob:.1f}%")
                    st.progress(p2_prob / 100)
                
                st.divider()
                
                st.subheader("📊 Key Factors")
                
                factors = result.get('factors', [])
                if factors:
                    for factor in factors:
                        st.write(f"• {factor}")
                else:
                    st.write("• Players are relatively evenly matched")
                
                st.divider()
                
                card_buffer = create_prediction_card(
                    player1, player2,
                    winner,
                    confidence,
                    predicted_margin,
                    result['player1_probability'],
                    result['player2_probability']
                )
                
                col_card1, col_card2 = st.columns([2, 1])
                with col_card1:
                    st.image(card_buffer, caption="Shareable Prediction Card", use_container_width=True)
                with col_card2:
                    st.write("#### Share This Prediction")
                    st.write("Download the card to share on social media!")
                    st.download_button(
                        label="📥 Download Card",
                        data=card_buffer,
                        file_name=f"prediction_{player1}_vs_{player2}.png",
                        mime="image/png",
                        use_container_width=True
                    )
    
    st.divider()
    
    with st.expander("ℹ️ About the Prediction Model"):
        st.write("""
        This prediction model utilizes statistical analysis and machine learning algorithms trained on historical match data.
        
        **Features analyzed:**
        - Player rank
        - Win rate
        - Recent performance trends
        - Experience (total matches played)
        - Follower metrics
        
        **Predictions include:**
        - Winner prediction with confidence percentage
        - Predicted score margin (positive = Player 1 wins by X, negative = Player 2 wins by X)
        
        **Methodology:** Random Forest classification (winner) and regression (margin) with feature scaling. Predictions are based on historical patterns and should be used for informational purposes only.
        """)


def show_character_matchup_tab():
    """Show Character Matchup Analysis tab."""
    st.title("🎭 Character Matchup Analysis")
    
    st.markdown("Analyze win rates and historical matchups between different characters/teams.")
    
    characters = get_all_characters()
    
    if not characters:
        st.error("No character data available!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        char1 = st.selectbox(
            "Select Character/Team 1",
            characters,
            index=0,
            key="char1"
        )
    
    with col2:
        char2 = st.selectbox(
            "Select Character/Team 2",
            characters,
            index=min(1, len(characters)-1),
            key="char2"
        )
    
    if st.button("Analyze Character Matchup", type="primary", use_container_width=True):
        if char1 == char2:
            st.error("Please select different characters!")
        else:
            with st.spinner("Analyzing matchups..."):
                analysis = analyze_character_matchup(char1, char2)
            
            st.divider()
            
            if analysis['has_history']:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(f"{char1} Wins", analysis['character1_wins'], f"{analysis['character1_win_rate']:.1f}%")
                with col2:
                    st.metric(f"{char2} Wins", analysis['character2_wins'], f"{analysis['character2_win_rate']:.1f}%")
                with col3:
                    st.metric("Total Matches", analysis['total_matches'])
                
                st.divider()
                
                if analysis['leader'] == "Tied":
                    st.info("🏆 The series is tied!")
                else:
                    st.success(f"🏆 **{analysis['leader']}** leads the matchup by {analysis['margin']} game(s)")
                
                st.divider()
                
                if analysis['matches']:
                    st.subheader("📋 Match History")
                    
                    match_data = []
                    for m in analysis['matches']:
                        match_data.append({
                            "Team 1": m['team1'],
                            "Team 1 Char": m['team1_character'],
                            "Score": f"{m['team1_score']} - {m['team2_score']}",
                            "Team 2 Char": m['team2_character'],
                            "Team 2": m['team2'],
                            "Winner": m['winner']
                        })
                    
                    match_df = pd.DataFrame(match_data)
                    st.dataframe(match_df, use_container_width=True, hide_index=True)
                    
            else:
                st.warning(f"No historical match data found between {char1} and {char2}.")
    
    st.divider()
    
    with st.expander("ℹ️ About Character Matchup Analysis"):
        st.write("""
        This analysis shows historical win rates between different characters/teams.
        
        **Note:** Character matchups are based on the team/character tag used in each match.
        Some matches may not have character data available.
        """)


def show_player_team_tab():
    """Show Player Team/Character Analysis tab."""
    st.title("🎮 Player Team Analysis")
    
    st.markdown("Enter a player name to see their most played team/character and win rate.")
    
    df = get_data()
    
    if df is None or df.empty:
        st.error("No data loaded!")
        return
    
    all_players = set()
    for col in ['team1_name', 'team2_name', 'team1p1_username', 'team2p1_username']:
        if col in df.columns:
            all_players.update(df[col].dropna().unique())
    available_players = sorted(list(all_players))
    
    player_name = st.selectbox(
        "Select a player",
        available_players,
        index=0,
        key="player_team_select"
    )
    
    if player_name:
        result = get_player_team_stats(player_name)
        
        if result:
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Most Played Team", result['most_played_team'])
            with col2:
                st.metric("Win Rate with Team", f"{result['win_rate']:.1f}%")
            with col3:
                st.metric("Total Games", result['total_games'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wins", result['wins'])
            with col2:
                st.metric("Losses", result['losses'])
            
            st.divider()
            st.subheader("📊 All Teams Played")
            
            team_data = []
            for team in result['all_teams']:
                team_data.append({
                    "Team": team['team'],
                    "Games": team['total'],
                    "Wins": team['wins'],
                    "Losses": team['losses'],
                    "Win Rate": f"{team['win_rate']:.1f}%"
                })
            
            team_df = pd.DataFrame(team_data)
            st.dataframe(team_df, use_container_width=True, hide_index=True)
        else:
            st.warning(f"No team/character data found for {player_name}.")


def main():
    """Main application."""
    apply_custom_styles()
    
    # Check if data loaded
    df = get_data()
    if df is None or df.empty:
        st.error("Failed to load data! Please ensure the CSV file exists.")
        return
    
    st.sidebar.title("🎮 1v1Metrics")
    st.sidebar.markdown("---")
    
    tabs = {
        "📊 Data": show_data_tab,
        "⚔️ Head-to-Head": show_h2h_tab,
        "🎭 Character Matchup": show_character_matchup_tab,
        "📈 Visualizations": show_visualization_tab,
        "🔮 Predictions": show_prediction_tab,
        "🎮 Player Team": show_player_team_tab
    }
    
    selected_tab = st.sidebar.radio(
        "Navigation",
        list(tabs.keys()),
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Data:** {len(df):,} matches loaded from CSV")
    
    tabs[selected_tab]()


if __name__ == "__main__":
    main()
