"""
1v1me Esports Analytics Streamlit Application
A browser-based analytics tool for 1v1.me esports data.
Uses full CSV data loaded in memory (not exposed to users).
"""

import streamlit as st
import pandas as pd
from streamlit_app.data.embedded_data import get_data, TOTAL_MATCHES, GAME_OPTIONS
from streamlit_app.utils.analysis import analyze_matchup, get_available_players, get_player_stats
from streamlit_app.utils.visualization import get_viz_options, get_visualization, plot_h2h_comparison
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
    if 'scrape_timestamp' in df.columns:
        try:
            scrape_date = pd.to_datetime(df['scrape_timestamp'].iloc[0]).strftime('%B %d, %Y')
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
            'winner_name', 'score_summary'
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
                st.metric("Total Earnings", f"${stats['earnings']:,.0f}")
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
    
    st.divider()
    
    st.subheader("📊 Quick Stats")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        team1_wins = int((df['team1_placement'] == 1).sum())
        team2_wins = int((df['team2_placement'] == 1).sum())
        total_wins = team1_wins + team2_wins
        st.metric("Team 1 Win %", f"{(team1_wins/total_wins)*100:.1f}%")
    
    with col2:
        top_game = df['game_name'].value_counts().idxmax()
        st.metric("Most Popular Game", top_game)
    
    with col3:
        top_winner = df['winner_name'].value_counts().idxmax()
        top_wins = df['winner_name'].value_counts().max()
        st.metric("Top Winner", f"{top_winner} ({top_wins})")


def show_prediction_tab():
    """Show Match Prediction tab."""
    st.title("🔮 Match Prediction")
    
    st.markdown("Get match predictions based on statistical analysis and machine learning models trained on historical match data.")
    
    df = get_data()
    
    if df is None or df.empty:
        st.error("No data loaded!")
        return
    
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
                result = predict_match(player1, player2)
                
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
    
    with st.expander("ℹ️ About the Prediction Model"):
        st.write("""
        This prediction model utilizes statistical analysis and machine learning algorithms trained on historical match data.
        
        **Features analyzed:**
        - Player rank
        - Win rate
        - Recent performance trends
        - Experience (total matches played)
        - Earnings history
        - Follower metrics
        
        **Methodology:** Random Forest classification with feature scaling. Predictions are based on historical patterns and should be used for informational purposes only.
        """)


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
        "📈 Visualizations": show_visualization_tab,
        "🔮 Predictions": show_prediction_tab
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
