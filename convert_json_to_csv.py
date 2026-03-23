import json
import pandas as pd

INPUT_FILE = "1v1me_5000_events_20260319_161339.json"
OUTPUT_FILE = "Mar19_with_characters.csv"

def convert_json_to_csv():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for match in data:
        play_info = match.get('play_info') or {}
        league_info = match.get('league_info') or {}
        teams = match.get('teams', [])
        commentator = match.get('commentator') or {}

        team1 = teams[0] if len(teams) > 0 else {}
        team2 = teams[1] if len(teams) > 1 else {}

        team1_roster = team1.get('roster', {})
        team2_roster = team2.get('roster', {})
        team1_stat = team1.get('stat', {})
        team2_stat = team2.get('stat', {})

        team1_players_list = team1_roster.get('players', [])
        team2_players_list = team2_roster.get('players', [])

        team1_player = team1_players_list[0].get('user_profile', {}) if team1_players_list else {}
        team2_player = team2_players_list[0].get('user_profile', {}) if team2_players_list else {}

        team1_stats = team1_player.get('user_stats', {})
        team2_stats = team2_player.get('user_stats', {})

        team1_tag = team1.get('tag') or {}
        team2_tag = team2.get('tag') or {}

        team1_placement = team1.get('placement', -2)
        team2_placement = team2.get('placement', -2)

        winner = team1_roster.get('team_name') if team1_placement == 1 else (team2_roster.get('team_name') if team2_placement == 1 else None)
        loser = team2_roster.get('team_name') if team1_placement == 1 else (team1_roster.get('team_name') if team2_placement == 1 else None)

        team1_scores = ""
        team2_scores = ""
        if team1.get('match_round_results') and team2.get('match_round_results'):
            t1_scores = [str(r.get('score', 0)) for r in team1.get('match_round_results', [])]
            t2_scores = [str(r.get('score', 0)) for r in team2.get('match_round_results', [])]
            team1_scores = "R1: " + ", ".join(t1_scores)
            team2_scores = "R1: " + ", ".join(t2_scores)

        row = {
            'scrape_timestamp': pd.Timestamp.now().isoformat(),
            'match_id': match.get('id'),
            'type': match.get('type'),
            'state': match.get('state'),
            'start_date': match.get('start_date'),
            'end_date': match.get('end_date'),
            'prize_pool': match.get('prize_pool'),
            'entry_fee': match.get('entry_fee'),
            'completed': match.get('completed'),
            'active': match.get('active'),
            'share_url': match.get('share_url'),
            'slug': match.get('slug'),
            'game_name': play_info.get('game_name'),
            'game_mode_title': play_info.get('game_mode_title'),
            'num_of_players': play_info.get('num_of_players'),
            'num_of_rounds': play_info.get('num_of_rounds'),
            'match_format_description': play_info.get('match_format_description'),
            'console_abbreviation': play_info.get('console_abbreviation'),
            'game_id': play_info.get('game_id'),
            'game_mode_id': play_info.get('game_mode_id'),
            'league_id': match.get('league_id'),
            'season_name': league_info.get('season_name'),
            'season_state': league_info.get('season_state'),
            'season_state_title_compact': league_info.get('season_state_title_compact'),
            'commentator_name': commentator.get('name'),
            'commentator_twitch_channel': commentator.get('twitch_channel'),
            'hls_url': commentator.get('hls_url'),
            'has_staking_challenges': match.get('has_staking_challenges'),
            'has_live_staking': match.get('has_live_staking'),
            'team1_name': team1_roster.get('team_name'),
            'team1_code': team1_roster.get('code'),
            'team1_color': team1_roster.get('team_color'),
            'team1_rank': team1.get('rank'),
            'team1_placement': team1_placement,
            'team1_stakes_placed': team1_stat.get('stakes_placed'),
            'team1_stakes_won': team1_stat.get('stakes_won'),
            'team1_wins': team1_stat.get('wins'),
            'team1_losses': team1_stat.get('losses'),
            'team1_completed': team1_stat.get('completed'),
            'team1_last_five': team1_stat.get('last_five_won'),
            'team1_franchise_id': team1_roster.get('franchise_id'),
            'team1_character_tag': team1_tag.get('title'),
            'team1_character_tag_bg_color': team1_tag.get('background_color'),
            'team1_character_tag_text_color': team1_tag.get('text_color'),
            'team1p1_username': team1_player.get('username'),
            'team1p1_type': team1_players_list[0].get('type') if team1_players_list else None,
            'team1p1_followers': team1_stats.get('follower_count'),
            'team1p1_following': team1_stats.get('following_count'),
            'team1p1_total_earnings': team1_stats.get('total_earnings'),
            'team1p1_wager_earnings': team1_stats.get('total_wager_earnings'),
            'team1p1_staked_earnings': team1_stats.get('total_staked_earnings'),
            'team1p1_tournament_earnings': team1_stats.get('total_tournament_earnings'),
            'team1p1_stakeable_events': team1_stats.get('num_of_stakeable_events'),
            'team1p1_staked_events': team1_stats.get('num_of_staked_events'),
            'team1p1_is_verified': team1_player.get('is_verified'),
            'team1p1_is_partner': team1_player.get('is_partner'),
            'team1p1_is_vip': team1_player.get('is_vip'),
            'team1p1_is_employee': team1_player.get('is_employee'),
            'team1p1_region': team1_player.get('region'),
            'team1p1_activity_status': team1_player.get('activity_status'),
            'team1p1_profile_image': team1_player.get('profile_image_url'),
            'team1p2_username': None,
            'team1p2_type': None,
            'team1p2_followers': None,
            'team1p2_following': None,
            'team1p2_total_earnings': None,
            'team1p2_wager_earnings': None,
            'team1p2_staked_earnings': None,
            'team1p2_tournament_earnings': None,
            'team1p2_stakeable_events': None,
            'team1p2_staked_events': None,
            'team1p2_is_verified': None,
            'team1p2_is_partner': None,
            'team1p2_is_vip': None,
            'team1p2_is_employee': None,
            'team1p2_region': None,
            'team1p2_activity_status': None,
            'team1p2_profile_image': None,
            'team2_name': team2_roster.get('team_name'),
            'team2_code': team2_roster.get('code'),
            'team2_color': team2_roster.get('team_color'),
            'team2_rank': team2.get('rank'),
            'team2_placement': team2_placement,
            'team2_stakes_placed': team2_stat.get('stakes_placed'),
            'team2_stakes_won': team2_stat.get('stakes_won'),
            'team2_wins': team2_stat.get('wins'),
            'team2_losses': team2_stat.get('losses'),
            'team2_completed': team2_stat.get('completed'),
            'team2_last_five': team2_stat.get('last_five_won'),
            'team2_franchise_id': team2_roster.get('franchise_id'),
            'team2_character_tag': team2_tag.get('title'),
            'team2_character_tag_bg_color': team2_tag.get('background_color'),
            'team2_character_tag_text_color': team2_tag.get('text_color'),
            'team2p1_username': team2_player.get('username'),
            'team2p1_type': team2_players_list[0].get('type') if team2_players_list else None,
            'team2p1_followers': team2_stats.get('follower_count'),
            'team2p1_following': team2_stats.get('following_count'),
            'team2p1_total_earnings': team2_stats.get('total_earnings'),
            'team2p1_wager_earnings': team2_stats.get('total_wager_earnings'),
            'team2p1_staked_earnings': team2_stats.get('total_staked_earnings'),
            'team2p1_tournament_earnings': team2_stats.get('total_tournament_earnings'),
            'team2p1_stakeable_events': team2_stats.get('num_of_stakeable_events'),
            'team2p1_staked_events': team2_stats.get('num_of_staked_events'),
            'team2p1_is_verified': team2_player.get('is_verified'),
            'team2p1_is_partner': team2_player.get('is_partner'),
            'team2p1_is_vip': team2_player.get('is_vip'),
            'team2p1_is_employee': team2_player.get('is_employee'),
            'team2p1_region': team2_player.get('region'),
            'team2p1_activity_status': team2_player.get('activity_status'),
            'team2p1_profile_image': team2_player.get('profile_image_url'),
            'team2p2_username': None,
            'team2p2_type': None,
            'team2p2_followers': None,
            'team2p2_following': None,
            'team2p2_total_earnings': None,
            'team2p2_wager_earnings': None,
            'team2p2_staked_earnings': None,
            'team2p2_tournament_earnings': None,
            'team2p2_stakeable_events': None,
            'team2p2_staked_events': None,
            'team2p2_is_verified': None,
            'team2p2_is_partner': None,
            'team2p2_is_vip': None,
            'team2p2_is_employee': None,
            'team2p2_region': None,
            'team2p2_activity_status': None,
            'team2p2_profile_image': None,
            'team1_odds': None,
            'team2_odds': None,
            'team1_score': team1_scores,
            'team2_score': team2_scores,
            'winner_name': winner,
            'score_summary': f"{team1_scores.replace('R1: ', '')} - {team2_scores.replace('R1: ', '')}" if team1_scores and team2_scores else None
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Converted {len(df)} matches to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_json_to_csv()
