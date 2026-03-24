import json
import csv
from datetime import datetime

INPUT_FILE = "1v1me_10000_events_20260324_112723.json"
OUTPUT_FILE = "1v1me_events.csv"

def get_nested(data, *keys, default=None):
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data

def process_player(player):
    user_profile = get_nested(player, "user_profile", default={})
    user_stats = get_nested(user_profile, "user_stats", default={})
    return {
        "username": get_nested(user_profile, "username", default=""),
        "type": get_nested(player, "type", default=""),
        "followers": get_nested(user_stats, "follower_count", default=0),
        "following": get_nested(user_stats, "following_count", default=0),
        "total_earnings": get_nested(user_stats, "total_earnings", default=0),
        "wager_earnings": get_nested(user_stats, "total_wager_earnings", default=0),
        "staked_earnings": get_nested(user_stats, "total_staked_earnings", default=0),
        "tournament_earnings": get_nested(user_stats, "total_tournament_earnings", default=0),
        "stakeable_events": get_nested(user_stats, "num_of_stakeable_events", default=0),
        "staked_events": get_nested(user_stats, "num_of_staked_events", default=0),
        "is_verified": get_nested(user_profile, "is_verified", default=False),
        "is_partner": get_nested(user_profile, "is_partner", default=False),
        "is_vip": get_nested(user_profile, "is_vip", default=False),
        "is_employee": get_nested(user_profile, "is_employee", default=False),
        "region": get_nested(user_profile, "region", default=""),
        "activity_status": get_nested(user_profile, "activity_status", default=""),
        "profile_image": get_nested(user_profile, "profile_image_url", default=""),
    }

def process_team(team, team_num):
    prefix = f"team{team_num}p"
    roster = get_nested(team, "roster", default={})
    stat = get_nested(team, "stat", default={})
    tag = get_nested(team, "tag", default={})
    
    players = get_nested(roster, "players", default=[])
    player1 = process_player(players[0]) if len(players) > 0 else {}
    player2 = process_player(players[1]) if len(players) > 1 else {}
    
    last_five = get_nested(stat, "last_five_won", default=[])
    last_five_str = "".join(["W" if x else "L" for x in last_five]) if last_five else ""
    
    return {
        f"team{team_num}_name": get_nested(roster, "team_name", default=""),
        f"team{team_num}_code": get_nested(roster, "code", default=""),
        f"team{team_num}_color": get_nested(roster, "team_color", default=""),
        f"team{team_num}_rank": get_nested(team, "rank", default=""),
        f"team{team_num}_placement": get_nested(team, "placement", default=""),
        f"team{team_num}_stakes_placed": get_nested(stat, "stakes_placed", default=0),
        f"team{team_num}_stakes_won": get_nested(stat, "stakes_won", default=0),
        f"team{team_num}_wins": get_nested(stat, "wins", default=0),
        f"team{team_num}_losses": get_nested(stat, "losses", default=0),
        f"team{team_num}_completed": get_nested(stat, "completed", default=0),
        f"team{team_num}_last_five": last_five_str,
        f"team{team_num}_franchise_id": get_nested(roster, "franchise_id", default=""),
        f"team{team_num}_character_tag": get_nested(tag, "title", default=""),
        f"team{team_num}_character_tag_bg_color": get_nested(tag, "background_color", default=""),
        f"team{team_num}_character_tag_text_color": get_nested(tag, "text_color", default=""),
        f"team{team_num}p1_username": player1.get("username", ""),
        f"team{team_num}p1_type": player1.get("type", ""),
        f"team{team_num}p1_followers": player1.get("followers", 0),
        f"team{team_num}p1_following": player1.get("following", 0),
        f"team{team_num}p1_total_earnings": player1.get("total_earnings", 0),
        f"team{team_num}p1_wager_earnings": player1.get("wager_earnings", 0),
        f"team{team_num}p1_staked_earnings": player1.get("staked_earnings", 0),
        f"team{team_num}p1_tournament_earnings": player1.get("tournament_earnings", 0),
        f"team{team_num}p1_stakeable_events": player1.get("stakeable_events", 0),
        f"team{team_num}p1_staked_events": player1.get("staked_events", 0),
        f"team{team_num}p1_is_verified": player1.get("is_verified", False),
        f"team{team_num}p1_is_partner": player1.get("is_partner", False),
        f"team{team_num}p1_is_vip": player1.get("is_vip", False),
        f"team{team_num}p1_is_employee": player1.get("is_employee", False),
        f"team{team_num}p1_region": player1.get("region", ""),
        f"team{team_num}p1_activity_status": player1.get("activity_status", ""),
        f"team{team_num}p1_profile_image": player1.get("profile_image", ""),
        f"team{team_num}p2_username": player2.get("username", ""),
        f"team{team_num}p2_type": player2.get("type", ""),
        f"team{team_num}p2_followers": player2.get("followers", 0),
        f"team{team_num}p2_following": player2.get("following", 0),
        f"team{team_num}p2_total_earnings": player2.get("total_earnings", 0),
        f"team{team_num}p2_wager_earnings": player2.get("wager_earnings", 0),
        f"team{team_num}p2_staked_earnings": player2.get("staked_earnings", 0),
        f"team{team_num}p2_tournament_earnings": player2.get("tournament_earnings", 0),
        f"team{team_num}p2_stakeable_events": player2.get("stakeable_events", 0),
        f"team{team_num}p2_staked_events": player2.get("staked_events", 0),
        f"team{team_num}p2_is_verified": player2.get("is_verified", False),
        f"team{team_num}p2_is_partner": player2.get("is_partner", False),
        f"team{team_num}p2_is_vip": player2.get("is_vip", False),
        f"team{team_num}p2_is_employee": player2.get("is_employee", False),
        f"team{team_num}p2_region": player2.get("region", ""),
        f"team{team_num}p2_activity_status": player2.get("activity_status", ""),
        f"team{team_num}p2_profile_image": player2.get("profile_image", ""),
    }

def process_match(event, scrape_timestamp):
    teams = get_nested(event, "teams", default=[])
    team1 = teams[0] if len(teams) > 0 else {}
    team2 = teams[1] if len(teams) > 1 else {}
    
    team1_data = process_team(team1, 1)
    team2_data = process_team(team2, 2)
    
    match_results = get_nested(team1, "match_round_results", default=[])
    team1_score = sum(r.get("score", 0) for r in match_results)
    match_results2 = get_nested(team2, "match_round_results", default=[])
    team2_score = sum(r.get("score", 0) for r in match_results2)
    
    winner_name = ""
    if get_nested(team1, "placement", default=0) == 1:
        winner_name = get_nested(team1, "roster", "team_name", default="")
    elif get_nested(team2, "placement", default=0) == 1:
        winner_name = get_nested(team2, "roster", "team_name", default="")
    
    score_summary = f"{team1_score} - {team2_score}"
    
    row = {
        "scrape_timestamp": scrape_timestamp,
        "match_id": get_nested(event, "id", default=""),
        "type": get_nested(event, "type", default=""),
        "state": get_nested(event, "state", default=""),
        "start_date": get_nested(event, "start_date", default=""),
        "end_date": get_nested(event, "end_date", default=""),
        "prize_pool": get_nested(event, "prize_pool", default=""),
        "entry_fee": get_nested(event, "entry_fee", default=""),
        "completed": get_nested(event, "completed", default=False),
        "active": get_nested(event, "active", default=False),
        "share_url": get_nested(event, "share_url", default=""),
        "slug": get_nested(event, "slug", default=""),
        "game_name": get_nested(event, "play_info", "game_name", default=""),
        "game_mode_title": get_nested(event, "play_info", "game_mode_title", default=""),
        "num_of_players": get_nested(event, "play_info", "num_of_players", default=""),
        "num_of_rounds": get_nested(event, "play_info", "num_of_rounds", default=""),
        "match_format_description": get_nested(event, "play_info", "match_format_description", default=""),
        "console_abbreviation": get_nested(event, "play_info", "console_abbreviation", default=""),
        "game_id": get_nested(event, "play_info", "game_id", default=""),
        "game_mode_id": get_nested(event, "play_info", "game_mode_id", default=""),
        "league_id": get_nested(event, "league_id", default=""),
        "season_name": get_nested(event, "league_info", "season_name", default=""),
        "season_state": get_nested(event, "league_info", "season_state", default=""),
        "season_state_title_compact": get_nested(event, "league_info", "season_state_title_compact", default=""),
        "commentator_name": get_nested(event, "commentator", "name", default=""),
        "commentator_twitch_channel": get_nested(event, "commentator", "twitch_channel", default=""),
        "hls_url": get_nested(event, "commentator", "hls_url", default=""),
        "has_staking_challenges": get_nested(event, "has_staking_challenges", default=False),
        "has_live_staking": get_nested(event, "has_live_staking", default=False),
        "team1_odds": "",
        "team2_odds": "",
        "team1_score": team1_score,
        "team2_score": team2_score,
        "winner_name": winner_name,
        "score_summary": score_summary,
    }
    row.update(team1_data)
    row.update(team2_data)
    return row

def main():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        events = json.load(f)
    
    scrape_timestamp = datetime.now().isoformat()
    print(f"Processing {len(events)} events...")
    
    if not events:
        print("No events found in JSON file.")
        return
    
    first_event = events[0]
    sample_row = process_match(first_event, scrape_timestamp)
    fieldnames = list(sample_row.keys())
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, event in enumerate(events):
            if i % 500 == 0:
                print(f"Processing event {i+1}/{len(events)}...")
            row = process_match(event, scrape_timestamp)
            writer.writerow(row)
    
    print(f"Done! Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()