import requests
import pandas as pd
from datetime import datetime, timedelta

class APIFootballClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"x-apisports-key": api_key}

    def _get(self, endpoint: str, params: dict):
        r = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_fixtures_by_league_season(self, league_id: int, season: int):
        data = self._get("fixtures", {"league": league_id, "season": season})
        return data.get("response", [])

    def get_team_statistics(self, league_id: int, season: int, team_id: int):
        data = self._get("teams/statistics", {"league": league_id, "season": season, "team": team_id})
        return data.get("response", {})

class OddsAPIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    def get_h2h_odds(self, sport="soccer_fifa_world_cup", regions="eu", markets="h2h"):
        url = f"{self.base_url}/sports/{sport}/odds"
        params = {"apiKey": self.api_key, "regions": regions, "markets": markets, "oddsFormat": "decimal"}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
