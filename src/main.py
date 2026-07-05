import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.settings import load_config, load_secrets
from src.api_clients import APIFootballClient, OddsAPIClient
from src.dataset import fixtures_to_df, odds_to_df, save_df
from src.features import build_match_features, merge_odds_features, make_feature_matrix
from src.model import EnsemblePredictor, LABELS
from src.evaluate import evaluate_model, save_metrics
from src.predict import make_predictions

def fetch_historical_data(cfg, secrets):
    football = APIFootballClient(cfg["api_football"]["base_url"], secrets.api_football_key)
    league_id = cfg["competition"]["world_cup_league_id"]

    all_rows = []
    current_year = pd.Timestamp.utcnow().year
    for season in range(current_year - cfg["data"]["season_years_back"], current_year + 1):
        try:
            fixtures = football.get_fixtures_by_league_season(league_id, season)
            all_rows.extend(fixtures)
        except Exception as e:
            print(f"Error season {season}: {e}")

    fixtures_df = fixtures_to_df(all_rows)
    save_df(fixtures_df, f'{cfg["data"]["save_raw_dir"]}/fixtures.csv')
    return fixtures_df

def fetch_odds_data(cfg, secrets):
    odds = OddsAPIClient(cfg["odds_api"]["base_url"], secrets.odds_api_key)
    odds_json = odds.get_h2h_odds()
    odds_df = odds_to_df(odds_json)
    save_df(odds_df, f'{cfg["data"]["save_raw_dir"]}/odds.csv')
    return odds_df

def train_pipeline():
    cfg = load_config()
    secrets = load_secrets()

    fixtures_df =
