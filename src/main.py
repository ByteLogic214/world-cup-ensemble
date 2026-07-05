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

    if not secrets.api_football_key:
        raise ValueError("Missing API_FOOTBALL_KEY")
    if not secrets.odds_api_key:
        raise ValueError("Missing ODDS_API_KEY")

    fixtures_df = fetch_historical_data(cfg, secrets)
    odds_df = fetch_odds_data(cfg, secrets)

    match_df = build_match_features(fixtures_df, windows=tuple(cfg["data"]["rolling_windows"]))
    if not odds_df.empty:
        match_df = merge_odds_features(match_df, odds_df)

    save_df(match_df, f'{cfg["data"]["save_processed_dir"]}/match_features.csv')

    X, y, feature_cols = make_feature_matrix(match_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"],
        stratify=y
    )

    predictor = EnsemblePredictor(cfg)
    predictor.fit(X_train, y_train)

    os.makedirs(cfg["data"]["save_models_dir"], exist_ok=True)
    predictor.save(f'{cfg["data"]["save_models_dir"]}/ensemble_model.joblib')

    metrics = evaluate_model(predictor.model, X_test, y_test, LABELS)
    save_metrics(metrics, f'{cfg["data"]["save_models_dir"]}/metrics.json')

    pd.DataFrame({"feature": feature_cols}).to_csv(
        f'{cfg["data"]["save_models_dir"]}/feature_columns.csv', index=False
    )

    print("Training complete")
    print(metrics)

if __name__ == "__main__":
    train_pipeline()
