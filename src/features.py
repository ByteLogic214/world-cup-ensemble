import numpy as np
import pandas as pd

RESULT_MAP = {"home": "H", "draw": "D", "away": "A"}

def safe_div(a, b):
    return a / b if b not in [0, None, np.nan] else 0.0

def odds_to_implied_probs(home_odds, draw_odds, away_odds):
    inv = np.array([1/home_odds, 1/draw_odds, 1/away_odds], dtype=float)
    probs = inv / inv.sum()
    return probs[0], probs[1], probs[2]

def normalize_fixture_row(raw):
    teams = raw["teams"]
    goals = raw["goals"]
    fixture = raw["fixture"]
    league = raw["league"]
    return {
        "fixture_id": raw["fixture"]["id"],
        "date": fixture["date"],
        "league_id": league["id"],
        "league_name": league["name"],
        "season": league["season"],
        "home_team_id": teams["home"]["id"],
        "home_team": teams["home"]["name"],
        "away_team_id": teams["away"]["id"],
        "away_team": teams["away"]["name"],
        "home_goals": goals["home"],
        "away_goals": goals["away"],
        "neutral": 1 if fixture.get("venue", {}).get("city") is None else 0,
        "status": fixture["status"]["short"]
    }

def add_target(df):
    df = df.copy()
    df["target"] = np.select(
        [df["home_goals"] > df["away_goals"], df["home_goals"] == df["away_goals"]],
        ["H", "D"],
        default="A"
    )
    return df

def build_long_team_view(df):
    home = df[[
        "fixture_id","date","season","league_id","home_team_id","home_team","away_team_id","away_team",
        "home_goals","away_goals","neutral","target"
    ]].copy()
    home.columns = [
        "fixture_id","date","season","league_id","team_id","team","opp_team_id","opponent",
        "goals_for","goals_against","neutral","result"
    ]
    home["is_home"] = 1

    away = df[[
        "fixture_id","date","season","league_id","away_team_id","away_team","home_team_id","home_team",
        "away_goals","home_goals","neutral","target"
    ]].copy()
    away.columns = [
        "fixture_id","date","season","league_id","team_id","team","opp_team_id","opponent",
        "goals_for","goals_against","neutral","result"
    ]
    away["is_home"] = 0
    away["result"] = away["result"].map({"H": "A", "D": "D", "A": "H"})
    return pd.concat([home, away], ignore_index=True)

def result_points(r):
    return 3 if r == "H" else 1 if r == "D" else 0

def rolling_team_features(team_df, windows=(3,5,10)):
    t = team_df.sort_values("date").copy()
    t["points"] = t["result"].map(result_points)
    t["goal_diff"] = t["goals_for"] - t["goals_against"]
    for w in windows:
        t[f"gf_avg_{w}"] = t["goals_for"].shift(1).rolling(w, min_periods=1).mean()
        t[f"ga_avg_{w}"] = t["goals_against"].shift(1).rolling(w, min_periods=1).mean()
        t[f"gd_avg_{w}"] = t["goal_diff"].shift(1).rolling(w, min_periods=1).mean()
        t[f"pts_avg_{w}"] = t["points"].shift(1).rolling(w, min_periods=1).mean()
        t[f"win_rate_{w}"] = (t["points"].shift(1).rolling(w, min_periods=1).apply(lambda x: np.mean(np.array(x)==3)))
    return t

def build_match_features(fixtures_df, windows=(3,5,10)):
    base = add_target(fixtures_df)
    long_df = build_long_team_view(base)
    feats = []
    for team_id, g in long_df.groupby("team_id"):
        feats.append(rolling_team_features(g, windows=windows))
    team_feats = pd.concat(feats, ignore_index=True)

    home_feats = team_feats[team_feats["is_home"] == 1].copy()
    away_feats = team_feats[team_feats["is_home"] == 0].copy()

    keep_cols = ["fixture_id","team_id"] + [c for c in team_feats.columns if any(c.startswith(p) for p in ["gf_avg_","ga_avg_","gd_avg_","pts_avg_","win_rate_"])]
    home_feats = home_feats[keep_cols].copy()
    away_feats = away_feats[keep_cols].copy()

    home_feats = home_feats.rename(columns={c: f"home_{c}" for c in home_feats.columns if c != "fixture_id"})
    away_feats = away_feats.rename(columns={c: f"away_{c}" for c in away_feats.columns if c != "fixture_id"})

    out = base.merge(home_feats, on="fixture_id", how="left").merge(away_feats, on="fixture_id", how="left")

    for w in windows:
        out[f"diff_gf_avg_{w}"] = out[f"home_gf_avg_{w}"] - out[f"away_gf_avg_{w}"]
        out[f"diff_ga_avg_{w}"] = out[f"home_ga_avg_{w}"] - out[f"away_ga_avg_{w}"]
        out[f"diff_gd_avg_{w}"] = out[f"home_gd_avg_{w}"] - out[f"away_gd_avg_{w}"]
        out[f"diff_pts_avg_{w}"] = out[f"home_pts_avg_{w}"] - out[f"away_pts_avg_{w}"]
        out[f"diff_win_rate_{w}"] = out[f"home_win_rate_{w}"] - out[f"away_win_rate_{w}"]

    return out

def merge_odds_features(df, odds_df):
    merged = df.merge(
        odds_df[["home_team","away_team","commence_time","home_odds","draw_odds","away_odds","p_home_odds","p_draw_odds","p_away_odds"]],
        left_on=["home_team","away_team"],
        right_on=["home_team","away_team"],
        how="left"
    )
    return merged

def make_feature_matrix(df):
    drop_cols = ["fixture_id","date","league_name","target","status","home_team","away_team"]
    feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype != "O"]
    X = df[feature_cols].fillna(0)
    y = df["target"]
    return X, y, feature_cols
