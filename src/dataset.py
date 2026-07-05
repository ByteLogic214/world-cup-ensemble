import os
import pandas as pd
from src.features import normalize_fixture_row, odds_to_implied_probs

def fixtures_to_df(fixtures_json):
    rows = [normalize_fixture_row(x) for x in fixtures_json if x["fixture"]["status"]["short"] in ["FT", "AET", "PEN"]]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df

def odds_to_df(odds_json):
    rows = []
    for event in odds_json:
        home_team = event.get("home_team")
        away_team = [x for x in event.get("away_team", [])] if isinstance(event.get("away_team"), list) else event.get("away_team")
        bookmakers = event.get("bookmakers", [])
        for bk in bookmakers:
            for market in bk.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes", [])
                prices = {o["name"]: o["price"] for o in outcomes}
                if home_team in prices and away_team in prices:
                    draw_odds = prices.get("Draw", None)
                    if draw_odds is None:
                        continue
                    h, d, a = odds_to_implied_probs(prices[home_team], draw_odds, prices[away_team])
                    rows.append({
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": event.get("commence_time"),
                        "bookmaker": bk.get("title"),
                        "home_odds": prices[home_team],
                        "draw_odds": draw_odds,
                        "away_odds": prices[away_team],
                        "p_home_odds": h,
                        "p_draw_odds": d,
                        "p_away_odds": a
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["home_team","away_team"]).groupby(["home_team","away_team"], as_index=False).first()
    return df

def save_df(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
