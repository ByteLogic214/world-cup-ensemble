import numpy as np
import pandas as pd
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

RESULT_MAP = {"home": "H", "draw": "D", "away": "A"}

def safe_div(a: float, b: float) -> float:
    """División segura que maneja casos especiales"""
    if b in [0, None] or np.isnan(b) or b == 0:
        return 0.0
    return float(a / b)

def odds_to_implied_probs(
    home_odds: float,
    draw_odds: float,
    away_odds: float
) -> Tuple[float, float, float]:
    """
    Convierte odds decimales a probabilidades implícitas normalizadas
    
    Args:
        home_odds: Odds del equipo local
        draw_odds: Odds del empate
        away_odds: Odds del equipo visitante
        
    Returns:
        Tuple: (prob_home, prob_draw, prob_away)
    """
    try:
        # Validar odds
        if any(odd <= 1.0 for odd in [home_odds, draw_odds, away_odds]):
            logger.warning(f"Odds inválidos: H={home_odds}, D={draw_odds}, A={away_odds}")
            return (0.33, 0.33, 0.34)
        
        # Calcular inversos
        inv_home = 1.0 / home_odds
        inv_draw = 1.0 / draw_odds
        inv_away = 1.0 / away_odds
        
        # Normalizar (eliminar overround)
        total = inv_home + inv_draw + inv_away
        
        prob_home = inv_home / total
        prob_draw = inv_draw / total
        prob_away = inv_away / total
        
        return (prob_home, prob_draw, prob_away)
        
    except Exception as e:
        logger.error(f"Error calculando probabilidades implícitas: {str(e)}")
        return (0.33, 0.33, 0.34)

def normalize_fixture_row(raw: dict) -> dict:
    """
    Normaliza un fixture del API a formato estándar
    
    Args:
        raw: Fixture en formato API
        
    Returns:
        dict: Fixture normalizado
    """
    teams = raw.get("teams", {})
    goals = raw.get("goals", {})
    fixture = raw.get("fixture", {})
    league = raw.get("league", {})
    
    return {
        "fixture_id": fixture.get("id"),
        "date": fixture.get("date"),
        "league_id": league.get("id"),
        "league_name": league.get("name"),
        "season": league.get("season"),
        "home_team_id": teams.get("home", {}).get("id"),
        "home_team": teams.get("home", {}).get("name"),
        "away_team_id": teams.get("away", {}).get("id"),
        "away_team": teams.get("away", {}).get("name"),
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "neutral": 1 if fixture.get("venue", {}).get("city") is None else 0,
        "status": fixture.get("status", {}).get("short")
    }

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Añade la columna target (resultado del partido)"""
    df = df.copy()
    
    conditions = [
        df["home_goals"] > df["away_goals"],
        df["home_goals"] == df["away_goals"]
    ]
    choices = ["H", "D"]
    
    df["target"] = np.select(conditions, choices, default="A")
    
    return df

def build_long_team_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte fixtures a formato largo (una fila por equipo)
    
    Args:
        df: DataFrame con fixtures
        
    Returns:
        pd.DataFrame: Vista larga con perspectiva de equipo
    """
    # Vista local
    home = df[[
        "fixture_id", "date", "season", "league_id",
        "home_team_id", "home_team", "away_team_id", "away_team",
        "home_goals", "away_goals", "neutral", "target"
    ]].copy()
    
    home.columns = [
        "fixture_id", "date", "season", "league_id",
        "team_id", "team", "opp_team_id", "opponent",
        "goals_for", "goals_against", "neutral", "result"
    ]
    home["is_home"] = 1
    
    # Vista visitante
    away = df[[
        "fixture_id", "date", "season", "league_id",
        "away_team_id", "away_team", "home_team_id", "home_team",
        "away_goals", "home_goals", "neutral", "target"
    ]].copy()
    
    away.columns = [
        "fixture_id", "date", "season", "league_id",
        "team_id", "team", "opp_team_id", "opponent",
        "goals_for", "goals_against", "neutral", "result"
    ]
    away["is_home"] = 0
    
    # Invertir resultado para visitante
    away["result"] = away["result"].map({"H": "A", "D": "D", "A": "H"})
    
    return pd.concat([home, away], ignore_index=True)

def result_points(result: str) -> int:
    """Convierte resultado a puntos (3-1-0)"""
    if result == "H":
        return 3
    elif result == "D":
        return 1
    else:
        return 0

def rolling_team_features(
    team_df: pd.DataFrame,
    windows: Tuple[int, ...] = (3, 5, 10)
) -> pd.DataFrame:
    """
    Calcula features rolling por equipo
    
    Args:
        team_df: DataFrame con partidos de un equipo
        windows: Ventanas de rolling a calcular
        
    Returns:
        pd.DataFrame: DataFrame con features rolling
    """
    t = team_df.sort_values("date").copy()
    
    # Features base
    t["points"] = t["result"].apply(result_points)
    t["goal_diff"] = t["goals_for"] - t["goals_against"]
    t["win"] = (t["result"] == "H").astype(int)
    t["draw"] = (t["result"] == "D").astype(int)
    t["loss"] = (t["result"] == "A").astype(int)
    
    # Features rolling por ventana
    for w in windows:
        # Promedios básicos
        t[f"gf_avg_{w}"] = t["goals_for"].shift(1).rolling(w, min_periods=1).mean()
        t[f"ga_avg_{w}"] = t["goals_against"].shift(1).rolling(w, min_periods=1).mean()
        t[f"gd_avg_{w}"] = t["goal_diff"].shift(1).rolling(w, min_periods=1).mean()
        t[f"pts_avg_{w}"] = t["points"].shift(1).rolling(w, min_periods=1).mean()
        
        # Tasas
        t[f"win_rate_{w}"] = t["win"].shift(1).rolling(w, min_periods=1).mean()
        t[f"draw_rate_{w}"] = t["draw"].shift(1).rolling(w, min_periods=1).mean()
        t[f"loss_rate_{w}"] = t["loss"].shift(1).rolling(w, min_periods=1).mean()
        
        # Desviaciones estándar (consistencia)
        t[f"gf_std_{w}"] = t["goals_for"].shift(1).rolling(w, min_periods=2).std().fillna(0)
        t[f"ga_std_{w}"] = t["goals_against"].shift(1).rolling(w, min_periods=2).std().fillna(0)
        
        # Máximos y mínimos
        t[f"gf_max_{w}"] = t["goals_for"].shift(1).rolling(w, min_periods=1).max()
        t[f"ga_max_{w}"] = t["goals_against"].shift(1).rolling(w, min_periods=1).max()
        
        # Forma reciente (últimos partidos como decimal: 1.0 = victoria, 0.5 = empate, 0 = derrota)
        t[f"form_{w}"] = (t["points"].shift(1).rolling(w, min_periods=1).sum() / (w * 3))
    
    return t

def build_match_features(
    fixtures_df: pd.DataFrame,
    windows: Tuple[int, ...] = (3, 5, 10)
) -> pd.DataFrame:
    """
    Construye features de partido completas
    
    Args:
        fixtures_df: DataFrame con fixtures
        windows: Ventanas rolling a calcular
        
    Returns:
        pd.DataFrame: DataFrame con features por partido
    """
    logger.info("Construyendo features de partido...")
    
    # Añadir target
    base = add_target(fixtures_df)
    
    # Convertir a vista larga
    long_df = build_long_team_view(base)
    
    # Calcular features por equipo
    logger.info("Calculando features rolling por equipo...")
    team_features_list = []
    
    for team_id, group in long_df.groupby("team_id"):
        team_feats = rolling_team_features(group, windows=windows)
        team_features_list.append(team_feats)
    
    team_feats_df = pd.concat(team_features_list, ignore_index=True)
    
    # Separar local y visitante
    home_feats = team_feats_df[team_feats_df["is_home"] == 1].copy()
    away_feats = team_feats_df[team_feats_df["is_home"] == 0].copy()
    
    # Seleccionar columnas de features
    feature_patterns = [
        "gf_avg_", "ga_avg_", "gd_avg_", "pts_avg_",
        "win_rate_", "draw_rate_", "loss_rate_",
        "gf_std_", "ga_std_", "gf_max_", "ga_max_", "form_"
    ]
    
    keep_cols = ["fixture_id", "team_id"] + [
        c for c in team_feats_df.columns
        if any(c.startswith(p) for p in feature_patterns)
    ]
    
    home_feats = home_feats[keep_cols].copy()
    away_feats = away_feats[keep_cols].copy()
    
    # Renombrar columnas
    home_feats = home_feats.rename(
        columns={c: f"home_{c}" for c in home_feats.columns if c != "fixture_id"}
    )
    away_feats = away_feats.rename(
        columns={c: f"away_{c}" for c in away_feats.columns if c != "fixture_id"}
    )
    
    # Merge con dataset base
    logger.info("Fusionando features...")
    out = base.merge(home_feats, on="fixture_id", how="left") \
              .merge(away_feats, on="fixture_id", how="left")
    
    # Calcular features de diferencia
    logger.info("Calculando features de diferencia...")
    for w in windows:
        for prefix in ["gf_avg", "ga_avg", "gd_avg", "pts_avg", "win_rate", 
                       "draw_rate", "loss_rate", "gf_std", "ga_std", "form"]:
            out[f"diff_{prefix}_{w}"] = (
                out[f"home_{prefix}_{w}"] - out[f"away_{prefix}_{w}"]
            )
    
    # Features adicionales
    out["home_advantage"] = 1 - out["neutral"]
    
    logger.info(f"Features construidos: {len(out.columns)} columnas")
    
    return out

def merge_odds_features(
    df: pd.DataFrame,
    odds_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fusiona features de odds con el dataset principal
    
    Args:
        df: DataFrame de partidos
        odds_df: DataFrame de odds
        
    Returns:
        pd.DataFrame: DataFrame fusionado
    """
    logger.info("Fusionando features de odds...")
    
    if odds_df.empty:
        logger.warning("DataFrame de odds vacío, saltando fusión")
        return df
    
    odds_cols = [
        "home_team", "away_team", "commence_time",
        "home_odds", "draw_odds", "away_odds",
        "p_home_odds", "p_draw_odds", "p_away_odds"
    ]
    
    merged = df.merge(
        odds_df[odds_cols],
        on=["home_team", "away_team"],
        how="left"
    )
    
    # Calcular features derivados de odds
    merged["odds_home_away_ratio"] = merged["home_odds"] / merged["away_odds"].replace(0, np.nan)
    merged["odds_margin"] = (
        (1/merged["home_odds"] + 1/merged["draw_odds"] + 1/merged["away_odds"]) - 1
    )
    
    odds_merged = merged["home_odds"].notna().sum()
    logger.info(f"Odds fusionados para {odds_merged} partidos")
    
    return merged

def make_feature_matrix(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Crea la matriz de features final para modelado
    
    Args:
        df: DataFrame con todas las features
        
    Returns:
        Tuple: (X, y, feature_columns)
    """
    logger.info("Creando matriz de features...")
    
    # Columnas a excluir
    drop_cols = [
        "fixture_id", "date", "league_name", "target", "status",
        "home_team", "away_team", "home_team_id", "away_team_id",
        "season", "league_id", "home_goals", "away_goals",
        "commence_time", "neutral"
    ]
    
    # Seleccionar solo columnas numéricas
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    X = df[feature_cols].fillna(0)
    y = df["target"]
    
    # Validar
    if X.isnull().any().any():
        logger.warning("Hay valores nulos en X después del fillna")
    
    logger.info(f"Matriz de features: {X.shape}")
    logger.info(f"Distribución del target:\n{y.value_counts()}")
    
    return X, y, feature_cols
