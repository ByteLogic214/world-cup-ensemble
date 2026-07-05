import os
import pandas as pd
import logging
from typing import List, Dict, Optional
from pathlib import Path
from src.features import normalize_fixture_row, odds_to_implied_probs

logger = logging.getLogger(__name__)

def fixtures_to_df(fixtures_json: List[dict]) -> pd.DataFrame:
    """
    Convierte fixtures JSON a DataFrame con validación
    
    Args:
        fixtures_json: Lista de fixtures en formato JSON
        
    Returns:
        pd.DataFrame: DataFrame con fixtures normalizados
    """
    if not fixtures_json:
        logger.warning("Lista de fixtures vacía")
        return pd.DataFrame()
    
    valid_statuses = {"FT", "AET", "PEN"}
    rows = []
    
    for fixture in fixtures_json:
        try:
            # Validar estructura básica
            if not all(key in fixture for key in ["fixture", "teams", "goals", "league"]):
                logger.warning(f"Fixture con estructura incompleta: {fixture.get('fixture', {}).get('id')}")
                continue
            
            status = fixture["fixture"]["status"]["short"]
            
            if status not in valid_statuses:
                continue
            
            normalized = normalize_fixture_row(fixture)
            
            # Validar datos críticos
            if normalized["home_goals"] is None or normalized["away_goals"] is None:
                logger.warning(f"Fixture {normalized['fixture_id']} con goles nulos")
                continue
            
            rows.append(normalized)
            
        except Exception as e:
            logger.error(f"Error procesando fixture: {str(e)}")
            continue
    
    if not rows:
        logger.warning("No se pudieron procesar fixtures válidos")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Convertir fecha con manejo de errores
    try:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    except Exception as e:
        logger.error(f"Error convirtiendo fechas: {str(e)}")
        # Intentar conversión fila por fila
        df["date"] = pd.to_datetime(df["date"], utc=True, errors='coerce')
    
    # Eliminar filas con fechas inválidas
    invalid_dates = df["date"].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Eliminadas {invalid_dates} filas con fechas inválidas")
        df = df.dropna(subset=["date"])
    
    logger.info(f"Procesados {len(df)} fixtures válidos de {len(fixtures_json)} totales")
    
    return df

def odds_to_df(odds_json: List[dict]) -> pd.DataFrame:
    """
    Convierte odds JSON a DataFrame con validación mejorada
    
    Args:
        odds_json: Lista de eventos con odds
        
    Returns:
        pd.DataFrame: DataFrame con odds normalizados
    """
    if not odds_json:
        logger.warning("Lista de odds vacía")
        return pd.DataFrame()
    
    rows = []
    
    for event in odds_json:
        try:
            home_team = event.get("home_team")
            away_team = event.get("away_team")
            
            # FIX CRÍTICO: away_team es string, no lista
            if not home_team or not away_team:
                logger.warning(f"Evento sin equipos válidos: {event.get('id')}")
                continue
            
            bookmakers = event.get("bookmakers", [])
            
            if not bookmakers:
                logger.debug(f"Evento sin bookmakers: {home_team} vs {away_team}")
                continue
            
            for bookmaker in bookmakers:
                markets = bookmaker.get("markets", [])
                
                for market in markets:
                    if market.get("key") != "h2h":
                        continue
                    
                    outcomes = market.get("outcomes", [])
                    
                    if len(outcomes) != 3:
                        logger.warning(f"Market h2h sin 3 outcomes: {home_team} vs {away_team}")
                        continue
                    
                    # Crear dict de precios
                    prices = {outcome["name"]: outcome["price"] for outcome in outcomes}
                    
                    # Validar que existan todas las opciones
                    if home_team not in prices or away_team not in prices or "Draw" not in prices:
                        logger.warning(f"Odds incompletos: {home_team} vs {away_team}")
                        continue
                    
                    home_odds = prices[home_team]
                    draw_odds = prices["Draw"]
                    away_odds = prices[away_team]
                    
                    # Validar que los odds sean válidos
                    if any(odd <= 1.0 for odd in [home_odds, draw_odds, away_odds]):
                        logger.warning(f"Odds inválidos (<=1.0): {home_team} vs {away_team}")
                        continue
                    
                    try:
                        p_home, p_draw, p_away = odds_to_implied_probs(
                            home_odds, draw_odds, away_odds
                        )
                    except Exception as e:
                        logger.error(f"Error calculando probabilidades implícitas: {str(e)}")
                        continue
                    
                    rows.append({
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": event.get("commence_time"),
                        "bookmaker": bookmaker.get("title"),
                        "home_odds": home_odds,
                        "draw_odds": draw_odds,
                        "away_odds": away_odds,
                        "p_home_odds": p_home,
                        "p_draw_odds": p_draw,
                        "p_away_odds": p_away
                    })
        
        except Exception as e:
            logger.error(f"Error procesando evento de odds: {str(e)}")
            continue
    
    if not rows:
        logger.warning("No se pudieron procesar odds válidos")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Agrupar por partido y tomar el primer bookmaker (o promedio)
    df = df.sort_values(["home_team", "away_team"]).groupby(
        ["home_team", "away_team"], as_index=False
    ).first()
    
    logger.info(f"Procesados odds para {len(df)} partidos únicos")
    
    return df

def save_df(df: pd.DataFrame, path: str) -> None:
    """
    Guarda DataFrame a CSV con validación
    
    Args:
        df: DataFrame a guardar
        path: Ruta donde guardar el archivo
    """
    if df.empty:
        logger.warning(f"DataFrame vacío, no se guardará: {path}")
        return
    
    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(path, index=False, encoding='utf-8')
        
        logger.info(f"DataFrame guardado exitosamente: {path} ({len(df)} filas)")
        
    except Exception as e:
        logger.error(f"Error guardando DataFrame a {path}: {str(e)}")
        raise
