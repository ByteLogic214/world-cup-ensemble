import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def make_predictions(
    model,
    X: pd.DataFrame,
    meta_df: pd.DataFrame,
    confidence_threshold: float = 0.4
) -> pd.DataFrame:
    """
    Genera predicciones con información de confianza
    
    Args:
        model: Modelo entrenado
        X: Features para predecir
        meta_df: DataFrame con metadata de partidos
        confidence_threshold: Umbral de confianza para recomendar apuesta
        
    Returns:
        pd.DataFrame: Predicciones con metadata
    """
    logger.info("Generando predicciones...")
    
    # Obtener predicciones y probabilidades
    proba = model.predict_proba(X)
    pred = model.predict(X)
    
    # Crear DataFrame de resultados
    out = meta_df[["fixture_id", "date", "home_team", "away_team"]].copy()
    
    # Añadir predicción
    out["pred"] = pred
    
    # Añadir probabilidades por clase
    classes = model.classes_
    for i, cls in enumerate(classes):
        out[f"prob_{cls}"] = proba[:, i]
    
    # Calcular confianza (máxima probabilidad)
    out["confidence"] = proba.max(axis=1)
    
    # Calcular segunda opción
    second_best_idx = np.argsort(proba, axis=1)[:, -2]
    out["second_choice"] = classes[second_best_idx]
    out["second_prob"] = proba[np.arange(len(proba)), second_best_idx]
    
    # Calcular margen (diferencia entre primera y segunda opción)
    out["margin"] = out["confidence"] - out["second_prob"]
    
    # Recomendar apuesta basada en umbral
    out["recommend_bet"] = out["confidence"] >= confidence_threshold
    
    # Calcular valor esperado si hay odds
    if "home_odds" in meta_df.columns:
        out = out.merge(
            meta_df[["fixture_id", "home_odds", "draw_odds", "away_odds"]],
            on="fixture_id",
            how="left"
        )
        
        # Calcular valor esperado para cada opción
        out["ev_home"] = out["prob_H"] * out["home_odds"] - 1
        out["ev_draw"] = out["prob_D"] * out["draw_odds"] - 1
        out["ev_away"] = out["prob_A"] * out["away_odds"] - 1
        
        # Valor esperado de la predicción
        out["ev_prediction"] = out.apply(
            lambda row: row[f"ev_{row['pred'].lower()}"] if row['pred'] != 'D' else row['ev_draw'],
            axis=1
        )
        
        # Recomendar apuesta si EV > 0
        out["positive_ev"] = out["ev_prediction"] > 0
    
    # Ordenar por confianza
    out = out.sort_values("confidence", ascending=False)
    
    logger.info(f"Generadas {len(out)} predicciones")
    logger.info(f"Predicciones con alta confianza (>{confidence_threshold}): {out['recommend_bet'].sum()}")
    
    if "positive_ev" in out.columns:
        logger.info(f"Predicciones con EV positivo: {out['positive_ev'].sum()}")
    
    return out
