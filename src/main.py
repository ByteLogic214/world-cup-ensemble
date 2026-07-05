import os
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.settings import load_config, load_secrets, setup_logging
from src.api_clients import APIFootballClient, OddsAPIClient
from src.dataset import fixtures_to_df, odds_to_df, save_df
from src.features import build_match_features, merge_odds_features, make_feature_matrix
from src.model import EnsemblePredictor, LABELS
from src.evaluate import evaluate_model, save_metrics
from src.predict import make_predictions

logger = logging.getLogger(__name__)

def create_directories(cfg: dict) -> None:
    """Crea todos los directorios necesarios"""
    directories = [
        cfg["data"]["save_raw_dir"],
        cfg["data"]["save_processed_dir"],
        cfg["data"]["save_models_dir"],
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio creado/verificado: {directory}")

def fetch_historical_data(cfg: dict, secrets) -> pd.DataFrame:
    """
    Descarga datos históricos de fixtures con manejo robusto de errores
    
    Args:
        cfg: Configuración del sistema
        secrets: Credenciales de API
        
    Returns:
        pd.DataFrame: DataFrame con fixtures históricos
    """
    logger.info("Iniciando descarga de datos históricos")
    
    football = APIFootballClient(
        cfg["api_football"]["base_url"],
        secrets.api_football_key
    )
    
    league_id = cfg["competition"]["world_cup_league_id"]
    current_year = pd.Timestamp.utcnow().year
    start_year = current_year - cfg["data"]["season_years_back"]
    
    all_fixtures = []
    successful_seasons = 0
    failed_seasons = []
    
    for season in range(start_year, current_year + 1):
        try:
            logger.info(f"Descargando temporada {season}...")
            fixtures = football.get_fixtures_by_league_season(league_id, season)
            
            if fixtures:
                all_fixtures.extend(fixtures)
                successful_seasons += 1
                logger.info(f"Temporada {season}: {len(fixtures)} fixtures descargados")
            else:
                logger.warning(f"Temporada {season}: Sin fixtures disponibles")
                failed_seasons.append(season)
                
        except Exception as e:
            logger.error(f"Error en temporada {season}: {str(e)}")
            failed_seasons.append(season)
            continue
    
    logger.info(
        f"Descarga completada: {successful_seasons} temporadas exitosas, "
        f"{len(failed_seasons)} fallidas"
    )
    
    if failed_seasons:
        logger.warning(f"Temporadas fallidas: {failed_seasons}")
    
    if not all_fixtures:
        raise ValueError("No se pudieron descargar fixtures de ninguna temporada")
    
    fixtures_df = fixtures_to_df(all_fixtures)
    
    if fixtures_df.empty:
        raise ValueError("No se pudieron procesar fixtures válidos")
    
    save_path = f'{cfg["data"]["save_raw_dir"]}/fixtures.csv'
    save_df(fixtures_df, save_path)
    
    logger.info(f"Total de fixtures procesados: {len(fixtures_df)}")
    
    return fixtures_df

def fetch_odds_data(cfg: dict, secrets) -> pd.DataFrame:
    """
    Descarga datos de odds actuales
    
    Args:
        cfg: Configuración del sistema
        secrets: Credenciales de API
        
    Returns:
        pd.DataFrame: DataFrame con odds
    """
    logger.info("Iniciando descarga de odds")
    
    try:
        odds_client = OddsAPIClient(
            cfg["odds_api"]["base_url"],
            secrets.odds_api_key
        )
        
        odds_json = odds_client.get_h2h_odds()
        
        if not odds_json:
            logger.warning("No se obtuvieron odds")
            return pd.DataFrame()
        
        odds_df = odds_to_df(odds_json)
        
        if not odds_df.empty:
            save_path = f'{cfg["data"]["save_raw_dir"]}/odds.csv'
            save_df(odds_df, save_path)
            logger.info(f"Odds procesados: {len(odds_df)} partidos")
        else:
            logger.warning("DataFrame de odds vacío después del procesamiento")
        
        return odds_df
        
    except Exception as e:
        logger.error(f"Error descargando odds: {str(e)}")
        return pd.DataFrame()

def train_pipeline():
    """Pipeline principal de entrenamiento mejorado"""
    
    # Configurar logging
    setup_logging()
    logger.info("="*80)
    logger.info("INICIANDO PIPELINE DE ENTRENAMIENTO")
    logger.info("="*80)
    
    try:
        # Cargar configuración y secretos
        logger.info("Cargando configuración...")
        cfg = load_config()
        secrets = load_secrets()
        
        # Crear directorios
        create_directories(cfg)
        
        # Descargar datos
        logger.info("\n" + "="*80)
        logger.info("FASE 1: DESCARGA DE DATOS")
        logger.info("="*80)
        
        fixtures_df = fetch_historical_data(cfg, secrets)
        odds_df = fetch_odds_data(cfg, secrets)
        
        # Validar datos mínimos
        if len(fixtures_df) < 50:
            raise ValueError(f"Datos insuficientes: solo {len(fixtures_df)} fixtures")
        
        # Construir features
        logger.info("\n" + "="*80)
        logger.info("FASE 2: INGENIERÍA DE FEATURES")
        logger.info("="*80)
        
        windows = tuple(cfg["data"]["rolling_windows"])
        logger.info(f"Ventanas de rolling: {windows}")
        
        match_df = build_match_features(fixtures_df, windows=windows)
        
        if not odds_df.empty:
            logger.info("Fusionando features de odds...")
            match_df = merge_odds_features(match_df, odds_df)
        else:
            logger.warning("Sin datos de odds para fusionar")
        
        # Guardar features procesados
        processed_path = f'{cfg["data"]["save_processed_dir"]}/match_features.csv'
        save_df(match_df, processed_path)
        
        # Crear matriz de features
        logger.info("Creando matriz de features...")
        X, y, feature_cols = make_feature_matrix(match_df)
        
        logger.info(f"Shape de X: {X.shape}")
        logger.info(f"Distribución de clases:\n{y.value_counts()}")
        logger.info(f"Número de features: {len(feature_cols)}")
        
        # Validar datos
        if X.shape[0] < 100:
            raise ValueError(f"Datos insuficientes para entrenamiento: {X.shape[0]} muestras")
        
        # Split train/test
        logger.info("\n" + "="*80)
        logger.info("FASE 3: PREPARACIÓN DE DATOS")
        logger.info("="*80)
        
        test_size = cfg["model"]["test_size"]
        random_state = cfg["model"]["random_state"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Train size: {len(X_train)} ({(1-test_size)*100:.1f}%)")
        logger.info(f"Test size: {len(X_test)} ({test_size*100:.1f}%)")
        logger.info(f"Train class distribution:\n{y_train.value_counts()}")
        logger.info(f"Test class distribution:\n{y_test.value_counts()}")
        
        # Entrenar modelo
        logger.info("\n" + "="*80)
        logger.info("FASE 4: ENTRENAMIENTO DEL MODELO")
        logger.info("="*80)
        
        predictor = EnsemblePredictor(cfg)
        predictor.fit(X_train, y_train)
        
        # Guardar modelo
        models_dir = cfg["data"]["save_models_dir"]
        model_path = f'{models_dir}/ensemble_model.joblib'
        predictor.save(model_path)
        
        # Evaluar modelo
        logger.info("\n" + "="*80)
        logger.info("FASE 5: EVALUACIÓN DEL MODELO")
        logger.info("="*80)
        
        metrics = evaluate_model(predictor.model, X_test, y_test, LABELS)
        
        # Guardar métricas
        metrics_path = f'{models_dir}/metrics.json'
        save_metrics(metrics, metrics_path)
        
        # Guardar feature columns
        feature_df = pd.DataFrame({"feature": feature_cols})
        feature_path = f'{models_dir}/feature_columns.csv'
        feature_df.to_csv(feature_path, index=False)
        logger.info(f"Feature columns guardados en: {feature_path}")
        
        # Obtener y guardar feature importance
        try:
            importance = predictor.get_feature_importance(feature_cols)
            importance_path = f'{models_dir}/feature_importance.json'
            
            import json
            with open(importance_path, 'w') as f:
                # Convertir a formato serializable
                serializable_importance = {
                    model: {k: float(v) for k, v in features.items()}
                    for model, features in importance.items()
                }
                json.dump(serializable_importance, f, indent=2)
            
            logger.info(f"Feature importance guardado en: {importance_path}")
        except Exception as e:
            logger.warning(f"No se pudo guardar feature importance: {str(e)}")
        
        # Resumen final
        logger.info("\n" + "="*80)
        logger.info("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        logger.info(f"\nMétricas finales:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")
        
        logger.info(f"\nArchivos generados:")
        logger.info(f"  - Modelo: {model_path}")
        logger.info(f"  - Métricas: {metrics_path}")
        logger.info(f"  - Features: {feature_path}")
        
        return predictor, metrics
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error("ERROR CRÍTICO EN EL PIPELINE")
        logger.error(f"{'='*80}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train_pipeline()
