import joblib
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

LABELS = ["A", "D", "H"]

class EnsemblePredictor:
    """
    Predictor ensemble mejorado con early stopping y validación
    """
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        rs = cfg["model"]["random_state"]
        
        # Pipeline de Regresión Logística con escalado
        self.lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                class_weight="balanced",  # Maneja desbalance de clases
                random_state=rs,
                n_jobs=-1
            ))
        ])
        
        # Random Forest con parámetros optimizados
        self.rf = RandomForestClassifier(
            n_estimators=cfg["model"].get("n_estimators_rf", 200),
            max_depth=cfg["model"].get("max_depth_rf", 15),
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=rs,
            n_jobs=-1,
            verbose=0
        )
        
        # XGBoost con early stopping
        self.xgb = XGBClassifier(
            n_estimators=cfg["model"].get("n_estimators_xgb", 500),
            max_depth=cfg["model"].get("max_depth_xgb", 6),
            learning_rate=cfg["model"].get("learning_rate_xgb", 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=3,
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            random_state=rs,
            n_jobs=-1,
            verbosity=0
        )
        
        # LightGBM con early stopping
        self.lgbm = LGBMClassifier(
            n_estimators=cfg["model"].get("n_estimators_lgbm", 500),
            learning_rate=cfg["model"].get("learning_rate_lgbm", 0.05),
            max_depth=cfg["model"].get("max_depth_lgbm", 6),
            num_leaves=31,
            min_child_samples=20,
            min_child_weight=0.001,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            early_stopping_rounds=50,
            random_state=rs,
            n_jobs=-1,
            verbose=-1
        )
        
        # Voting Classifier (soft voting para probabilidades)
        self.voting = VotingClassifier(
            estimators=[
                ("lr", self.lr),
                ("rf", self.rf),
                ("xgb", self.xgb),
                ("lgbm", self.lgbm),
            ],
            voting="soft",
            n_jobs=-1
        )
        
        # Calibración final
        self.model = CalibratedClassifierCV(
            self.voting,
            method="sigmoid",
            cv=cfg["model"].get("calibration_cv", 5),
            n_jobs=-1
        )
        
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Entrena el modelo ensemble con early stopping para XGB y LGBM
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            
        Returns:
            self: Modelo entrenado
        """
        logger.info("Iniciando entrenamiento del ensemble")
        
        # Validación de entrada
        if len(X) != len(y):
            raise ValueError("X e y deben tener la misma longitud")
        
        if len(X) < 100:
            logger.warning("Dataset muy pequeño para entrenamiento robusto")
        
        # Split interno para early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.15,
            random_state=self.cfg["model"]["random_state"],
            stratify=y
        )
        
        logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
        
        # Entrenar modelos base individualmente para poder usar early stopping
        logger.info("Entrenando Logistic Regression...")
        self.lr.fit(X_train, y_train)
        
        logger.info("Entrenando Random Forest...")
        self.rf.fit(X_train, y_train)
        
        logger.info("Entrenando XGBoost con early stopping...")
        self.xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        logger.info(f"XGBoost mejor iteración: {self.xgb.best_iteration}")
        
        logger.info("Entrenando LightGBM con early stopping...")
        self.lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[logging.LightGBMCallback()] if logger.level == logging.DEBUG else None
        )
        logger.info(f"LightGBM mejor iteración: {self.lgbm.best_iteration_}")
        
        # Ahora entrenar el voting ensemble con todos los datos
        # Los modelos ya están entrenados, el voting solo aprende los pesos
        logger.info("Entrenando ensemble completo...")
        
        # Reentrenar con todos los datos
        self.voting.fit(X, y)
        
        # Calibrar el modelo final
        logger.info("Calibrando probabilidades...")
        self.model.fit(X, y)
        
        self.is_fitted = True
        logger.info("Entrenamiento completado exitosamente")
        
        return self
    
    def predict(self, X) -> np.ndarray:
        """Predice las clases"""
        if not self.is_fitted and not hasattr(self.model, 'classes_'):
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predice las probabilidades por clase"""
        if not self.is_fitted and not hasattr(self.model, 'classes_'):
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        return self.model.predict_proba(X)
    
    def save(self, path: str) -> None:
        """Guarda el modelo entrenado"""
        try:
            joblib.dump(self.model, path)
            logger.info(f"Modelo guardado exitosamente en: {path}")
        except Exception as e:
            logger.error(f"Error guardando modelo: {str(e)}")
            raise
    
    def load(self, path: str):
        """Carga un modelo pre-entrenado"""
        try:
            self.model = joblib.load(path)
            self.is_fitted = True
            logger.info(f"Modelo cargado exitosamente desde: {path}")
            return self
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            raise
    
    def get_feature_importance(self, feature_names: list) -> dict:
        """
        Obtiene la importancia de features de los modelos base
        
        Returns:
            dict: Diccionario con importancias por modelo
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")
        
        importance_dict = {}
        
        # Random Forest
        if hasattr(self.rf, 'feature_importances_'):
            importance_dict['rf'] = dict(zip(feature_names, self.rf.feature_importances_))
        
        # XGBoost
        if hasattr(self.xgb, 'feature_importances_'):
            importance_dict['xgb'] = dict(zip(feature_names, self.xgb.feature_importances_))
        
        # LightGBM
        if hasattr(self.lgbm, 'feature_importances_'):
            importance_dict['lgbm'] = dict(zip(feature_names, self.lgbm.feature_importances_))
        
        return importance_dict
