import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

LABELS = ["A", "D", "H"]

class EnsemblePredictor:
    def __init__(self, cfg):
        rs = cfg["model"]["random_state"]

        self.lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                random_state=rs
            ))
        ])

        self.rf = RandomForestClassifier(
            n_estimators=cfg["model"]["n_estimators_rf"],
            max_depth=cfg["model"]["max_depth_rf"],
            random_state=rs
        )

        self.xgb = XGBClassifier(
            n_estimators=cfg["model"]["n_estimators_xgb"],
            max_depth=cfg["model"]["max_depth_xgb"],
            learning_rate=cfg["model"]["learning_rate_xgb"],
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=rs
        )

        # SECCIÓN CORREGIDA: Ajuste de parámetros para evitar el error de "hojas"
        self.lgbm = LGBMClassifier(
            n_estimators=cfg["model"]["n_estimators_lgbm"],
            learning_rate=cfg["model"]["learning_rate_xgb"],
            random_state=rs,
            min_child_samples=5,   # Reduce el número mínimo de datos por hoja
            min_child_weight=0.001 # Reduce la exigencia de peso en los nodos
        )

        voting = VotingClassifier(
            estimators=[
                ("lr", self.lr),
                ("rf", self.rf),
                ("xgb", self.xgb),
                ("lgbm", self.lgbm),
            ],
            voting="soft"
        )

        self.model = CalibratedClassifierCV(
            voting,
            method="sigmoid",
            cv=cfg["model"]["calibration_cv"]
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self
