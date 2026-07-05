import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

LABELS = ["A", "D", "H"]

class EnsemblePredictor:
    def __init__(self, cfg):
        rs = cfg["model"]["random_state"]
        self.lr = LogisticRegression(max_iter=2000, multi_class="multinomial")
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
        self.lgbm = LGBMClassifier(
            n_estimators=cfg["model"]["n_estimators_lgbm"],
            learning_rate=cfg["model"]["learning_rate_xgb"],
            random_state=rs
        )
        voting = VotingClassifier(
            estimators=[("lr", self.lr), ("rf", self.rf), ("xgb", self.xgb), ("lgbm", self.lgbm)],
            voting="soft"
        )
        self.model = CalibratedClassifierCV(voting, method="sigmoid", cv=cfg["model"]["calibration_cv"])

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
