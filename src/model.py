import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

LABELS = ["A", "D", "H"]


class EnsemblePredictor:
    def __init__(self, cfg):
        model_cfg = cfg.get("model", {})
        rs = model_cfg.get("random_state", 42)

        # Logistic Regression
        self.lr = LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            random_state=rs
        )

        # Random Forest
        self.rf = RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators_rf", 400),
            max_depth=model_cfg.get("max_depth_rf", 8),
            random_state=rs,
            n_jobs=-1,
        )

        # XGBoost
        self.xgb = XGBClassifier(
            n_estimators=model_cfg.get("n_estimators_xgb", 500),
            max_depth=model_cfg.get("max_depth_xgb", 5),
            learning_rate=model_cfg.get("learning_rate_xgb", 0.03),
            subsample=model_cfg.get("subsample_xgb", 0.9),
            colsample_bytree=model_cfg.get("colsample_bytree_xgb", 0.9),
            eval_metric="mlogloss",
            objective="multi:softprob",
            random_state=rs,
            n_jobs=-1,
        )

        # LightGBM
        self.lgbm = LGBMClassifier(
            n_estimators=model_cfg.get("n_estimators_lgbm", 500),
            max_depth=model_cfg.get("max_depth_lgbm", -1),
            learning_rate=model_cfg.get("learning_rate_lgbm", 0.03),
            num_leaves=model_cfg.get("num_leaves_lgbm", 31),
            random_state=rs,
            n_jobs=-1,
            verbose=-1,
        )

        voting = VotingClassifier(
            estimators=[
                ("lr", self.lr),
                ("rf", self.rf),
                ("xgb", self.xgb),
                ("lgbm", self.lgbm),
            ],
            voting="soft",
            n_jobs=-1,
        )

        self.model = CalibratedClassifierCV(
            estimator=voting,
            method="sigmoid",
            cv=model_cfg.get("calibration_cv", 3),
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

    @classmethod
    def load(cls, path):
        model = cls({"model": {}})
        model.model = joblib.load(path)
        return model
