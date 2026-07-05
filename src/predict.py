import pandas as pd

def make_predictions(model, X, meta_df):
    proba = model.predict_proba(X)
    pred = model.predict(X)
    out = meta_df[["fixture_id","date","home_team","away_team"]].copy()
    out["pred"] = pred
    out["prob_A"] = proba[:, list(model.classes_).index("A")]
    out["prob_D"] = proba[:, list(model.classes_).index("D")]
    out["prob_H"] = proba[:, list(model.classes_).index("H")]
    out["confidence"] = out[["prob_A","prob_D","prob_H"]].max(axis=1)
    return out.sort_values("confidence", ascending=False)
