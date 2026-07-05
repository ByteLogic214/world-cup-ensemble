import json
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report

def evaluate_model(model, X_test, y_test, labels):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "log_loss": float(log_loss(y_test, proba, labels=labels)),
        "classification_report": classification_report(y_test, pred, output_dict=True)
    }
    return metrics

def save_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
