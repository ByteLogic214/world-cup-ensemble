import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, labels):
    """
    Evalúa el modelo con métricas completas
    
    Args:
        model: Modelo entrenado
        X_test: Features de test
        y_test: Target de test
        labels: Lista de labels
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    logger.info("Evaluando modelo...")
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Métricas básicas
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    logloss = float(log_loss(y_test, y_proba, labels=labels))
    
    # Precision, Recall, F1 por clase
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calibration metrics (Brier score por clase)
    brier_scores = {}
    for i, label in enumerate(labels):
        y_true_binary = (y_test == label).astype(int)
        y_prob_class = y_proba[:, i]
        brier = np.mean((y_true_binary - y_prob_class) ** 2)
        brier_scores[label] = float(brier)
    
    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "log_loss": logloss,
        "precision_per_class": {labels[i]: float(precision[i]) for i in range(len(labels))},
        "recall_per_class": {labels[i]: float(recall[i]) for i in range(len(labels))},
        "f1_per_class": {labels[i]: float(f1[i]) for i in range(len(labels))},
        "support_per_class": {labels[i]: int(support[i]) for i in range(len(labels))},
        "confusion_matrix": cm.tolist(),
        "brier_scores": brier_scores,
        "classification_report": class_report
    }
    
    # Log resumen
    logger.info(f"\nRESULTADOS DE EVALUACIÓN:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1-Macro: {f1_macro:.4f}")
    logger.info(f"  F1-Weighted: {f1_weighted:.4f}")
    logger.info(f"  Log Loss: {logloss:.4f}")
    logger.info(f"\nPor clase:")
    for label in labels:
        logger.info(
            f"  {label}: Precision={metrics['precision_per_class'][label]:.4f}, "
            f"Recall={metrics['recall_per_class'][label]:.4f}, "
            f"F1={metrics['f1_per_class'][label]:.4f}"
        )
    
    return metrics

def save_metrics(metrics: dict, path: str) -> None:
    """Guarda métricas a JSON"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Métricas guardadas en: {path}")
        
    except Exception as e:
        logger.error(f"Error guardando métricas: {str(e)}")
        raise

def plot_confusion_matrix(
    metrics: dict,
    labels: list,
    save_path: str = None
) -> None:
    """
    Plotea la matriz de confusión
    
    Args:
        metrics: Diccionario con métricas (debe contener confusion_matrix)
        labels: Lista de labels
        save_path: Ruta donde guardar la imagen (opcional)
    """
    cm = np.array(metrics["confusion_matrix"])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Matriz de confusión guardada en: {save_path}")
    
    plt.close()
