"""
Modelagem — Breast Cancer Wisconsin (Diagnostic)

Treina três técnicas de classificação (mais que o mínimo de duas exigido):
    1. Regressão Logística  (usa features escalonadas)
    2. Árvore de Decisão    (usa features não-escalonadas — invariante à escala)
    3. Random Forest        (usa features não-escalonadas — baseline mais forte)

O conjunto de treino/teste já foi separado em src/preprocessing.py — este
script apenas carrega os artefatos gerados por ele.

Uso:
    python src/modeling.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

RANDOM_STATE = 42


def load_processed():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    X_train_scaled = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
    X_test_scaled = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def train_models(X_train, X_train_scaled, y_train):
    models = {}

    # 1) Regressão Logística — modelo linear, interpretável, serve de baseline
    #    clínico (coeficientes indicam direção/força do efeito de cada feature).
    #    Usa dados escalonados: essencial para regularização/convergência.
    log_reg = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_train)
    models["logistic_regression"] = (log_reg, "scaled")

    # 2) Árvore de Decisão — não-linear, captura interações entre features,
    #    fácil de visualizar/explicar para profissionais não-técnicos.
    #    Profundidade limitada para reduzir overfitting.
    tree = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    models["decision_tree"] = (tree, "raw")

    # 3) Random Forest — ensemble de árvores, geralmente mais robusto e com
    #    melhor generalização; usado como baseline mais forte de comparação.
    forest = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
    )
    forest.fit(X_train, y_train)
    models["random_forest"] = (forest, "raw")

    return models


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_processed()

    models = train_models(X_train, X_train_scaled, y_train)

    for name, (model, feature_type) in models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        X_eval = X_train_scaled if feature_type == "scaled" else X_train
        train_acc = model.score(X_eval, y_train)
        print(f"{name:22s} | tipo de features: {feature_type:7s} | "
              f"acurácia (treino): {train_acc:.4f}")

    print(f"\nModelos treinados e salvos em: {MODELS_DIR}")
    print("Avaliação detalhada no conjunto de TESTE: ver src/evaluation.py")


if __name__ == "__main__":
    main()
