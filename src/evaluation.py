"""
Avaliação e explicabilidade dos modelos — Breast Cancer Wisconsin (Diagnostic)

- Avalia os 3 modelos treinados (src/modeling.py) no conjunto de TESTE, nunca
  visto durante o treino, com accuracy, precision, recall, F1-score, matriz de
  confusão e curva ROC/AUC.
- Explicabilidade:
    * Feature importance nativa (Árvore de Decisão / Random Forest) e
      coeficientes (Regressão Logística).
    * Permutation importance (model-agnostic) no conjunto de teste.
    * Explicação tipo-SHAP para a Regressão Logística: como o modelo é
      linear, a contribuição exata de cada feature para uma predição
      individual é `coef_i * (x_i - média_treino_i)` — esta é a formulação
      analítica exata do SHAP para modelos lineares (Lundberg & Lee, 2017),
      calculada aqui sem depender do pacote `shap` (indisponível para
      instalação neste ambiente sandbox, sem acesso ao índice completo do
      PyPI; ele segue listado em requirements.txt para uso local/Docker,
      onde pode ser usado via `shap.TreeExplainer` / `shap.LinearExplainer`
      para os demais modelos).

Uso:
    python src/evaluation.py
"""

from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "reports" / "figures"

sns.set_theme(style="whitegrid")

# target: 0 = maligno (classe positiva de interesse clínico), 1 = benigno
POS_LABEL = 0
CLASS_NAMES = ["malignant (0)", "benign (1)"]

MODEL_SPECS = {
    "logistic_regression": "scaled",
    "decision_tree": "raw",
    "random_forest": "raw",
}


def load_test_data():
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    X_test_scaled = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_train_scaled = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    return X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test


def load_models():
    return {name: joblib.load(MODELS_DIR / f"{name}.joblib") for name in MODEL_SPECS}


def evaluate_models(models, X_test, X_test_scaled, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        X_eval = X_test_scaled if MODEL_SPECS[name] == "scaled" else X_test
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, POS_LABEL]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=POS_LABEL)
        rec = recall_score(y_test, y_pred, pos_label=POS_LABEL)
        f1 = f1_score(y_test, y_pred, pos_label=POS_LABEL)
        auc = roc_auc_score((y_test == POS_LABEL).astype(int), y_proba)

        rows.append({
            "model": name, "accuracy": acc,
            "precision_malignant": prec, "recall_malignant": rec,
            "f1_malignant": f1, "roc_auc": auc,
        })

        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

        cm = confusion_matrix(y_test, y_pred)
        fn = cm[0, 1]  # malignos classificados como benignos (o pior erro clínico)
        print(f"Falsos negativos (maligno predito como benigno): {fn}")

    return pd.DataFrame(rows).set_index("model").round(4)


def plot_confusion_matrices(models, X_test, X_test_scaled, y_test) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, model) in zip(axes, models.items()):
        X_eval = X_test_scaled if MODEL_SPECS[name] == "scaled" else X_test
        y_pred = model.predict(X_eval)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(models, X_test, X_test_scaled, y_test) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    y_true_bin = (y_test == POS_LABEL).astype(int)
    for name, model in models.items():
        X_eval = X_test_scaled if MODEL_SPECS[name] == "scaled" else X_test
        y_proba = model.predict_proba(X_eval)[:, POS_LABEL]
        RocCurveDisplay.from_predictions(y_true_bin, y_proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aleatório")
    ax.set_title("Curva ROC — classe 'maligno' como positiva")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_roc_curves.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(models, X_train, X_train_scaled) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Regressão Logística: magnitude dos coeficientes
    log_reg = models["logistic_regression"]
    coefs = pd.Series(log_reg.coef_[0], index=X_train_scaled.columns)
    coefs.reindex(coefs.abs().sort_values(ascending=True).index).tail(15).plot(
        kind="barh", ax=axes[0], color="#4C72B0"
    )
    axes[0].set_title("Regressão Logística\n(coeficientes, top 15)")

    # Árvore de Decisão / Random Forest: feature_importances_
    for ax, name in zip(axes[1:], ["decision_tree", "random_forest"]):
        model = models[name]
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        importances.sort_values().tail(15).plot(kind="barh", ax=ax, color="#55A868")
        ax.set_title(f"{name}\n(feature importance, top 15)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_feature_importance.png", dpi=150)
    plt.close(fig)


def plot_permutation_importance(models, X_test, X_test_scaled, y_test) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, (name, model) in zip(axes, models.items()):
        X_eval = X_test_scaled if MODEL_SPECS[name] == "scaled" else X_test
        result = permutation_importance(
            model, X_eval, y_test, n_repeats=20, random_state=42, scoring="f1_macro"
        )
        importances = pd.Series(result.importances_mean, index=X_eval.columns)
        importances.sort_values().tail(15).plot(kind="barh", ax=ax, color="#C44E52")
        ax.set_title(f"{name}\n(permutation importance, top 15)")
    fig.suptitle(
        "Permutation importance (model-agnostic, medida no conjunto de teste)",
        y=1.03, fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def linear_shap_explanation(model, X_train_scaled, X_test_scaled, y_test, n_examples=4):
    """Explicação exata tipo-SHAP para um modelo linear (Regressão Logística).

    Para um modelo linear f(x) = intercept + sum(coef_i * x_i), o valor SHAP
    exato de cada feature (assumindo independência entre elas) é:
        phi_i = coef_i * (x_i - E[x_i])
    onde E[x_i] é a média da feature no conjunto de treino (o "baseline").
    A soma de todos os phi_i mais o intercepto reproduz exatamente o log-odds
    predito pelo modelo. Esta é a base do `shap.LinearExplainer`.

    IMPORTANTE sobre o sinal: no scikit-learn, `model.classes_` = [0, 1] e
    `coef_` corresponde ao log-odds da classe 1 (benigno), não da classe 0
    (maligno). Como o resto deste script trata 'maligno' (0) como a classe
    positiva de interesse clínico, invertemos o sinal aqui: phi_i =
    -coef_i * (x_i - E[x_i]), de forma que valores POSITIVOS empurrem a
    predição em direção a 'maligno' e valores NEGATIVOS em direção a
    'benigno' — consistente com o restante da análise.
    """
    baseline = X_train_scaled.mean()
    # Sinal invertido: ver nota acima — phi>0 empurra para "maligno" (classe 0).
    coef = -pd.Series(model.coef_[0], index=X_train_scaled.columns)

    # Escolhe exemplos: os 2 casos malignos e 2 benignos com maior confiança
    proba = model.predict_proba(X_test_scaled)[:, POS_LABEL]
    idx_malignant = y_test[y_test == POS_LABEL].index
    idx_benign = y_test[y_test == 1].index
    chosen = list(pd.Series(proba, index=y_test.index).loc[idx_malignant].sort_values(ascending=False).head(2).index) + \
             list(pd.Series(proba, index=y_test.index).loc[idx_benign].sort_values(ascending=True).head(2).index)

    explanations = {}
    for i in chosen:
        x = X_test_scaled.loc[i]
        contributions = coef * (x - baseline)
        top_contributions = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(8)
        explanations[i] = {
            "true_label": CLASS_NAMES[0 if y_test.loc[i] == 0 else 1],
            "predicted_proba_malignant": float(proba[list(y_test.index).index(i)]) if i in y_test.index else None,
            "top_feature_contributions": top_contributions,
        }
    return explanations


def print_shap_like_explanations(explanations) -> None:
    print("\n" + "=" * 70)
    print("EXPLICABILIDADE (SHAP-like) — Regressão Logística")
    print("(contribuição de cada feature para o log-odds de 'maligno';")
    print(" positivo empurra a predição em direção a 'maligno')")
    print("=" * 70)
    for idx, info in explanations.items():
        print(f"\nCaso #{idx} | rótulo real: {info['true_label']}")
        print(info["top_feature_contributions"].round(3).to_string())


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test = load_test_data()
    models = load_models()

    metrics_df = evaluate_models(models, X_test, X_test_scaled, y_test)
    print("\n" + "=" * 70)
    print("RESUMO COMPARATIVO (conjunto de TESTE)")
    print("=" * 70)
    print(metrics_df.to_string())
    metrics_df.to_csv(BASE_DIR / "reports" / "model_comparison.csv")

    plot_confusion_matrices(models, X_test, X_test_scaled, y_test)
    plot_roc_curves(models, X_test, X_test_scaled, y_test)
    plot_feature_importance(models, X_train, X_train_scaled)
    plot_permutation_importance(models, X_test, X_test_scaled, y_test)

    explanations = linear_shap_explanation(
        models["logistic_regression"], X_train_scaled, X_test_scaled, y_test
    )
    print_shap_like_explanations(explanations)

    print(f"\nFiguras e métricas salvas em: {FIG_DIR} e reports/model_comparison.csv")


if __name__ == "__main__":
    main()
