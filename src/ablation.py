"""
Ablação das decisões de pré-processamento — Breast Cancer Wisconsin

MOTIVAÇÃO
---------
O conjunto de teste tem 114 amostras: 2 ou 3 acertos a mais movem as métricas
em ~2 pontos percentuais. Julgar uma decisão de pré-processamento por um único
split é, portanto, julgar em cima de ruído.

Este script mede o efeito isolado das duas decisões tomadas em
`src/preprocessing.py` e `src/modeling.py`:

    1. remover as 8 features redundantes (30 -> 22 features);
    2. usar class_weight='balanced' nos classificadores.

As 4 combinações são avaliadas com validação cruzada estratificada REPETIDA
(10 folds x 3 repetições = 30 medições por configuração), reportando média e
desvio entre folds. O desvio é a informação decisiva: uma diferença de médias
menor que ele não é distinguível do sorteio das amostras.

O escalonamento entra como primeira etapa de um Pipeline, e não aplicado uma
vez sobre todo o X: dentro do Pipeline o scaler é reajustado a cada fold,
usando apenas os dados de treino daquele fold. Ajustá-lo antes faria o fold de
validação influenciar a média/desvio da transformação — vazamento que infla a
métrica.

Uso:
    python src/ablation.py
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from preprocessing import build_features_target, drop_redundant_features, load_and_clean

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
N_SPLITS = 10
N_REPEATS = 3
POS_LABEL = 0  # 'maligno' — a classe de interesse clínico

# maligno é a classe 0, e não a 1 (padrão do sklearn), então precision/recall/f1
# precisam de pos_label explícito.
SCORING = {
    "recall_mal": make_scorer(recall_score, pos_label=POS_LABEL),
    "precision_mal": make_scorer(precision_score, pos_label=POS_LABEL),
    "f1_mal": make_scorer(f1_score, pos_label=POS_LABEL),
    "accuracy": "accuracy",
}


def montar_modelos(class_weight):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=5000, class_weight=class_weight, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, class_weight=class_weight, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight=class_weight,
            random_state=RANDOM_STATE, n_jobs=-1),
    }


def avaliar(X, y, nome_features: str) -> list[dict]:
    # cv criado uma vez e reutilizado: todas as configurações veem exatamente
    # os mesmos folds, o que torna a comparação pareada válida.
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )

    linhas = []
    for class_weight in [None, "balanced"]:
        for nome_modelo, clf in montar_modelos(class_weight).items():
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            r = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
            linhas.append({
                "modelo": nome_modelo,
                "features": nome_features,
                "class_weight": class_weight or "none",
                **{m: round(r[f"test_{m}"].mean(), 4) for m in SCORING},
                "recall_std": round(r["test_recall_mal"].std(), 4),
            })
    return linhas


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()
    X_completo, y = build_features_target(df)
    df_reduzido, _ = drop_redundant_features(df)
    X_reduzido, _ = build_features_target(df_reduzido)

    print(f"\nProtocolo: {N_SPLITS} folds x {N_REPEATS} repetições = "
          f"{N_SPLITS * N_REPEATS} medições por configuração\n")

    resultados = pd.DataFrame(
        avaliar(X_completo, y, f"{X_completo.shape[1]} features")
        + avaliar(X_reduzido, y, f"{X_reduzido.shape[1]} features")
    )

    pd.set_option("display.width", 200)
    for modelo in resultados["modelo"].unique():
        print(f"=== {modelo} ===")
        print(resultados[resultados["modelo"] == modelo]
              .drop(columns="modelo").to_string(index=False))
        print()

    destino = REPORTS_DIR / "ablacao_preprocessamento.csv"
    resultados.to_csv(destino, index=False)
    print(f"Tabela salva em: {destino}")
    print("Leitura dos resultados: seção 6.1 do reports/RELATORIO_TECNICO.md")


if __name__ == "__main__":
    main()
