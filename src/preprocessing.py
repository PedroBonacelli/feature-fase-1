"""
Pipeline de pré-processamento — Breast Cancer Wisconsin (Diagnostic)

Etapas:
    1. Carrega os dados brutos.
    2. Trata valores ausentes/inconsistentes (defensivo — o dataset já está
       limpo, mas o pipeline fica preparado para lidar com eles).
    3. Analisa a correlação entre features e REMOVE a redundância
       (multicolinearidade): de cada par com |correlação| >= 0.92, mantém
       apenas a feature mais correlacionada com o alvo.
    4. Separa features (X) e alvo (y), removendo colunas redundantes.
    5. Faz a separação treino/teste (estratificada, antes de qualquer ajuste
       de escala, para evitar vazamento de dados/data leakage).
    6. Escalona as features numéricas (StandardScaler) — necessário pois as
       features têm escalas muito diferentes (ex.: 'area' vs 'smoothness').
    7. Salva os conjuntos processados em data/processed/, o scaler ajustado
       em models/ e o registro das features removidas em reports/.

Uso:
    python src/preprocessing.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "breast_cancer_wisconsin.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TARGET_COL = "target"          # 0 = maligno, 1 = benigno
DROP_COLS = ["diagnosis"]      # coluna categórica redundante com 'target'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Limiar de multicolinearidade. 0.92 separa bem os grupos reais de redundância
# do dataset (as medidas de tamanho — radius/perimeter/area — e o par
# concavity/concave points) sem descartar features que apenas se parecem.
CORR_THRESHOLD = 0.92


def load_and_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    n_before = len(df)

    # Tratamento defensivo de inconsistências / ausentes, mesmo que o
    # dataset já esteja limpo (garante robustez do pipeline para outras
    # execuções/dados futuros vindos do mesmo processo de coleta).
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include="number").columns
    n_missing = df[numeric_cols].isna().sum().sum()
    if n_missing > 0:
        # Imputação simples pela mediana (robusta a outliers) por coluna.
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Medidas físicas (radius, area, etc.) não podem ser <= 0 -- linhas
    # assim seriam consideradas inconsistentes e removidas.
    physical_cols = [c for c in numeric_cols if c not in (TARGET_COL,)]
    invalid_mask = (df[physical_cols] < 0).any(axis=1)
    df = df.loc[~invalid_mask].copy()

    # Colunas totalmente vazias não carregam informação (o CSV do Kaggle traz
    # uma coluna 'Unnamed: 32' nesse estado). Removidas por segurança.
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        print(f"Colunas totalmente vazias removidas: {empty_cols}")

    n_after = len(df)
    print(f"Linhas removidas na limpeza (duplicatas/inconsistências): {n_before - n_after}")
    print(f"Valores ausentes tratados (imputação por mediana): {n_missing}")

    # Balanceamento das classes: define o piso de acurácia contra o qual os
    # modelos precisam ser comparados (um classificador que chutasse sempre a
    # classe majoritária já acertaria essa porcentagem).
    proporcao = df[TARGET_COL].value_counts(normalize=True).sort_index() * 100
    print(f"Distribuição das classes — maligno (0): {proporcao.loc[0]:.2f}% | "
          f"benigno (1): {proporcao.loc[1]:.2f}%")

    return df


def high_corr_pairs(corr: pd.DataFrame, threshold: float) -> list[tuple[str, str, float]]:
    """Pares de features com |correlação| >= threshold, do mais alto ao mais baixo."""
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if abs(value) >= threshold:
                pairs.append((cols[i], cols[j], value))
    return sorted(pairs, key=lambda p: -abs(p[2]))


def drop_redundant_features(
    df: pd.DataFrame, threshold: float = CORR_THRESHOLD
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove features redundantes por multicolinearidade.

    Duas features com |correlação| >= threshold carregam praticamente a mesma
    informação. Manter as duas não acrescenta sinal e traz custos concretos:
        - na Regressão Logística, o peso do efeito se divide arbitrariamente
          entre as colunas correlacionadas, o que instabiliza os coeficientes e
          distorce a leitura de importância usada na explicabilidade;
        - no Random Forest, a importância de uma variável relevante se dilui
          entre suas cópias, empurrando-a para baixo no ranking;
        - toda coluna a mais é ruído adicional em um dataset de apenas 569
          amostras.

    Critério de desempate: entre as duas features do par, mantemos a que tem
    maior |correlação| com o alvo — ou seja, a que carrega mais sinal
    diagnóstico. O processo é guloso e iterativo: a cada passo elimina-se o par
    mais correlacionado ainda presente e recalcula-se o que sobrou, até não
    restar nenhum par acima do limiar.
    """
    features = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    target_corr = features.corrwith(df[TARGET_COL]).abs()

    remaining = list(features.columns)
    decisions = []

    while True:
        corr = features[remaining].corr()
        pairs = high_corr_pairs(corr, threshold)
        if not pairs:
            break

        a, b, value = pairs[0]
        # Mantém a feature com maior correlação (em módulo) com o alvo.
        kept, dropped = (a, b) if target_corr[a] >= target_corr[b] else (b, a)
        remaining.remove(dropped)
        decisions.append({
            "feature_removida": dropped,
            "correlacionada_com": kept,
            "corr_entre_features": round(value, 4),
            "corr_removida_vs_alvo": round(target_corr[dropped], 4),
            "corr_mantida_vs_alvo": round(target_corr[kept], 4),
        })

    report = pd.DataFrame(decisions)

    print(f"\n{'=' * 70}")
    print(f"REMOÇÃO DE FEATURES REDUNDANTES (|correlação| >= {threshold})")
    print("=" * 70)
    if report.empty:
        print("Nenhum par acima do limiar — todas as features foram mantidas.")
    else:
        for row in decisions:
            print(f"  removida '{row['feature_removida']}' "
                  f"(corr {row['corr_entre_features']:+.3f} com "
                  f"'{row['correlacionada_com']}'); "
                  f"alvo: {row['corr_removida_vs_alvo']:.3f} vs "
                  f"{row['corr_mantida_vs_alvo']:.3f} da mantida")
    print(f"\nFeatures: {features.shape[1]} -> {len(remaining)} "
          f"({features.shape[1] - len(remaining)} removidas)")

    kept_cols = [c for c in df.columns if c in remaining or c in DROP_COLS + [TARGET_COL]]
    return df[kept_cols].copy(), report


def build_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    y = df[TARGET_COL]
    return X, y


def split_and_scale(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    # Ajusta o scaler SOMENTE no treino, para evitar vazamento de dados.
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()
    df, corr_report = drop_redundant_features(df)

    X, y = build_features_target(df)
    (X_train, X_test, X_train_scaled, X_test_scaled,
     y_train, y_test, scaler) = split_and_scale(X, y)

    print(f"\nTreino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
    print(f"Proporção da classe 'maligno' (target=0) — treino: "
          f"{(y_train == 0).mean():.3f} | teste: {(y_test == 0).mean():.3f}")

    # Salva versões não-escalonadas (úteis p/ árvores e interpretabilidade)
    # e escalonadas (úteis p/ modelos lineares e KNN).
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    X_train_scaled.to_csv(PROCESSED_DIR / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(PROCESSED_DIR / "X_test_scaled.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    # Registro auditável de quais features foram descartadas e por quê.
    corr_report.to_csv(REPORTS_DIR / "features_removidas.csv", index=False)

    # Estatísticas descritivas transpostas (uma linha por variável): o terminal
    # trunca as 30 colunas, então o CSV é a forma legível de consultá-las.
    X.describe().T.to_csv(REPORTS_DIR / "estatisticas_descritivas.csv")

    print(f"\nArquivos processados salvos em: {PROCESSED_DIR}")
    print(f"Scaler salvo em: {MODELS_DIR / 'scaler.joblib'}")
    print(f"Features removidas registradas em: {REPORTS_DIR / 'features_removidas.csv'}")
    print(f"Estatísticas descritivas em: {REPORTS_DIR / 'estatisticas_descritivas.csv'}")


if __name__ == "__main__":
    main()
