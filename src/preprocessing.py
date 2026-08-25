"""
Pipeline de pré-processamento — Breast Cancer Wisconsin (Diagnostic)

Etapas:
    1. Carrega os dados brutos.
    2. Trata valores ausentes/inconsistentes (defensivo — o dataset já está
       limpo, mas o pipeline fica preparado para lidar com eles).
    3. Separa features (X) e alvo (y), removendo colunas redundantes.
    4. Faz a separação treino/teste (estratificada, antes de qualquer ajuste
       de escala, para evitar vazamento de dados/data leakage).
    5. Escalona as features numéricas (StandardScaler) — necessário pois as
       features têm escalas muito diferentes (ex.: 'area' vs 'smoothness').
    6. Analisa correlação entre features para identificar redundância
       (multicolinearidade) e documenta a decisão de mantê-las, contando com
       regularização/modelos baseados em árvore para lidar com isso.
    7. Salva os conjuntos processados em data/processed/ e o scaler ajustado
       em models/.

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

TARGET_COL = "target"          # 0 = maligno, 1 = benigno
DROP_COLS = ["diagnosis"]      # coluna categórica redundante com 'target'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Features com correlação > 0.95 entre si (ver EDA) — mantidas no dataset,
# mas documentadas aqui: a decisão foi NÃO removê-las de forma cega, e sim
# deixar que a regularização (Regressão Logística) e a robustez a
# multicolinearidade dos modelos baseados em árvore lidem com a redundância.
HIGH_CORR_GROUPS_NOTE = {
    "tamanho": ["mean radius", "mean perimeter", "mean area",
                "worst radius", "worst perimeter", "worst area"],
}


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

    n_after = len(df)
    print(f"Linhas removidas na limpeza (duplicatas/inconsistências): {n_before - n_after}")
    print(f"Valores ausentes tratados (imputação por mediana): {n_missing}")

    return df


def analyze_correlation(df: pd.DataFrame) -> None:
    numeric_df = df.drop(columns=DROP_COLS, errors="ignore")
    corr = numeric_df.drop(columns=[TARGET_COL]).corr()

    # Pares de features com correlação > 0.95 (fora da diagonal)
    high_corr_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if abs(value) > 0.95:
                high_corr_pairs.append((cols[i], cols[j], round(value, 3)))

    print(f"\nPares de features com |correlação| > 0.95: {len(high_corr_pairs)}")
    for a, b, v in sorted(high_corr_pairs, key=lambda x: -abs(x[2]))[:10]:
        print(f"  {a} <-> {b}: {v}")
    print(
        "Decisão: manter todas as features. A multicolinearidade será tratada "
        "via regularização (modelos lineares) e é naturalmente tolerada por "
        "modelos baseados em árvore (ver src/modeling.py)."
    )


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

    df = load_and_clean()
    analyze_correlation(df)

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

    print(f"\nArquivos processados salvos em: {PROCESSED_DIR}")
    print(f"Scaler salvo em: {MODELS_DIR / 'scaler.joblib'}")


if __name__ == "__main__":
    main()
