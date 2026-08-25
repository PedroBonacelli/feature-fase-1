"""
Carrega o dataset Breast Cancer Wisconsin (Diagnostic) e salva uma cópia bruta
em data/raw/breast_cancer_wisconsin.csv.

Este é o mesmo dataset disponibilizado no Kaggle
(https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data), obtido
aqui via scikit-learn (fonte original: UCI Machine Learning Repository) para
garantir reprodutibilidade sem depender de credenciais do Kaggle.

Uso:
    python src/load_data.py
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "breast_cancer_wisconsin.csv"


def load_breast_cancer_df() -> pd.DataFrame:
    """Retorna o dataset Breast Cancer Wisconsin como um DataFrame único.

    Colunas:
        - 30 features numéricas (medidas de núcleos celulares extraídas de
          imagens digitalizadas de biópsias por agulha fina - FNA).
        - 'target': 0 = maligno, 1 = benigno (codificação original do sklearn).
        - 'diagnosis': versão legível de 'target' ('malignant' / 'benign').
    """
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = load_breast_cancer_df()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset salvo em: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(df["diagnosis"].value_counts())


if __name__ == "__main__":
    main()
