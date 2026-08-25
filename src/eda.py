"""
Análise Exploratória de Dados (EDA) — Breast Cancer Wisconsin (Diagnostic)

Carrega o dataset, gera estatísticas descritivas, visualizações de
distribuições e a matriz de correlação, salvando tudo em reports/figures/
para uso no relatório técnico.

Uso:
    python src/eda.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # ambiente sem display

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "breast_cancer_wisconsin.csv"
FIG_DIR = BASE_DIR / "reports" / "figures"

sns.set_theme(style="whitegrid")

# Features "mean" mais interpretáveis para inspeção visual detalhada
KEY_FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def print_overview(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("VISÃO GERAL DO DATASET")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print()
    print("Tipos de dados:")
    print(df.dtypes.value_counts())
    print()
    print("Valores ausentes por coluna (top 5, se houver):")
    na_counts = df.isna().sum().sort_values(ascending=False)
    print(na_counts.head(5))
    print(f"Total de valores ausentes: {df.isna().sum().sum()}")
    print()
    print("Duplicatas:", df.duplicated().sum())
    print()
    print("Distribuição da variável alvo (diagnosis):")
    print(df["diagnosis"].value_counts())
    print(df["diagnosis"].value_counts(normalize=True).round(3) * 100, "%")
    print()
    print("Estatísticas descritivas (features 'mean'):")
    print(df[KEY_FEATURES].describe().T)


def plot_class_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    order = df["diagnosis"].value_counts().index
    sns.countplot(data=df, x="diagnosis", order=order, hue="diagnosis",
                   palette={"benign": "#4C9A8E", "malignant": "#C0546B"}, legend=False, ax=ax)
    ax.set_title("Distribuição de diagnósticos (classe alvo)")
    ax.set_xlabel("Diagnóstico")
    ax.set_ylabel("Número de casos")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_distribution.png", dpi=150)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, feat in enumerate(KEY_FEATURES):
        sns.histplot(data=df, x=feat, hue="diagnosis", kde=True, element="step",
                     palette={"benign": "#4C9A8E", "malignant": "#C0546B"}, ax=axes[i])
        axes[i].set_title(feat)
    fig.suptitle("Distribuição das principais features por diagnóstico", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, feat in enumerate(KEY_FEATURES):
        sns.boxplot(data=df, x="diagnosis", y=feat, hue="diagnosis",
                    palette={"benign": "#4C9A8E", "malignant": "#C0546B"}, legend=False, ax=axes[i])
        axes[i].set_title(feat)
    fig.suptitle("Boxplots das principais features por diagnóstico", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_feature_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> pd.Series:
    numeric_df = df.drop(columns=["diagnosis"])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(corr, cmap="RdBu_r", center=0, square=True, linewidths=0.3,
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Matriz de correlação entre as features")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # Correlação de cada feature com o target (0=maligno, 1=benigno)
    target_corr = corr["target"].drop(["target"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 10))
    target_corr.plot(kind="barh", ax=ax, color=["#C0546B" if v < 0 else "#4C9A8E" for v in target_corr])
    ax.set_title("Correlação de cada feature com o alvo\n(target: 0=maligno, 1=benigno)")
    ax.set_xlabel("Coeficiente de correlação (Pearson)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_target_correlation.png", dpi=150)
    plt.close(fig)

    return target_corr


def print_correlation_discussion(target_corr: pd.Series) -> None:
    print()
    print("=" * 70)
    print("CORRELAÇÃO COM O ALVO (target: 0=maligno, 1=benigno)")
    print("=" * 70)
    print("Features mais NEGATIVAMENTE correlacionadas com 'benigno'")
    print("(ou seja, mais associadas a malignidade):")
    print(target_corr.head(8))
    print()
    print("Features mais POSITIVAMENTE correlacionadas com 'benigno':")
    print(target_corr.tail(8))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print_overview(df)
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_boxplots(df)
    target_corr = plot_correlation_heatmap(df)
    print_correlation_discussion(target_corr)

    print()
    print(f"Figuras salvas em: {FIG_DIR}")


if __name__ == "__main__":
    main()
