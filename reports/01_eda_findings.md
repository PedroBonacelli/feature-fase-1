# Análise Exploratória de Dados (EDA) — Breast Cancer Wisconsin

## Visão geral do dataset

- **569 registros**, **30 features numéricas** (medidas extraídas de imagens digitalizadas
  de biópsias por agulha fina — FNA — de massas mamárias), mais a variável alvo.
- **Nenhum valor ausente** e **nenhuma duplicata** — dataset já curado, o que simplifica o
  pré-processamento (não será necessário imputação).
- As 30 features são organizadas em 3 grupos de 10 medidas cada: `mean` (média),
  `error` (erro padrão) e `worst` (pior/maior valor observado) para: radius, texture,
  perimeter, area, smoothness, compactness, concavity, concave points, symmetry e
  fractal dimension.

## Distribuição da variável alvo

| Diagnóstico | Casos | % |
|---|---|---|
| Benigno | 357 | 62,7% |
| Maligno | 212 | 37,3% |

Há um **desbalanceamento moderado** (aprox. 63/37). Não é extremo a ponto de exigir
técnicas de balanceamento (SMOTE, undersampling), mas é relevante o suficiente para
que a métrica de avaliação não seja só *accuracy* — ver discussão na etapa de
avaliação do modelo (recall da classe maligno é a métrica crítica, já que um falso
negativo em câncer é muito mais grave que um falso positivo).

## Distribuições por diagnóstico

Ver `reports/figures/02_feature_distributions.png` e `03_feature_boxplots.png`.

Para várias features de forma/tamanho — `mean radius`, `mean perimeter`, `mean area`,
`mean compactness`, `mean concavity` e `mean concave points` — há **separação visível**
entre as distribuições de casos malignos e benignos: tumores malignos tendem a ser
maiores e ter contornos mais irregulares (maior concavidade e mais pontos côncavos).
Isso é clinicamente coerente: tumores malignos costumam ter crescimento mais
desorganizado e bordas irregulares, enquanto tumores benignos tendem a ser mais
regulares e compactos.

Já `mean texture` e `mean smoothness` mostram sobreposição maior entre as classes,
sugerindo menor poder discriminativo isolado — ainda assim podem contribuir em
conjunto com outras features no modelo.

## Correlação entre features

Ver `reports/figures/04_correlation_heatmap.png` e `05_target_correlation.png`.

- As features de **tamanho** (`radius`, `perimeter`, `area`) são **fortemente
  correlacionadas entre si** (correlação > 0.98 em vários pares) — isso é esperado
  matematicamente, já que perímetro e área derivam do raio. Isso indica
  **multicolinearidade**, relevante para a etapa de modelagem: modelos lineares
  (ex. regressão logística) podem sofrer com isso, enquanto modelos baseados em
  árvore são mais robustos a essa redundância.
- As features `concave points`, `perimeter` e `radius` (tanto na versão `mean`
  quanto `worst`) são as que apresentam **maior correlação (em módulo) com o
  diagnóstico**, reforçando que tamanho e irregularidade de contorno são os
  sinais mais fortes de malignidade nesse dataset.
- Features de "erro" (`*_error`) em geral têm correlação fraca com o alvo,
  sugerindo que a variabilidade da medição em si é menos informativa que a
  medida em si.

## Implicações para as próximas etapas

1. **Pré-processamento**: sem necessidade de tratar valores ausentes; será necessário
   apenas escalonar as features (elas têm escalas muito diferentes, ex. `area` na
   casa de centenas/milhares vs. `smoothness` na casa de centésimos), o que é
   importante especialmente para KNN e Regressão Logística.
2. **Seleção/redundância de features**: dada a alta multicolinearidade entre medidas
   de tamanho, `src/preprocessing.py` remove as features redundantes — de cada par
   com |correlação| ≥ 0,92, mantém apenas a mais correlacionada com o alvo,
   reduzindo de 30 para 22 features (detalhamento na seção 4 do
   `RELATORIO_TECNICO.md` e em `reports/features_removidas.csv`).
3. **Modelagem**: o bom poder discriminativo de várias features individuais sugere
   que mesmo modelos simples (Regressão Logística, Árvore de Decisão) devem
   performar bem — o que será testado e comparado na etapa de modelagem.
4. **Métrica de avaliação**: o desbalanceamento moderado das classes reforça a
   necessidade de acompanhar recall/F1 da classe maligno, não só accuracy.
