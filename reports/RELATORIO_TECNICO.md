# Relatório Técnico — Tech Challenge Fase 1

**Sistema de apoio ao diagnóstico de câncer de mama (Breast Cancer Wisconsin)**

## 1. O problema

Uma rede de hospitais quer identificar precocemente condições que afetam a
segurança e saúde da mulher. Nesta fase, o foco é construir a base de um
sistema de ML que classifica exames de câncer de mama como **malignos** ou
**benignos** a partir de medidas extraídas de biópsias, apoiando (nunca
substituindo) a decisão médica.

## 2. Dataset

- **Breast Cancer Wisconsin (Diagnostic)** — 569 exames, 30 features
  numéricas derivadas de imagens digitalizadas de biópsias por agulha fina
  (FNA), organizadas em três grupos: `mean`, `error` e `worst` para 10
  medidas de forma/textura do núcleo celular (radius, texture, perimeter,
  area, smoothness, compactness, concavity, concave points, symmetry,
  fractal dimension).
- Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
  (carregado via `sklearn.datasets.load_breast_cancer`, mesma base, origem
  UCI Machine Learning Repository — ver `src/load_data.py`).
- Variável alvo: `target` (0 = maligno, 1 = benigno). Distribuição:
  **357 benignos (62,7%) / 212 malignos (37,3%)** — desbalanceamento
  moderado, relevante para a escolha de métrica (seção 5).

## 3. Análise exploratória (EDA)

*(detalhamento completo em `reports/01_eda_findings.md`, figuras
`reports/figures/01` a `05`)*

- Dataset limpo: **zero valores ausentes, zero duplicatas** — não exigiu
  imputação.
- Há **separação visual clara** entre malignos e benignos em features de
  tamanho e irregularidade de contorno (`radius`, `perimeter`, `area`,
  `compactness`, `concavity`, `concave points`): tumores malignos tendem a
  ser maiores e com bordas mais irregulares, coerente com o conhecimento
  clínico sobre morfologia tumoral.
- `texture` e `smoothness` mostram maior sobreposição entre classes —
  menor poder discriminativo isolado.
- **Forte multicolinearidade** entre medidas de tamanho (`radius`,
  `perimeter`, `area` — correlação > 0,98 entre pares), o que é esperado
  matematicamente (perímetro e área derivam do raio) e orientou decisões de
  modelagem (seção 4).
- As features mais correlacionadas com o alvo são `concave points`,
  `perimeter` e `radius` (tanto `mean` quanto `worst`).

## 4. Pré-processamento

*(código em `src/preprocessing.py`)*

1. **Limpeza defensiva**: remoção de duplicatas, imputação por mediana de
   eventuais ausentes, descarte de medidas fisicamente inválidas
   (negativas). Neste dataset, 0 registros foram afetados — mas o pipeline
   fica preparado para dados futuros mais "sujos" vindos do mesmo processo
   de coleta.
2. **Separação treino/teste estratificada** (80/20 → 455/114 amostras),
   preservando a proporção de ~37% de malignos em ambos os conjuntos.
3. **Escalonamento (StandardScaler)** ajustado **somente no conjunto de
   treino** (para evitar vazamento de dados) — necessário pois as features
   têm escalas muito diferentes (ex.: `area` na casa de centenas/milhares
   vs. `smoothness` na casa de centésimos), o que afeta diretamente modelos
   sensíveis a escala como Regressão Logística e KNN.
4. **Análise de correlação**: confirmou 15 pares de features com
   correlação > 0,95. Decisão tomada: **manter todas as features**, em vez
   de removê-las às cegas, e deixar a regularização (Regressão Logística) e
   a robustez natural a multicolinearidade dos modelos baseados em árvore
   lidarem com a redundância.
5. Como todas as variáveis já são numéricas (não há variáveis categóricas
   no dataset original, além do alvo), não foi necessário one-hot
   encoding — o `diagnosis` (rótulo textual redundante com `target`) foi
   descartado do conjunto de features.

## 5. Modelagem

*(código em `src/modeling.py`)*

Foram treinadas **três técnicas de classificação** (acima do mínimo de duas
exigido):

| Modelo | Por que foi escolhido |
|---|---|
| **Regressão Logística** | Baseline linear, altamente interpretável (coeficientes indicam direção/força de cada feature) — importante em contexto clínico. |
| **Árvore de Decisão** (`max_depth=5`) | Captura relações não-lineares, fácil de visualizar e explicar a profissionais não-técnicos. |
| **Random Forest** (300 árvores, `max_depth=8`) | Ensemble mais robusto, usado como baseline de comparação mais forte. |

### Escolha da métrica de avaliação

Accuracy sozinha é enganosa com classes desbalanceadas (63%/37%): um modelo
que sempre previsse "benigno" já teria ~63% de acurácia sem detectar nenhum
câncer. A métrica mais crítica aqui é o **recall da classe maligno**
(quantos cânceres realmente detectamos) e, secundariamente, o **F1-score**.
Um **falso negativo** (câncer classificado como benigno) é o erro mais
grave clinicamente — muito mais custoso que um falso positivo.

## 6. Resultados

*(detalhamento completo em `reports/02_model_evaluation.md`, figuras
`reports/figures/06` a `09`, tabela em `reports/model_comparison.csv`)*

Métricas no conjunto de **teste** (114 amostras nunca vistas no treino):

| Modelo | Accuracy | Precision (maligno) | Recall (maligno) | F1 (maligno) | ROC AUC | Falsos negativos |
|---|---|---|---|---|---|---|
| **Regressão Logística** | **0,983** | 0,976 | **0,976** | **0,976** | **0,995** | **1** |
| Árvore de Decisão | 0,921 | 0,867 | 0,929 | 0,897 | 0,916 | 3 |
| Random Forest | 0,947 | 0,929 | 0,929 | 0,929 | 0,994 | 3 |

**A Regressão Logística obteve o melhor resultado** — maior recall e F1 na
classe maligno, maior AUC, e apenas 1 falso negativo em 42 casos malignos no
teste. Notavelmente, o Random Forest teve 100% de acurácia no treino mas
ficou atrás da Regressão Logística no teste — indício de leve overfitting
que a Logística, por ser mais simples e regularizada, evitou melhor.

### Explicabilidade

Três camadas de interpretação foram aplicadas:

1. **Feature importance nativa** (coeficientes / `feature_importances_`).
2. **Permutation importance** (model-agnostic, medida diretamente no
   conjunto de teste).
3. **Explicação tipo-SHAP** para a Regressão Logística: como o pacote
   `shap` não pôde ser instalado no ambiente sandbox usado para este
   projeto (sem acesso ao índice completo do PyPI — segue listado em
   `requirements.txt` para uso local/Docker), implementamos em
   `src/evaluation.py` a **fórmula analítica exata do SHAP para modelos
   lineares**: `phi_i = coef_i · (x_i − média_treino_i)` — a mesma base
   matemática do `shap.LinearExplainer` (Lundberg & Lee, 2017).

As três abordagens convergem: as features mais relevantes, tanto em nível
global quanto em explicações de casos individuais, são consistentemente
**tamanho** (`worst radius`, `worst area`, `worst perimeter`) e
**irregularidade de contorno** (`worst concave points`, `concavity`) — essa
convergência entre métodos independentes é um bom sinal de que o modelo
aprendeu um padrão clinicamente plausível.

## 7. [Extra] Visão computacional (CNN)

Um pipeline completo de CNN para diagnóstico via mamografia (dataset
CBIS-DDSM), incluindo Grad-CAM para explicabilidade visual, foi
**desenvolvido e documentado** em `src/cnn_mammography.py`, mas **não
executado** nesta entrega — o ambiente de desenvolvimento não tem acesso à
internet para baixar o dataset nem permite instalar TensorFlow. Detalhes,
justificativa e instruções completas de execução em
`reports/03_cnn_extra.md`.

## 8. Discussão crítica: esse modelo pode ser usado na prática?

**Sim, mas apenas como ferramenta de apoio à decisão — nunca como
substituto do diagnóstico médico.**

1. **O médico sempre tem a palavra final.** Mesmo com 97,6% de recall, o
   modelo ainda erra. Em produção, funcionaria como uma camada de
   triagem/priorização, nunca liberando um caso como "benigno" sem revisão
   humana.
2. **Dataset pequeno e de fonte única** (569 casos, um único processo de
   coleta/equipamento). Validação em dados de outras clínicas/equipamentos
   seria pré-requisito para qualquer uso clínico real.
3. **Explicabilidade é pré-requisito, não luxo**, em saúde. As três camadas
   de interpretação usadas permitem que o profissional entenda *por que* o
   modelo sinalizou um caso, e julgue se aquilo faz sentido clinicamente.
4. **Custo assimétrico dos erros** deveria orientar o ajuste do limiar de
   decisão em produção — pode valer a pena aceitar mais falsos positivos
   (mais exames de confirmação) em troca de recall ainda mais alto na
   classe maligno.
5. **Próximos passos para uso real**: validação clínica prospectiva,
   auditoria de viés populacional, monitoramento contínuo pós-deploy, e um
   fluxo de trabalho onde a predição é sempre uma entre várias informações
   consideradas pelo médico.

## 9. Conclusão

A Regressão Logística treinada sobre o Breast Cancer Wisconsin atingiu
97,6% de recall e AUC de 0,995 na classe maligno, com pipeline de
pré-processamento, análise de correlação e três camadas de explicabilidade
documentadas end-to-end. O resultado valida a viabilidade técnica de um
sistema de apoio à triagem para essa tarefa, respeitando desde já o
princípio de que a decisão final é sempre do profissional de saúde.
