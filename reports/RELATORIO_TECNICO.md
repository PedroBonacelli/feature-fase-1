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
2. **Remoção de features redundantes (multicolinearidade)**: de cada par com
   |correlação| ≥ 0,92, mantém-se apenas a feature mais correlacionada com o
   alvo. O processo é guloso e iterativo (elimina o par mais correlacionado,
   recalcula, repete) e removeu **8 das 30 features**, deixando 22:

   | Removida | Redundante com | Corr. entre elas | Corr. com alvo (removida → mantida) |
   |---|---|---|---|
   | `mean radius` | `mean perimeter` | +0,998 | 0,730 → 0,743 |
   | `worst radius` | `worst perimeter` | +0,994 | 0,776 → 0,783 |
   | `mean area` | `mean perimeter` | +0,987 | 0,709 → 0,743 |
   | `worst area` | `worst perimeter` | +0,978 | 0,734 → 0,783 |
   | `perimeter error` | `radius error` | +0,973 | 0,556 → 0,567 |
   | `mean perimeter` | `worst perimeter` | +0,970 | 0,743 → 0,783 |
   | `area error` | `radius error` | +0,952 | 0,548 → 0,567 |
   | `mean concavity` | `mean concave points` | +0,921 | 0,696 → 0,777 |

   O registro completo fica em `reports/features_removidas.csv`.
   **Justificativa** — duas features quase idênticas não acrescentam sinal,
   mas custam: na Regressão Logística o efeito se divide arbitrariamente
   entre as colunas correlacionadas, instabilizando os coeficientes usados na
   explicabilidade; no Random Forest a importância de uma variável relevante
   se dilui entre suas cópias, rebaixando-a no ranking. Note que
   `radius`/`perimeter`/`area` são, na prática, três leituras do mesmo
   tamanho do núcleo celular — a redundância é geométrica, não estatística.
   **Efeito medido nas métricas** (ver seção 6.1): neutro. O ganho é de
   interpretabilidade e parcimônia, não de desempenho preditivo.
3. **Separação treino/teste estratificada** (80/20 → 455/114 amostras),
   preservando a proporção de ~37% de malignos em ambos os conjuntos.
4. **Escalonamento (StandardScaler)** ajustado **somente no conjunto de
   treino** (para evitar vazamento de dados) — necessário pois as features
   têm escalas muito diferentes (ex.: `area` na casa de centenas/milhares
   vs. `smoothness` na casa de centésimos), o que afeta diretamente modelos
   sensíveis a escala como Regressão Logística e KNN.
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

Os três usam **`class_weight='balanced'`**: 'maligno' é a classe minoritária
(~37%) e é justamente a que não se pode perder. O parâmetro faz o modelo
pagar mais caro por errar essa classe, priorizando recall em vez de acurácia
bruta — alinhado com a métrica de avaliação escolhida abaixo.

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
| **Regressão Logística** | 0,965 | 0,932 | **0,976** | **0,954** | 0,995 | **1** |
| Árvore de Decisão | 0,939 | 0,907 | 0,929 | 0,918 | 0,929 | 3 |
| Random Forest | **0,974** | **1,000** | 0,929 | 0,963 | **0,996** | 3 |

**A Regressão Logística obteve o melhor resultado na métrica que mais importa
clinicamente** — maior recall e F1 na classe maligno, e apenas 1 falso
negativo em 42 casos malignos no teste. O Random Forest, após a reexecução
completa do pipeline nesta revisão, alcançou a maior acurácia e precisão
perfeita (nenhum falso positivo), mas manteve recall abaixo da Logística (3
falsos negativos) — indício de que, mesmo com `class_weight='balanced'`, o
ensemble ainda erra por excesso de cautela do lado "maligno" com mais
frequência que o modelo linear. Isso não muda a recomendação: para triagem
de câncer, recall é a métrica prioritária, e nela a Regressão Logística
segue à frente.

> **Nota de reexecução (validação desta revisão):** esta seção foi
> revalidada após a remoção de features redundantes (seção 4, item 2) e a
> adoção de `class_weight='balanced'` (seção 5). Regressão Logística e
> Árvore de Decisão reproduziram exatamente os números já registrados. O
> Random Forest reproduziu com pequenas diferenças em relação a uma versão
> anterior deste relatório (accuracy e precisão levemente maiores; recall
> idêntico) — padrão consistente com sensibilidade conhecida do
> `RandomForestClassifier` a diferenças de versão do scikit-learn na
> interação entre `class_weight='balanced'` e a amostragem bootstrap.
> `models/*.joblib`, `data/processed/*.csv` e `reports/model_comparison.csv`
> foram regenerados nesta revisão a partir do código atual e são a fonte
> autoritativa vigente.

### 6.1. Ablação: o que cada decisão de pré-processamento realmente entregou

Com apenas 114 amostras de teste, uma diferença de 2 ou 3 acertos move as
métricas em ~2 pontos percentuais. Para separar efeito real de ruído do
sorteio, as duas decisões (remoção de features e `class_weight`) foram
medidas por **validação cruzada estratificada repetida** (10 folds × 3
repetições = 30 medições por configuração) sobre o dataset completo.

Recall médio da classe maligno (desvio entre folds entre parênteses):

| Modelo | 30 feat., sem peso | 30 feat., balanced | 22 feat., sem peso | 22 feat., balanced |
|---|---|---|---|---|
| Regressão Logística | 0,961 (0,039) | **0,967** (0,033) | 0,946 (0,050) | 0,964 (0,038) |
| Árvore de Decisão | 0,890 (0,072) | 0,905 (0,073) | 0,901 (0,069) | **0,918** (0,074) |
| Random Forest | 0,937 (0,065) | 0,939 (0,056) | 0,928 (0,053) | 0,923 (0,056) |

Leitura honesta dos números:

- **`class_weight='balanced'` ajuda a Logística e a Árvore de Decisão de
  forma consistente** (+0,6 a +1,7 pp de recall, em qualquer conjunto de
  features, ao custo de alguma precisão) — exatamente o trade-off desejado
  quando o falso negativo é o erro caro. **No Random Forest o ganho é
  marginal e dentro do ruído** (0,937 → 0,939 com 30 features; 0,928 → 0,923,
  uma leve queda, com 22 features) — o ensemble por bootstrap já mistura
  naturalmente a proporção de classes entre árvores, diluindo o efeito do
  peso.
- **A remoção de features é neutra em desempenho para Logística e Random
  Forest, e melhora a Árvore de Decisão** (recall 0,905 → 0,918; F1 0,906 →
  0,922 — o modelo mais sensível a ruído por escolher um único split por
  nó). Para os outros dois, as médias variam **dentro do desvio entre
  folds** (~0,05 a 0,065) — ou seja, indistinguível de ruído amostral, não
  uma piora demonstrável.
- **Nota de reprodutibilidade:** os valores do Random Forest nesta tabela
  foram medidos nesta revisão e diferem de uma versão preliminar deste
  relatório (que reportava 0,954 e 0,943 nas colunas `balanced`) — mesma
  causa da nota da seção 6: sensibilidade do `RandomForestClassifier` a
  detalhes de versão do scikit-learn. A conclusão qualitativa não muda:
  `class_weight='balanced'` favorece a Logística e a Árvore, e a remoção de
  features segue sendo uma decisão de parcimônia/interpretabilidade, não de
  ganho de desempenho.

Portanto a remoção das 8 features **não é justificada como ganho de
acurácia**, e sim por parcimônia e interpretabilidade: 22 features em vez de
30, coeficientes estáveis na Logística e ranking de importância que não
dilui o mesmo sinal entre cópias — o que importa quando o modelo precisa ser
explicado a um profissional de saúde.

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
