# Avaliação e Explicabilidade dos Modelos — Breast Cancer Wisconsin

## Métricas no conjunto de teste (114 amostras, nunca vistas no treino)

| Modelo | Accuracy | Precision (maligno) | Recall (maligno) | F1 (maligno) | ROC AUC | Falsos negativos |
|---|---|---|---|---|---|---|
| Regressão Logística | 0.983 | 0.976 | 0.976 | 0.976 | 0.995 | **1** |
| Árvore de Decisão | 0.921 | 0.867 | 0.929 | 0.897 | 0.916 | 3 |
| Random Forest | 0.947 | 0.929 | 0.929 | 0.929 | 0.994 | 3 |

*(Ver `reports/model_comparison.csv`, `reports/figures/06_confusion_matrices.png` e
`07_roc_curves.png`.)*

## Por que essa métrica: recall da classe "maligno" é a mais crítica

Accuracy sozinha é enganosa aqui: como as classes são desbalanceadas (63%
benigno / 37% maligno), um modelo "preguiçoso" que sempre previsse "benigno"
já teria ~63% de acurácia sem detectar nenhum câncer. O que importa
clinicamente é o **recall da classe maligno** (quantos casos de câncer
realmente detectamos) e, secundariamente, o **F1-score** (equilíbrio entre
recall e precisão).

- Um **falso negativo** (câncer classificado como benigno) é o erro mais
  grave: atrasa o diagnóstico e o tratamento de uma paciente que realmente
  tem câncer.
- Um **falso positivo** (benigno classificado como maligno) gera ansiedade e
  exames adicionais, mas é um erro muito menos custoso em termos de saúde.

Por isso, entre os três modelos, a **Regressão Logística é a mais adequada**
para esse problema: maior recall (97,6%) e apenas **1 falso negativo** em 42
casos malignos no teste, além do maior ROC AUC (0.995).

## Matriz de confusão (resumo)

- **Regressão Logística**: 41/42 malignos corretos, 71/72 benignos corretos.
- **Árvore de Decisão**: 39/42 malignos corretos (3 falsos negativos).
- **Random Forest**: 39/42 malignos corretos (3 falsos negativos), apesar de
  100% de acurácia no treino — confirma que o Random Forest, mesmo com
  profundidade limitada, teve leve overfitting em relação à Regressão
  Logística, que generalizou melhor para dados novos.

## Explicabilidade

### Feature importance nativa e permutation importance

Ver `reports/figures/08_feature_importance.png` e `09_permutation_importance.png`.

Nos três modelos, as features mais relevantes se concentram consistentemente em:
- **Tamanho**: `worst radius`, `worst area`, `worst perimeter`, `mean area`.
- **Irregularidade de contorno**: `worst concave points`, `worst concavity`,
  `mean concave points`.
- **Erros de medição de tamanho**: `radius error`, `area error` — indicam
  que a variabilidade da medição do tumor também carrega sinal.

Isso é consistente com a EDA (etapa 3) e reforça, sob uma ótica de múltiplos
modelos e métodos de importância diferentes, que tumores maiores e com bordas
mais irregulares (mais côncavas) são o padrão associado à malignidade nesse
dataset — coerente com o conhecimento médico sobre morfologia tumoral.

Para o Random Forest, a permutation importance ficou com valores baixos e
concentrados em poucas features — esperado, já que o modelo está muito
próximo do teto de desempenho no teste (pouca margem para a métrica cair ao
embaralhar uma única feature).

### Explicação tipo-SHAP (nível de predição individual)

Como o pacote `shap` não pôde ser instalado neste ambiente de execução
(sandbox sem acesso ao índice completo do PyPI — ele segue listado em
`requirements.txt` para uso local/Docker com `shap.LinearExplainer` /
`shap.TreeExplainer`), implementei em `src/evaluation.py` a **fórmula exata
do SHAP para modelos lineares**: para a Regressão Logística,
`f(x) = intercepto + Σ coef_i·x_i`, o valor SHAP de cada feature é
`coef_i · (x_i − média_treino_i)` — a base matemática do
`shap.LinearExplainer` (Lundberg & Lee, 2017), sem depender da biblioteca.

Exemplo de dois casos malignos corretamente classificados: as maiores
contribuições positivas (empurrando para "maligno") vieram de `worst area`,
`worst radius`, `worst perimeter` e `mean concave points` — exatamente as
features de tamanho/irregularidade identificadas como mais importantes
globalmente. Para casos benignos, as mesmas features tiveram contribuição
negativa (empurrando para "benigno"), com destaque também para
`worst texture` e `worst smoothness`.

Essa consistência entre explicação global (feature importance) e local
(SHAP-like por caso) é um bom sinal de que o modelo aprendeu um padrão
clinicamente plausível, e não um artefato espúrio dos dados.

## Discussão crítica: esse modelo pode ser usado na prática?

**Sim, mas apenas como ferramenta de apoio à decisão — nunca como
substituto do diagnóstico médico.** Alguns pontos centrais:

1. **O médico deve sempre ter a palavra final.** Mesmo com 97,6% de recall,
   a Regressão Logística ainda errou 1 em 42 casos malignos no teste. Em um
   cenário real, esse modelo funcionaria como uma **camada de triagem/apoio**
   — sinalizando casos de maior risco para priorização e revisão,
   nunca liberando um caso como "benigno" sem revisão humana.
2. **O dataset é pequeno e de uma única fonte** (569 casos, features
   derivadas de biópsias por agulha fina de um processo de imagem
   específico). Um modelo em produção precisaria ser validado em dados de
   outras clínicas/equipamentos antes de qualquer uso clínico real, para
   garantir que generaliza além do viés de coleta desse dataset específico
   (data drift entre hospitais, equipamentos, populações).
3. **Explicabilidade é pré-requisito, não luxo.** Em saúde, um modelo "caixa
   preta" que erra sem explicação é inaceitável. As técnicas usadas aqui
   (feature importance, permutation importance, explicação tipo-SHAP)
   permitem que o profissional de saúde entenda *por que* o modelo sinalizou
   um caso como suspeito, e decida se aquilo faz sentido clinicamente.
4. **Custo assimétrico dos erros.** Como discutido, um falso negativo é
   muito mais grave que um falso positivo neste contexto — isso deveria
   inclusive guiar o ajuste do limiar de decisão (threshold) do modelo em
   produção: pode valer a pena aceitar mais falsos positivos (mais exames
   de confirmação) em troca de recall ainda mais alto na classe maligno.
5. **Próximos passos para uso real**: validação clínica prospectiva,
   auditoria de viés (o dataset representa adequadamente a população de
   pacientes do hospital?), monitoramento contínuo de performance pós-
   deploy, e um fluxo de trabalho claro onde a predição do modelo é sempre
   uma entre várias informações consideradas pelo médico — nunca a decisão
   final.

## Escolha final para a Fase 1

A **Regressão Logística** é o modelo recomendado desta fase: melhor recall e
F1 na classe maligno, maior AUC, e é o modelo mais interpretável dos três
(coeficientes diretamente relacionáveis a cada feature). O Random Forest fica
como segunda opção robusta, e a Árvore de Decisão como versão mais simples
de visualizar/explicar em uma reunião com não-técnicos, ao custo de menor
recall.
