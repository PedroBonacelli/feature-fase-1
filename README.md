# Tech Challenge – Fase 1: IA para Saúde e Segurança da Mulher

Projeto desenvolvido para o Tech Challenge da Fase 1 (IADT). Constrói a base de um
sistema de suporte ao diagnóstico e detecção de riscos para a saúde da mulher, usando
Machine Learning sobre dados médicos estruturados e, na etapa extra, Visão
Computacional (CNN) sobre imagens de mamografia.

## Desafio

Uma rede de hospitais quer identificar precocemente condições que afetam a segurança
e a saúde feminina (ex.: câncer de mama). Esta fase constrói a base de IA/ML que
processa dados médicos para apoiar (nunca substituir) a decisão do profissional de
saúde.

## Datasets utilizados

- **Breast Cancer Wisconsin (Diagnostic)** — dados estruturados de exames para
  classificação de tumores em malignos/benignos.
  Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **[EXTRA] CBIS-DDSM** — imagens de mamografia para diagnóstico via CNN.
  Fonte: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

Os datasets não são versionados no repositório (ver `.gitignore`); instruções de
download estão em `data/raw/README.md`.

## Estrutura do projeto

```
tech-challenge-fase1/
├── data/
│   ├── raw/            # dados brutos (não versionados; ver data/raw/README.md)
│   └── processed/      # dados após pré-processamento (gerados por src/preprocessing.py)
├── src/                 # scripts Python do pipeline (ver "Como executar")
├── models/               # modelos treinados salvos (.joblib)
├── reports/
│   ├── figures/          # gráficos gerados por cada etapa
│   ├── RELATORIO_TECNICO.md   # relatório técnico consolidado (leia primeiro)
│   ├── 01_eda_findings.md     # discussão detalhada da EDA
│   ├── 02_model_evaluation.md # discussão detalhada da avaliação/explicabilidade
│   ├── 03_cnn_extra.md        # detalhes da etapa extra de CNN
│   └── model_comparison.csv   # tabela comparativa dos modelos
├── requirements.txt
├── Dockerfile
└── README.md
```

## Como executar

Este projeto usa **scripts Python** (não notebooks) — cada um roda uma etapa
do pipeline e imprime/salva os resultados em `reports/`. Rode-os na ordem:

### Opção 1: ambiente local (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/load_data.py       # baixa/gera o dataset em data/raw/
python src/eda.py             # análise exploratória -> reports/figures/
python src/preprocessing.py   # limpeza, split treino/teste, scaling -> data/processed/
python src/modeling.py        # treina os 3 modelos -> models/
python src/evaluation.py      # avaliação + explicabilidade -> reports/figures/, reports/model_comparison.csv
```

O script extra de CNN (`src/cnn_mammography.py`) requer TensorFlow e o
dataset CBIS-DDSM baixado à parte — ver `reports/03_cnn_extra.md`.

### Opção 2: Docker

```bash
docker build -t tech-challenge-fase1 .
docker run -v $(pwd):/app tech-challenge-fase1 python src/load_data.py
# (repita para os demais scripts, na mesma ordem acima)
```

## Resultados

Modelo recomendado: **Regressão Logística** — 97,6% de recall e AUC de 0,995
na classe "maligno" no conjunto de teste (114 amostras). Comparação completa
dos 3 modelos treinados em `reports/model_comparison.csv` e discussão em
`reports/RELATORIO_TECNICO.md`.

## Roteiro do projeto

1. Análise exploratória dos dados (EDA) — ✅ `src/eda.py`
2. Pré-processamento e pipeline — ✅ `src/preprocessing.py`
3. Modelagem (3 algoritmos de classificação) — ✅ `src/modeling.py`
4. Avaliação e explicabilidade (feature importance, permutation importance, SHAP) — ✅ `src/evaluation.py`
5. [Extra] CNN para diagnóstico via imagem (mamografias) — código pronto, não executado (ver `reports/03_cnn_extra.md`)
6. Relatório técnico com discussão crítica dos resultados — ✅ `reports/RELATORIO_TECNICO.md`

## Aviso

Este sistema é uma prova de conceito de apoio à decisão. O diagnóstico final é
sempre responsabilidade do profissional de saúde.
