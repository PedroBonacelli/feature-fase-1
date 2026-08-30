# [EXTRA] Visão Computacional — CNN para Mamografias (CBIS-DDSM)

## Status

O código da CNN (`src/cnn_mammography.py`, mais `src/cnn_data_prep.py` pra
organizar o dataset) está completo e pronto pra rodar, mas não foi
executado nesta entrega — explico por quê e como rodar abaixo.

### Por que não rodei aqui

O ambiente usado pra construir o resto do projeto é um sandbox na nuvem
sem acesso geral à internet (só um índice restrito de pacotes já
instalados) e sem GPU. TensorFlow/Keras não instala nesse índice, e o
CBIS-DDSM não dá pra baixar (precisa de autenticação Kaggle e tem dezenas
de GB de imagens).

Como essa etapa é marcada como extra/opcional no enunciado, priorizei
entregar com qualidade as etapas obrigatórias (EDA, pré-processamento,
modelagem, avaliação e explicabilidade) e deixar aqui um pipeline de CNN
completo e documentado, pronto pra quem tiver GPU/Colab e acesso ao
Kaggle.

### Como rodar

1. Baixe o CBIS-DDSM: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
2. Rode `src/cnn_data_prep.py --cbis-root <pasta baixada>` pra organizar
   as imagens na estrutura que o script espera:
   ```
   data/raw/cbis-ddsm/
       train/{benign,malignant}/*.jpg
       test/{benign,malignant}/*.jpg
   ```
3. `pip install tensorflow scikit-learn` (já no `requirements.txt`)
4. `python src/cnn_mammography.py`

## O que o pipeline faz

Carrega as imagens via `image_dataset_from_directory` em escala de cinza
(mamografia não tem informação de cor relevante), com split
treino/validação interno de 15% e um conjunto de teste separado. Aplica
data augmentation moderada (flip horizontal, rotação e zoom pequenos,
contraste) — evitando transformações que distorçam a anatomia.

Arquitetura: 4 blocos convolucionais (32→64→128→256 filtros) com batch
norm e max pooling, seguidos de global average pooling (menos parâmetros
e menos overfitting que Flatten + Dense grande) e uma cabeça densa com
dropout. Pesos de classe calculados automaticamente
(`compute_class_weight`), já que o CBIS-DDSM costuma ter mais casos
benignos — mesma lógica de cuidado do dataset estruturado. Monitora
`val_auc` (não só accuracy), com early stopping e redução de learning
rate.

Avaliação: classification report, matriz de confusão e curva ROC no
teste, salvos em `reports/figures/` (`10_cnn_training_curves`,
`11_cnn_confusion_matrix`, `12_cnn_roc`).

Explicabilidade via Grad-CAM: a função `grad_cam()` gera um mapa de calor
mostrando quais regiões da imagem mais influenciaram a predição — o
equivalente, em visão computacional, ao que o SHAP faz pros dados
estruturados. Deixa um radiologista auditar visualmente se o modelo está
olhando pra região suspeita ou pra um artefato irrelevante da imagem.

## Discussão crítica (mesmo sem execução)

Os mesmos princípios da avaliação do modelo estruturado valem aqui, com
um agravante: modelos de visão computacional em imagem médica são
particularmente propensos a aprender atalhos espúrios (marcação de
equipamento, diferença de protocolo entre hospitais que vira artefato
visual associado a uma classe). Por isso o Grad-CAM não é opcional — é
verificação obrigatória antes de considerar qualquer uso, mesmo
experimental. O resultado da CNN também deveria funcionar só como
segunda opinião de triagem, nunca diagnóstico automatizado, com o
radiologista sempre revisando imagem e predição.
