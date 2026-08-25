# [EXTRA] Visão Computacional — CNN para Mamografias (CBIS-DDSM)

## Status desta etapa

O código da CNN (`src/cnn_mammography.py`) está **completo e pronto para
execução**, mas **não foi executado** nesta entrega — abaixo explico por quê
e como rodar.

### Por que não foi executado aqui

O ambiente de desenvolvimento usado para construir este projeto é um sandbox
na nuvem sem acesso geral à internet (apenas um índice restrito de pacotes
Python já instalados) e sem GPU. Nele:

- **TensorFlow/Keras não pôde ser instalado** — o índice de pacotes
  disponível é uma lista fechada que não inclui `tensorflow` nem `torch`.
- **O dataset CBIS-DDSM não pôde ser baixado** — requer autenticação via
  Kaggle API e tem várias dezenas de GB (imagens DICOM/JPEG de alta
  resolução), e o sandbox não tem acesso à internet fora do índice de
  pacotes.

Como o próprio enunciado marca esta etapa como **EXTRA/opcional** ("não é
obrigatório, mas pode aumentar sua nota"), priorizei entregar com qualidade
total as etapas obrigatórias (EDA, pré-processamento, modelagem, avaliação e
explicabilidade — etapas 3 a 6) e disponibilizar aqui um pipeline de CNN
completo, correto e documentado, pronto para ser executado por quem tiver
GPU/Colab e acesso ao Kaggle.

### Como executar

1. Baixe o dataset CBIS-DDSM do Kaggle:
   https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
2. Organize as imagens na estrutura esperada pelo script (ver
   `data/raw/README.md`):
   ```
   data/raw/cbis-ddsm/
       train/
           benign/     *.png
           malignant/  *.png
       test/
           benign/     *.png
           malignant/  *.png
   ```
   (O dataset original do Kaggle já vem com metadados de patologia
   BENIGN/MALIGNANT por exame — um script de organização a partir do CSV de
   metadados é o próximo passo natural, adaptado conforme a versão exata do
   dataset baixada.)
3. Instale as dependências de imagem: `pip install tensorflow scikit-learn`
   (já listadas em `requirements.txt`).
4. Rode: `python src/cnn_mammography.py`

## O que o pipeline implementa

- **Carregamento** via `image_dataset_from_directory`, imagens em escala de
  cinza (mamografias não carregam informação de cor), split treino/validação
  interno de 15%, mais um conjunto de teste separado.
- **Data augmentation** moderada (flip horizontal, pequenas rotações, zoom,
  contraste) — evitando transformações que distorçam a anatomia de forma
  clinicamente implausível.
- **Arquitetura**: CNN com 4 blocos convolucionais (32→64→128→256 filtros)
  com batch normalization e max pooling, seguidos de global average pooling
  (reduz parâmetros e overfitting comparado a `Flatten` + `Dense` grande) e
  uma cabeça densa com dropout.
- **Balanceamento de classes**: pesos de classe calculados automaticamente
  (`compute_class_weight`), já que o CBIS-DDSM tende a ter mais casos
  benignos que malignos — mesma lógica de cuidado aplicada ao dataset
  estruturado na etapa 4.
- **Métrica de monitoramento**: `val_auc` (não apenas accuracy), com
  early stopping e redução de learning rate — coerente com a discussão da
  etapa 6 sobre a importância do recall/AUC da classe maligna neste
  domínio.
- **Avaliação**: classification report, matriz de confusão e curva ROC no
  conjunto de teste, salvos em `reports/figures/` (`10_cnn_training_curves`,
  `11_cnn_confusion_matrix`, `12_cnn_roc`).
- **Explicabilidade — Grad-CAM**: a função `grad_cam()` gera um mapa de
  calor sobre a imagem mostrando quais regiões mais influenciaram a
  predição do modelo. É o equivalente, em visão computacional, ao papel que
  o SHAP cumpre para os dados estruturados na etapa 6 — permite que um
  radiologista audite visualmente se o modelo está "olhando" para a região
  suspeita da mamografia, e não para um artefato irrelevante da imagem.

## Discussão crítica (mesmo sem execução)

Os mesmos princípios da etapa 6 se aplicam, com um agravante: modelos de
visão computacional em imagens médicas são particularmente propensos a
aprender **atalhos espúrios** (ex.: marcações do equipamento, diferenças de
protocolo entre hospitais que geram artefatos visuais consistentes com uma
classe). Por isso, o Grad-CAM não é opcional nessa etapa — é uma
verificação obrigatória antes de considerar qualquer uso do modelo, mesmo
experimental. Assim como no modelo estruturado, o resultado da CNN deveria
funcionar apenas como uma segunda opinião de triagem, nunca como
diagnóstico automatizado, com o radiologista sempre revisando a imagem e a
predição final.
