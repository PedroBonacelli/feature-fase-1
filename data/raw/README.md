# Dados brutos

Este diretório recebe os datasets originais (não versionados no Git).

## Breast Cancer Wisconsin (Diagnostic)

- Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- Também disponível via `scikit-learn`:
  `from sklearn.datasets import load_breast_cancer`
- Salve o CSV baixado do Kaggle como `data/raw/breast_cancer_wisconsin.csv`.

## [EXTRA] CBIS-DDSM (mamografias)

- Fonte: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
- Dataset grande (imagens DICOM/JPEG); baixar via Kaggle API.
- Organize os arquivos na estrutura esperada por `src/cnn_mammography.py`:

  ```
  data/raw/cbis-ddsm/
      train/
          benign/     *.png
          malignant/  *.png
      test/
          benign/     *.png
          malignant/  *.png
  ```

- Este dataset não foi baixado nem processado no ambiente de desenvolvimento
  usado neste projeto (sandbox sem acesso à internet fora do índice de
  pacotes Python). Ver `reports/03_cnn_extra.md` para detalhes e instruções
  completas de execução local/Colab.
