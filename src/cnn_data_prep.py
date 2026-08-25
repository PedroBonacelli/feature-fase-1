"""
[EXTRA] Organiza o dataset CBIS-DDSM (baixado via kagglehub) na estrutura de
pastas benign/malignant esperada por src/cnn_mammography.py.

O download do kagglehub vem em duas pastas:
    <root>/csv/   -> metadados (mass_case_description_*.csv, calc_case_description_*.csv, dicom_info.csv)
    <root>/jpeg/  -> imagens, uma pasta por SeriesInstanceUID

Estratégia de mapeamento (validada inspecionando os CSVs reais do dataset):
    1. Cada linha dos CSVs de descrição de caso ('mass_case_description_*' e
       'calc_case_description_*') representa uma ANORMALIDADE (não a imagem
       inteira) e traz a coluna 'pathology' (BENIGN / BENIGN_WITHOUT_CALLBACK
       / MALIGNANT) e a coluna 'image file path', cujo primeiro componente
       (ex.: 'Mass-Training_P_00001_LEFT_CC') identifica a mamografia
       completa a que a anormalidade pertence.
    2. Em 'dicom_info.csv', as linhas com SeriesDescription ==
       'full mammogram images' têm essa mesma string em 'PatientName' e o
       caminho real do JPEG em 'image_path' (formato
       'CBIS-DDSM/jpeg/<SeriesInstanceUID>/<arquivo>.jpg').
    3. Uma mesma mamografia pode ter mais de uma anormalidade (linhas) — o
       rótulo final da imagem é 'malignant' se QUALQUER anormalidade nela for
       maligna (abordagem conservadora), senão 'benign'.
    4. Os CSVs de treino/teste já vêm pré-divididos por paciente (evita
       vazamento de dados entre os conjuntos) — usamos essa divisão original
       em vez de re-embaralhar.

Uso:
    python src/cnn_data_prep.py --cbis-root "C:\\Users\\pedro\\.cache\\kagglehub\\datasets\\awsaf49\\cbis-ddsm-breast-cancer-image-dataset\\versions\\1"

Isso vai popular:
    data/raw/cbis-ddsm/train/{benign,malignant}/*.jpg
    data/raw/cbis-ddsm/test/{benign,malignant}/*.jpg
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "cbis-ddsm"

CASE_DESCRIPTION_FILES = {
    "train": ["mass_case_description_train_set.csv", "calc_case_description_train_set.csv"],
    "test": ["mass_case_description_test_set.csv", "calc_case_description_test_set.csv"],
}


def load_full_mammogram_lookup(csv_dir: Path) -> pd.DataFrame:
    """Retorna um DataFrame [image_id -> jpeg_relative_path] a partir de dicom_info.csv,
    considerando apenas as imagens de mamografia completa (não recortes/ROI)."""
    dicom_info = pd.read_csv(csv_dir / "dicom_info.csv")
    full = dicom_info[dicom_info["SeriesDescription"] == "full mammogram images"].copy()
    # 'image_path' vem como 'CBIS-DDSM/jpeg/<uid>/<file>.jpg' -> mantemos só 'jpeg/<uid>/<file>.jpg'
    full["jpeg_relative_path"] = full["image_path"].str.replace(r"^CBIS-DDSM/", "", regex=True)
    lookup = full[["PatientName", "jpeg_relative_path"]].drop_duplicates(subset=["PatientName"])
    lookup = lookup.rename(columns={"PatientName": "image_id"})
    return lookup


def binarize_pathology(pathology: str) -> str:
    return "malignant" if str(pathology).strip().upper() == "MALIGNANT" else "benign"


def build_image_label_table(csv_dir: Path, split: str, lookup: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for filename in CASE_DESCRIPTION_FILES[split]:
        path = csv_dir / filename
        if not path.exists():
            print(f"  [aviso] arquivo não encontrado, pulando: {path}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]  # normaliza 'breast density' vs 'breast_density'
        df["image_id"] = df["image file path"].str.split("/").str[0]
        df["label"] = df["pathology"].apply(binarize_pathology)
        frames.append(df[["image_id", "label"]])

    if not frames:
        return pd.DataFrame(columns=["image_id", "label", "jpeg_relative_path"])

    combined = pd.concat(frames, ignore_index=True)

    # Uma mesma imagem pode ter múltiplas anormalidades (linhas). Regra
    # conservadora: se qualquer uma for maligna, a imagem é 'malignant'.
    def worst_case(labels: pd.Series) -> str:
        return "malignant" if (labels == "malignant").any() else "benign"

    per_image = combined.groupby("image_id")["label"].apply(worst_case).reset_index()
    per_image = per_image.merge(lookup, on="image_id", how="left")

    n_unmatched = per_image["jpeg_relative_path"].isna().sum()
    if n_unmatched:
        print(f"  [aviso] {n_unmatched} imagens não encontradas em dicom_info.csv (serão ignoradas)")
    per_image = per_image.dropna(subset=["jpeg_relative_path"])

    return per_image


def copy_images(per_image: pd.DataFrame, cbis_root: Path, split: str) -> None:
    for label in ("benign", "malignant"):
        (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

    counts = {"benign": 0, "malignant": 0, "faltando_no_disco": 0}
    for _, row in per_image.iterrows():
        src = cbis_root / row["jpeg_relative_path"]
        if not src.exists():
            counts["faltando_no_disco"] += 1
            continue
        dest_name = f"{row['image_id']}.jpg"
        dest = OUTPUT_DIR / split / row["label"] / dest_name
        shutil.copyfile(src, dest)
        counts[row["label"]] += 1

    print(f"  {split}: {counts['benign']} benignas, {counts['malignant']} malignas "
          f"copiadas ({counts['faltando_no_disco']} arquivos não encontrados no disco)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cbis-root", required=True,
        help="Caminho para a pasta baixada pelo kagglehub "
             "(contém as subpastas 'csv' e 'jpeg'). Ex.: "
             r"C:\Users\pedro\.cache\kagglehub\datasets\awsaf49\cbis-ddsm-breast-cancer-image-dataset\versions\1",
    )
    args = parser.parse_args()

    cbis_root = Path(args.cbis_root)
    csv_dir = cbis_root / "csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"Pasta 'csv' não encontrada em {cbis_root}")

    print("Carregando lookup de mamografias completas (dicom_info.csv)...")
    lookup = load_full_mammogram_lookup(csv_dir)
    print(f"  {len(lookup)} mamografias completas indexadas.")

    for split in ("train", "test"):
        print(f"\nProcessando split '{split}'...")
        per_image = build_image_label_table(csv_dir, split, lookup)
        print(f"  {len(per_image)} imagens únicas rotuladas (mass + calc, benign/malignant).")
        copy_images(per_image, cbis_root, split)

    print(f"\nConcluído. Dados organizados em: {OUTPUT_DIR}")
    print("Agora você pode rodar: python src/cnn_mammography.py")


if __name__ == "__main__":
    main()
