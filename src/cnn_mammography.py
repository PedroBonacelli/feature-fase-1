"""
[EXTRA] Visão computacional — CNN para diagnóstico via mamografia (CBIS-DDSM)

IMPORTANTE — leia antes de rodar:
    Este script requer TensorFlow/Keras e o dataset CBIS-DDSM baixado
    localmente. Ele foi desenvolvido e revisado para ser executado em um
    ambiente com GPU (Google Colab ou máquina local) — NÃO foi executado no
    ambiente sandbox usado para o restante deste projeto, pois esse ambiente
    não tem acesso à internet para baixar o dataset do Kaggle nem permite
    instalar TensorFlow (índice de pacotes restrito). Por isso não há,
    nesta entrega, gráficos/métricas reais desta etapa — apenas o código,
    pronto para rodar. Ver `reports/03_cnn_extra.md` para mais detalhes
    sobre essa limitação e como executar.

Estrutura de dados esperada (ver instruções de download/organização em
data/raw/README.md):

    data/raw/cbis-ddsm/
        train/
            benign/     *.png
            malignant/  *.png
        test/
            benign/     *.png
            malignant/  *.png

Uso (após organizar os dados e instalar tensorflow):
    python src/cnn_mammography.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score, roc_curve
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Este script requer tensorflow e scikit-learn. Instale com:\n"
        "    pip install tensorflow scikit-learn\n"
        "Ele não foi executado no ambiente sandbox deste projeto — ver "
        "reports/03_cnn_extra.md para detalhes."
    ) from e

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "cbis-ddsm"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "reports" / "figures"

IMG_SIZE = (128, 128)     # tamanho reduzido para treinar rápido em CPU/GPU modesta
BATCH_SIZE = 32
SEED = 42
EPOCHS = 30
CLASS_NAMES = ["benign", "malignant"]  # ordem alfabética = ordem do Keras


def load_datasets():
    """Carrega os dados a partir da estrutura de pastas benign/malignant.

    Usa 15% do conjunto de treino como validação. Imagens em escala de
    cinza (mamografias não têm informação de cor relevante).
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=0.15, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale",
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=0.15, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale",
        label_mode="binary",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        color_mode="grayscale", label_mode="binary", shuffle=False,
    )

    # Normalização [0, 255] -> [0, 1]
    normalize = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalize(x), y))
    val_ds = val_ds.map(lambda x, y: (normalize(x), y))
    test_ds = test_ds.map(lambda x, y: (normalize(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_augmentation():
    """Data augmentation leve — mamografias não devem ser espelhadas
    verticalmente de forma agressiva nem giradas demais, para preservar
    a anatomia; usamos apenas flip horizontal, pequenas rotações e zoom."""
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")


def build_cnn(input_shape=(128, 128, 1)) -> tf.keras.Model:
    """CNN convolucional simples (treinada do zero) para classificação
    binária benigno/maligno a partir de recortes (patches) de mamografia.

    Arquitetura: 4 blocos conv-batchnorm-pool com número crescente de
    filtros, seguidos de GlobalAveragePooling (reduz overfitting em
    relação a Flatten + Dense grande) e uma cabeça densa com dropout.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = build_augmentation()(inputs)

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="cnn_mammography")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


def compute_class_weights(train_ds) -> dict:
    """Calcula pesos de classe para compensar eventual desbalanceamento
    (o CBIS-DDSM costuma ter mais casos benignos que malignos)."""
    labels = np.concatenate([y.numpy() for _, y in train_ds]).flatten()
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes.astype(int), weights))


def train(model, train_ds, val_ds, class_weight):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "cnn_mammography.keras"),
            monitor="val_auc", mode="max", save_best_only=True,
        ),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        class_weight=class_weight, callbacks=callbacks,
    )
    return history


def plot_training_curves(history) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric in zip(axes, ["loss", "accuracy", "auc"]):
        ax.plot(history.history[metric], label="treino")
        ax.plot(history.history[f"val_{metric}"], label="validação")
        ax.set_title(metric)
        ax.set_xlabel("época")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_cnn_training_curves.png", dpi=150)
    plt.close(fig)


def evaluate(model, test_ds) -> None:
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).flatten()
    y_proba = model.predict(test_ds).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    auc = roc_auc_score(y_true, y_proba)
    print(f"ROC AUC: {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusão — CNN mamografia")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_cnn_confusion_matrix.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"CNN (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Taxa de falso positivo")
    ax.set_ylabel("Taxa de verdadeiro positivo")
    ax.set_title("Curva ROC — CNN mamografia")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_cnn_roc.png", dpi=150)
    plt.close(fig)


def grad_cam(model, img_array, last_conv_layer_name="conv2d_3"):
    """Grad-CAM: gera um mapa de calor mostrando quais regiões da mamografia
    mais influenciaram a predição — o equivalente, em imagem, ao papel que o
    SHAP cumpre para os dados estruturados (etapa 6). Essencial para que o
    radiologista possa auditar visualmente onde o modelo 'olhou'."""
    grad_model = tf.keras.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {TRAIN_DIR}. Baixe e organize o "
            f"CBIS-DDSM conforme instruções em data/raw/README.md antes de "
            f"rodar este script."
        )

    train_ds, val_ds, test_ds = load_datasets()
    class_weight = compute_class_weights(train_ds)
    print(f"Pesos de classe (compensando desbalanceamento): {class_weight}")

    model = build_cnn(input_shape=IMG_SIZE + (1,))
    model.summary()

    history = train(model, train_ds, val_ds, class_weight)
    plot_training_curves(history)
    evaluate(model, test_ds)

    print(f"\nModelo salvo em: {MODELS_DIR / 'cnn_mammography.keras'}")
    print(f"Figuras salvas em: {FIG_DIR}")


if __name__ == "__main__":
    main()
