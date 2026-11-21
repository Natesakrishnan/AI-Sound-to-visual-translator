"""
Module: Task 509 CNN Model Architecture
Exact reproduction of Step 391 (ESC50_ResNet_CNN).
Multi-Scale Residual 2D CNN with Dual Global Pooling (Avg + Max).
"""

import os
import random
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import layers, models, optimizers

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    """Deterministic seeding across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def build_audio_cnn(
    input_shape: tuple = (128, 216, 1),
    num_classes: int = 50,
    learning_rate: float = 0.0005
) -> keras.Model:
    """
    Construct the EXACT Multi-Scale Residual 2D CNN with Dual Global Pooling for ESC-50
    as defined in Task 509 (Step 391).
    """
    set_seed(RANDOM_SEED)

    inputs = layers.Input(shape=input_shape, name="mel_spectrogram_input")

    # Convolutional Block 1 (Output: 64 x 108 x 32)
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Residual Block 2 (Output: 32 x 54 x 64)
    res2 = layers.Conv2D(64, kernel_size=(1, 1), padding="same", name="res2_proj")(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.add([x, res2], name="res2_add")
    x = layers.Activation("relu", name="relu2_out")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Residual Block 3 (Output: 16 x 27 x 128)
    res3 = layers.Conv2D(128, kernel_size=(1, 1), padding="same", name="res3_proj")(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.add([x, res3], name="res3_add")
    x = layers.Activation("relu", name="relu3_out")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Residual Block 4 (Output: 8 x 13 x 256)
    res4 = layers.Conv2D(256, kernel_size=(1, 1), padding="same", name="res4_proj")(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="relu4_1")(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", name="conv4_2")(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.add([x, res4], name="res4_add")
    x = layers.Activation("relu", name="relu4_out")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

    # Dual Global Pooling: Average Pooling (stationary textures) + Max Pooling (sharp transients)
    gap = layers.GlobalAveragePooling2D(name="gap")(x)
    gmp = layers.GlobalMaxPooling2D(name="gmp")(x)
    concat = layers.concatenate([gap, gmp], name="concat_pools")  # 512 dimensions

    # Dense Classification Head with controlled regularization
    dense = layers.Dense(256, activation="relu", name="dense1")(concat)
    dense = layers.BatchNormalization(name="bn_dense")(dense)
    dense = layers.Dropout(0.25, name="drop_dense")(dense)

    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(dense)

    model = models.Model(inputs=inputs, outputs=outputs, name="ESC50_ResNet_CNN")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

