"""
Module 3: CNN-Based Audio Pattern Recognition Model
Implements a Multi-Scale Residual 2D Convolutional Neural Network with Dual Global Pooling
(Average + Max Pooling) using Keras 3 (PyTorch Backend).
Specifically engineered for high generalization on the ESC-50 dataset.
"""

import os
import random
import numpy as np

# Configure Keras backend to PyTorch before importing Keras
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import layers, models, optimizers, regularizers


RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    """Set global seeds for deterministic reproducibility with PyTorch backend without importing TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_audio_cnn(
    input_shape: tuple = (128, 216, 1),
    num_classes: int = 50,
    learning_rate: float = 0.0005,
    weight_decay: float = 1e-4
) -> keras.Model:
    """
    Construct a Multi-Scale Residual 2D CNN with Dual Global Pooling for ESC-50.

    Architecture Overview:
        Input: (128, 216, 1) Instance-Standardized Log-Mel Spectrogram (mean=0, std=1)
        Stem: Conv2D(32) -> BN -> ReLU -> Conv2D(32) -> BN -> ReLU -> MaxPool(2,2) -> Dropout(0.15)
        Block 1 (Residual): Conv2D(64, 3x3, stride=2) + Conv2D(64, 1x1, stride=2) shortcut -> ReLU -> Dropout(0.20)
        Block 2 (Residual): Conv2D(128, 3x3, stride=2) + Conv2D(128, 1x1, stride=2) shortcut -> ReLU -> Dropout(0.25)
        Block 3 (Residual): Conv2D(256, 3x3, stride=2) + Conv2D(256, 1x1, stride=2) shortcut -> ReLU -> Dropout(0.30)
        Dual Pooling: Concatenate(GlobalAvgPool, GlobalMaxPool) -> 512 dimensions
        Dense Head: Dense(256, ReLU, L2) -> BatchNorm -> Dropout(0.40) -> Dense(50, Softmax)

    Returns:
        Compiled Keras Model instance.
    """
    set_seed(RANDOM_SEED)

    inputs = layers.Input(shape=input_shape, name="mel_spectrogram_input")
    l2_reg = regularizers.l2(weight_decay)

    # 1. Stem: Initial Feature Extraction (Output: 64 x 108 x 32)
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", use_bias=False,
                      kernel_regularizer=l2_reg, name="stem_conv1")(inputs)
    x = layers.BatchNormalization(name="stem_bn1")(x)
    x = layers.Activation("relu", name="stem_relu1")(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", use_bias=False,
                      kernel_regularizer=l2_reg, name="stem_conv2")(x)
    x = layers.BatchNormalization(name="stem_bn2")(x)
    x = layers.Activation("relu", name="stem_relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="stem_pool")(x)
    x = layers.Dropout(0.15, name="stem_drop")(x)

    # 2. Residual Block 1 (Output: 32 x 54 x 64)
    res1 = layers.Conv2D(64, kernel_size=(1, 1), strides=(2, 2), padding="same", use_bias=False,
                         kernel_regularizer=l2_reg, name="res1_proj")(x)
    res1 = layers.BatchNormalization(name="res1_bn_proj")(res1)

    c1 = layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False,
                       kernel_regularizer=l2_reg, name="res1_conv1")(x)
    c1 = layers.BatchNormalization(name="res1_bn1")(c1)
    c1 = layers.Activation("relu", name="res1_relu1")(c1)
    c1 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
                       kernel_regularizer=l2_reg, name="res1_conv2")(c1)
    c1 = layers.BatchNormalization(name="res1_bn2")(c1)

    x = layers.add([c1, res1], name="res1_add")
    x = layers.Activation("relu", name="res1_out")(x)
    x = layers.Dropout(0.20, name="res1_drop")(x)

    # 3. Residual Block 2 (Output: 16 x 27 x 128)
    res2 = layers.Conv2D(128, kernel_size=(1, 1), strides=(2, 2), padding="same", use_bias=False,
                         kernel_regularizer=l2_reg, name="res2_proj")(x)
    res2 = layers.BatchNormalization(name="res2_bn_proj")(res2)

    c2 = layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False,
                       kernel_regularizer=l2_reg, name="res2_conv1")(x)
    c2 = layers.BatchNormalization(name="res2_bn1")(c2)
    c2 = layers.Activation("relu", name="res2_relu1")(c2)
    c2 = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
                       kernel_regularizer=l2_reg, name="res2_conv2")(c2)
    c2 = layers.BatchNormalization(name="res2_bn2")(c2)

    x = layers.add([c2, res2], name="res2_add")
    x = layers.Activation("relu", name="res2_out")(x)
    x = layers.Dropout(0.25, name="res2_drop")(x)

    # 4. Residual Block 3 (Output: 8 x 14 x 256)
    res3 = layers.Conv2D(256, kernel_size=(1, 1), strides=(2, 2), padding="same", use_bias=False,
                         kernel_regularizer=l2_reg, name="res3_proj")(x)
    res3 = layers.BatchNormalization(name="res3_bn_proj")(res3)

    c3 = layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False,
                       kernel_regularizer=l2_reg, name="res3_conv1")(x)
    c3 = layers.BatchNormalization(name="res3_bn1")(c3)
    c3 = layers.Activation("relu", name="res3_relu1")(c3)
    c3 = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
                       kernel_regularizer=l2_reg, name="res3_conv2")(c3)
    c3 = layers.BatchNormalization(name="res3_bn2")(c3)

    x = layers.add([c3, res3], name="res3_add")
    x = layers.Activation("relu", name="res3_out")(x)
    x = layers.Dropout(0.30, name="res3_drop")(x)

    # 5. Dual Global Pooling: Average Pooling (stationary background) + Max Pooling (sharp acoustic onsets)
    gap = layers.GlobalAveragePooling2D(name="gap")(x)
    gmp = layers.GlobalMaxPooling2D(name="gmp")(x)
    concat = layers.concatenate([gap, gmp], name="concat_pools")  # 512 dimensions

    # 6. Dense Classification Head with controlled regularization
    dense = layers.Dense(256, activation="relu", kernel_regularizer=l2_reg, name="dense1")(concat)
    dense = layers.BatchNormalization(name="bn_dense")(dense)
    dense = layers.Dropout(0.40, name="drop_dense")(dense)

    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(dense)

    model = models.Model(inputs=inputs, outputs=outputs, name="ESC50_ResNet_CNN")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    m = build_audio_cnn()
    m.summary()
    print("Multi-Scale Residual Audio CNN successfully built!")
