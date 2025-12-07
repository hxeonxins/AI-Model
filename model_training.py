"""모델 정의와 학습 유틸리티."""
import os
import datetime
from typing import Tuple
import tensorflow as tf
from tensorflow import keras


def build_cnn_model(image_size: Tuple[int, int], num_classes: int, dropout_rate: float = 0.3) -> keras.Model:
    inputs = keras.Input(shape=(*image_size, 3))
    x = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="simple_cnn")


def build_resnet_model(
    image_size: Tuple[int, int],
    num_classes: int,
    fine_tune: bool = False,
    base_trainable_layers: int = 20,
    dropout_rate: float = 0.3,
) -> keras.Model:
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
    )
    base_model.trainable = fine_tune
    if fine_tune and base_trainable_layers > 0:
        for layer in base_model.layers[:-base_trainable_layers]:
            layer.trainable = False

    inputs = keras.Input(shape=(*image_size, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="resnet_transfer")


def compile_model(model: keras.Model, learning_rate: float, optimizer_name: str = "adam") -> keras.Model:
    optimizers = {
        "adam": keras.optimizers.Adam(learning_rate),
        "rmsprop": keras.optimizers.RMSprop(learning_rate),
        "sgd": keras.optimizers.SGD(learning_rate, momentum=0.9),
    }
    optimizer = optimizers.get(optimizer_name.lower(), optimizers["adam"])
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def get_callbacks(log_dir: str, checkpoint_dir: str, patience: int = 5):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=log_dir),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]
    return callbacks


def train_model(
    model_name: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    image_size: Tuple[int, int],
    num_classes: int,
    learning_rate: float = 1e-3,
    epochs: int = 15,
    optimizer_name: str = "adam",
    fine_tune: bool = False,
    base_trainable_layers: int = 20,
    log_root: str = "logs",
    save_root: str = "saved_models",
):
    if model_name == "cnn":
        model = build_cnn_model(image_size, num_classes)
    elif model_name == "resnet":
        model = build_resnet_model(
            image_size, num_classes, fine_tune=fine_tune, base_trainable_layers=base_trainable_layers
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = compile_model(model, learning_rate, optimizer_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_root, f"{model_name}_{timestamp}")
    checkpoint_dir = os.path.join(save_root, f"{model_name}_{timestamp}")

    callbacks = get_callbacks(log_dir, checkpoint_dir)

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    eval_results = model.evaluate(val_ds, verbose=0)
    return model, history, eval_results, log_dir, checkpoint_dir


if __name__ == "__main__":
    print("이 모듈은 main.py에서 사용됩니다.")
