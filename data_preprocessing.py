"""데이터 로드와 전처리/증강 파이프라인."""
import numpy as np
import tensorflow as tf
from typing import Tuple, List

CLASS_ID_MAP = {0: 0, 1: 1, 9: 2}  # airplane, automobile, truck
CLASS_NAMES: List[str] = ["airplane", "automobile", "truck"]


def _filter_and_remap(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """CIFAR-10에서 3개 클래스만 추출하고 라벨을 0~2로 리매핑."""
    y = y.flatten()
    mask = np.isin(y, list(CLASS_ID_MAP.keys()))
    x_filtered = x[mask]
    y_filtered = y[mask]
    remapped = np.vectorize(CLASS_ID_MAP.get)(y_filtered)
    return x_filtered, remapped


def _build_augmentation(kinds: List[str]) -> tf.keras.Sequential:
    """선택된 증강 종류만 적용."""
    layers = []
    if "flip" in kinds:
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    if "rotation" in kinds:
        layers.append(tf.keras.layers.RandomRotation(0.1))
    if "brightness" in kinds:
        layers.append(tf.keras.layers.RandomBrightness(factor=0.1))
    return tf.keras.Sequential(layers, name="data_augmentation")


def build_datasets(
    batch_size: int = 64,
    image_size: Tuple[int, int] = (96, 96),
    val_split: float = 0.2,
    seed: int = 42,
    augmentations: Tuple[str, ...] = ("flip", "rotation", "brightness"),
    sample_fraction: float = 1.0,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """CIFAR-10에서 3개 클래스만 추출해 학습/검증 세트 생성."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    x, y = _filter_and_remap(x, y)

    # 셔플 후 분할
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    x, y = x[indices], y[indices]

    if sample_fraction < 1.0:
        take_n = max(1, int(len(x) * sample_fraction))
        x, y = x[:take_n], y[:take_n]

    val_size = int(len(x) * val_split)
    x_val, y_val = x[:val_size], y[:val_size]
    x_train, y_train = x[val_size:], y[val_size:]

    use_aug = augmentations is not None and len(augmentations) > 0
    data_augmentation = _build_augmentation(list(augmentations)) if use_aug else None

    def _preprocess(image: tf.Tensor, label: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        if training and data_augmentation is not None:
            image = data_augmentation(image, training=True)
        return image, label

    def _make_ds(images: np.ndarray, labels: np.ndarray, training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if training:
            ds = ds.shuffle(buffer_size=2000, seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda img, lbl: _preprocess(img, lbl, training), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _make_ds(x_train, y_train, training=True)
    val_ds = _make_ds(x_val, y_val, training=False)
    return train_ds, val_ds, CLASS_NAMES


if __name__ == "__main__":
    train_ds, val_ds, class_names = build_datasets()
    batch = next(iter(train_ds))
    print("Train batches:", len(train_ds))
    print("Val batches:", len(val_ds))
    print("Classes:", class_names)
