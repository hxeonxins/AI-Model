"""프로젝트 실행 진입점."""
import argparse
from pathlib import Path
import tensorflow as tf

from data_preprocessing import build_datasets, CLASS_NAMES
from model_training import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 소형 분류기 학습 스크립트")
    parser.add_argument("--model", choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "rmsprop", "sgd"], default="adam")
    parser.add_argument("--image_size", type=int, default=96, help="정사각형 입력 크기")
    parser.add_argument("--fine_tune", type=lambda v: str(v).lower() == "true", default=False,
                        help="전이학습 모델 파인튜닝 여부(ResNet 전용)")
    parser.add_argument("--base_trainable_layers", type=int, default=20,
                        help="파인튜닝 시 학습할 ResNet 하위 레이어 수")
    parser.add_argument("--augmentations", type=str, default="flip,rotation,brightness",
                        help="증강 종류 콤마 구분 (none 또는 flip,rotation,brightness)")
    parser.add_argument("--val_split", type=float, default=0.2, help="검증 비율 (0.2=8:2)")
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="데이터 샘플링 비율(1.0=전체, 0.1=10%만 사용)")
    return parser.parse_args()


def main():
    args = parse_args()
    image_size = (args.image_size, args.image_size)
    augmentations = tuple([a for a in args.augmentations.split(",") if a]) if args.augmentations.lower() != "none" else ()

    print("데이터 로드 및 전처리 중...")
    train_ds, val_ds, class_names = build_datasets(
        batch_size=args.batch_size,
        image_size=image_size,
        val_split=args.val_split,
        augmentations=augmentations,
        sample_fraction=args.sample_fraction,
    )

    print("모델 학습 시작...")
    model, history, eval_results, log_dir, checkpoint_dir = train_model(
        model_name=args.model,
        train_ds=train_ds,
        val_ds=val_ds,
        image_size=image_size,
        num_classes=len(class_names),
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        optimizer_name=args.optimizer,
        fine_tune=args.fine_tune,
        base_trainable_layers=args.base_trainable_layers,
    )

    val_loss, val_acc = eval_results
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")
    print(f"TensorBoard 로그: {log_dir}")
    print(f"모델 체크포인트: {checkpoint_dir}")

    # 최종 모델 저장 위치 안내 (checkpoint 외 추가 저장)
    final_dir = Path("saved_models") / "model_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"{args.model}_final.keras"
    model.save(final_path)
    print(f"최종 모델 저장 완료: {final_path}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
