# 인공지능 모델링 수행평가 프로젝트

TensorFlow/Keras를 사용해 CIFAR-10 데이터셋에서 3개 교통수단 클래스(비행기, 자동차, 트럭)를 분류하는 예제 프로젝트입니다. 기본 CNN과 전이학습(ResNet50)을 포함하며, 데이터 전처리/학습/실행 스크립트를 분리해 구성했습니다.

## 구조
```
├── main.py                 # 학습/평가/추론 진입점
├── model_training.py       # 모델 정의와 학습 루프
├── data_preprocessing.py   # 데이터 로드 및 전처리/증강 파이프라인
├── experiments.py          # 보고서용 실험 배치 실행
├── requirements.txt        # 의존성 목록
├── saved_models/           # 모델 저장 디렉터리(버전별)
│   ├── model_v1/
│   ├── model_v2/
│   └── model_final/
├── logs/                   # TensorBoard 로그
└── README.md
```

## 빠른 시작
1. 가상환경 생성 후 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```
2. 학습 실행(기본 CNN):
   ```bash
   python main.py --model cnn --epochs 20
   ```
3. 전이학습(ResNet50) 실행:
   ```bash
   python main.py --model resnet --epochs 15 --fine_tune True
   ```
4. TensorBoard 확인:
   ```bash
   tensorboard --logdir logs
   ```
5. 보고서용 일괄 실험 실행 및 CSV 저장:
   ```bash
   python experiments.py
   ```
6. 빠른 샘플 학습(데이터 10%만, 1에포크 예시):
   ```bash
   python main.py --model cnn --epochs 1 --batch_size 256 --image_size 64 --augmentations none --sample_fraction 0.1
   ```

## 데이터셋
- CIFAR-10 내 3개 교통수단 클래스 사용: `airplane`, `automobile`, `truck`
- 학습/검증 8:2 분할, 이미지 96x96 리사이즈(기본), 0~1 정규화
- 증강: 좌우 반전, 랜덤 회전, 밝기 조절

## 요구사항 충족 체크
- CNN 2개 이상, Dropout 포함 (`model_training.py: build_cnn_model`)
- 전이학습/파인튜닝 옵션(`--model resnet`, `--fine_tune True`)
- 최소 3가지 최적화: EarlyStopping, ReduceLROnPlateau, 옵티마이저 변경, 증강 조합, 하이퍼파라미터 튜닝(학습률/배치/에포크)
- 학습/검증 분리(8:2)
- TensorBoard/체크포인트 저장 지원
- 증강 on/off 및 종류 선택(`--augmentations flip,rotation,brightness` 혹은 `--augmentations none`)
- 데이터 샘플링(`--sample_fraction`)으로 빠른 실험/리허설 가능

## 기타
- `saved_models/`와 `logs/`는 학습 시 자동 생성/갱신됩니다.
- 하이퍼파라미터는 `main.py` CLI 인자로 조정 가능합니다.
