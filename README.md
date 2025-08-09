# MLflow 실습 프로젝트

이 프로젝트는 MLflow를 활용한 머신러닝 모델 개발, 추적, 관리 및 배포의 전체 워크플로우를 학습하기 위한 실습 코드를 포함하고 있습니다.

## 프로젝트 구성

- `mlflow_example.py`: 기본적인 MLflow 실험 추적 및 모델 등록 예제
- `mlflow_model_serving.py`: 등록된 MLflow 모델을 로드하고 추론하는 예제
- `mlflow_hyperparameter_tuning.py`: MLflow를 활용한 하이퍼파라미터 튜닝 예제
- `mlflow_model_deployment.py`: MLflow 모델을 Flask 웹 서비스로 배포하는 예제

## 설치 방법

1. 필요한 패키지 설치:

```bash
uv sync
```

## 사용 방법

### 1. 기본 모델 학습 및 추적

```bash
uv run mlflow_example.py
```

이 스크립트는 Iris 데이터셋에 대한 RandomForest 분류기를 학습하고, 모델 및 관련 메타데이터를 MLflow에 기록합니다.

### 2. 하이퍼파라미터 튜닝

```bash
uv run mlflow_hyperparameter_tuning.py
```

이 스크립트는 여러 하이퍼파라미터 조합을 시도하고, 각 실험 결과를 MLflow에 기록하여 최적의 모델을 찾습니다.

### 3. 모델 로딩 및 추론

```bash
uv run mlflow_model_serving.py
```

이 스크립트는 MLflow 모델 레지스트리에서 모델을 로드하고 테스트 데이터에 대한 예측을 수행합니다.

### 4. 모델 배포 (REST API)

```bash
uv run mlflow_model_deployment.py
```

이 스크립트는 등록된 모델을 Flask 웹 서버로 배포하여 REST API를 통해 예측 서비스를 제공합니다.

## MLflow UI 확인하기

학습 및 실험 결과를 시각적으로 확인하려면 MLflow UI를 실행하세요:

```bash
mlflow ui
```

그리고 웹 브라우저에서 http://localhost:5000 에 접속하세요.

### 5. 사용 flow 예시
MLflow를 사용하려면 다음과 같은 단계로 진행하면 됩니다:

* 먼저 `mlflow_example.py` 를 실행하여 기본 모델을 학습하고 등록

* `mlflow ui` 명령어로 MLflow UI를 실행하여 학습 결과 확인

* `mlflow_hyperparameter_tuning.py` 로 최적의 모델 찾기

* `mlflow_model_serving.py` 로 모델 로딩 및 추론 테스트

* `mlflow_model_deployment.py` 로 모델을 API 서비스로 배포

이 예제들을 통해 MLflow의 주요 기능인 실험 추적, 모델 관리, 모델 배포 프로세스를 모두 경험해볼 수 있습니다.

## MLflow의 주요 기능

1. **실험 추적 (Experiment Tracking)**
   - 실험 매개변수, 메트릭, 모델 및 아티팩트 추적
   - 실험 간 비교 및 시각화

2. **모델 레지스트리 (Model Registry)**
   - 모델 버전 관리
   - 모델 스테이지 관리 (개발, 스테이징, 프로덕션)
   - 모델 계보 및 메타데이터 관리

3. **모델 서빙 (Model Serving)**
   - REST API를 통한 모델 배포
   - 배치 추론 지원

4. **모델 패키징 (Model Packaging)**
   - 다양한 ML 프레임워크 지원 (scikit-learn, TensorFlow, PyTorch 등)
   - 일관된 모델 형식 및 인터페이스

## 참고 자료

- [MLflow 공식 문서](https://mlflow.org/docs/latest/index.html)
- [MLflow GitHub 저장소](https://github.com/mlflow/mlflow)
