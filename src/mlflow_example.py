"""
MLflow 실습 예제: 간단한 머신러닝 모델 학습 및 추적
"""

# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# MLflow 실험 이름 설정
mlflow.set_experiment("iris-classification")

# Iris 데이터셋 로드
iris = load_iris(as_frame=True)
X = iris.data  # 특성 데이터
y = iris.target  # 타겟 데이터

# 훈련 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 하이퍼파라미터 정의
params = {
    "n_estimators": 100,     # 결정 트리의 개수
    "max_depth": 5,          # 트리의 최대 깊이
    "random_state": 42,      # 재현성을 위한 랜덤 시드
    "min_samples_split": 2,  # 내부 노드를 분할하는 데 필요한 최소 샘플 수
    "min_samples_leaf": 1    # 리프 노드에 있어야 하는 최소 샘플 수
}

# MLflow 실행 시작 및 모델 학습
with mlflow.start_run() as run:
    # 실행 시작 메시지
    print(f"MLflow 실행 시작: {run.info.run_id}")
    
    # 하이퍼파라미터 로깅
    mlflow.log_params(params)
    
    # 모델 생성 및 학습
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # 모델 예측 및 평가
    y_pred = model.predict(X_test)
    
    # 평가 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # 평가 지표 로깅
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # 특성 중요도 계산 및 로깅
    feature_importances = pd.Series(
        model.feature_importances_, 
        index=X.columns
    ).sort_values(ascending=False)
    
    # 특성 중요도를 JSON으로 로깅
    mlflow.log_dict(feature_importances.to_dict(), "feature_importances.json")
    
    # 모델 서명(signature) 생성
    signature = infer_signature(X_train, model.predict(X_train))
    
    # 모델 로깅
    mlflow.sklearn.log_model(
        model, 
        "random_forest_model",
        signature=signature,
        input_example=X_train.iloc[:5]  # 입력 예시 제공
    )
    
    # 모델을 MLflow 모델 레지스트리에 등록
    model_uri = f"runs:/{run.info.run_id}/random_forest_model"
    model_name = "iris_classifier"
    registered_model = mlflow.register_model(model_uri, model_name)
    
    print(f"모델이 성공적으로 등록되었습니다: {model_name}, 버전: {registered_model.version}")
    print(f"정확도: {accuracy:.4f}")
    print(f"F1 점수: {f1:.4f}")

# MLflow UI 접속 방법 안내
print("\nMLflow UI를 확인하려면 터미널에서 다음 명령어를 실행하세요:")
print("mlflow ui")
print("그리고 브라우저에서 http://localhost:5000 을 방문하세요.\n")
