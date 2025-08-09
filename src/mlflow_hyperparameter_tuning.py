"""
MLflow와 함께 하이퍼파라미터 튜닝 수행하기
"""

import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 하이퍼파라미터 튜닝 범위 정의
def hyperparameter_tuning():
    """
    그리드 서치 방식으로 하이퍼파라미터 튜닝을 수행하고 MLflow로 모든 실험을 추적합니다.
    """
    # MLflow 실험 생성
    experiment_name = "iris-hyperparameter-tuning"
    mlflow.set_experiment(experiment_name)
    
    # Iris 데이터셋 로드
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    
    # 훈련 및 테스트 세트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 하이퍼파라미터 그리드 정의
    n_estimators_list = [50, 100, 200]
    max_depth_list = [5, 10, 15, None]
    min_samples_split_list = [2, 5, 10]
    
    best_accuracy = 0.0
    best_params = {}
    best_model = None
    
    # 하이퍼파라미터 조합에 대한 그리드 서치 수행
    total_combinations = len(n_estimators_list) * len(max_depth_list) * len(min_samples_split_list)
    print(f"총 {total_combinations}개의 하이퍼파라미터 조합을 평가합니다...")
    
    combination_count = 0
    
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_split in min_samples_split_list:
                combination_count += 1
                print(f"조합 {combination_count}/{total_combinations} 평가 중...")
                
                # 현재 하이퍼파라미터 조합
                params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "random_state": 42
                }
                
                # MLflow 실행 시작
                with mlflow.start_run(run_name=f"run_{combination_count}"):
                    # 하이퍼파라미터 로깅
                    mlflow.log_params(params)
                    
                    # 모델 생성
                    model = RandomForestClassifier(**params)
                    
                    # 교차 검증 수행
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    # 교차 검증 결과 로깅
                    mlflow.log_metric("cv_mean_accuracy", cv_mean)
                    mlflow.log_metric("cv_std_accuracy", cv_std)
                    
                    # 전체 훈련 세트로 모델 훈련
                    model.fit(X_train, y_train)
                    
                    # 테스트 세트에서 성능 평가
                    y_pred = model.predict(X_test)
                    test_accuracy = accuracy_score(y_test, y_pred)
                    
                    # 테스트 성능 로깅
                    mlflow.log_metric("test_accuracy", test_accuracy)
                    
                    # 최고 성능 모델 갱신
                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        best_params = params.copy()
                        best_model = model
                    
                    # 특성 중요도 계산 및 로깅
                    feature_importances = pd.Series(
                        model.feature_importances_, 
                        index=X.columns
                    ).sort_values(ascending=False)
                    
                    # 특성 중요도를 JSON으로 로깅
                    mlflow.log_dict(feature_importances.to_dict(), "feature_importances.json")
                    
                    print(f"  테스트 정확도: {test_accuracy:.4f}, CV 정확도: {cv_mean:.4f} (±{cv_std:.4f})")
    
    # 최적 모델 정보 출력
    print("\n하이퍼파라미터 튜닝 완료!")
    print(f"최적 하이퍼파라미터: {best_params}")
    print(f"최고 테스트 정확도: {best_accuracy:.4f}")
    
    # 최적 모델을 별도 실행으로 로깅 및 등록
    with mlflow.start_run(run_name="best_model"):
        # 하이퍼파라미터 로깅
        mlflow.log_params(best_params)
        
        # 성능 지표 로깅
        mlflow.log_metric("test_accuracy", best_accuracy)
        
        # 모델 서명(signature) 생성
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        # 최적 모델 로깅
        mlflow.sklearn.log_model(
            best_model, 
            "best_random_forest_model",
            signature=signature,
            input_example=X_train.iloc[:5]  # 입력 예시 제공
        )
        
        # 모델을 MLflow 모델 레지스트리에 등록
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_random_forest_model"
        model_name = "iris_classifier_optimized"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        print(f"최적 모델이 성공적으로 등록되었습니다: {model_name}, 버전: {registered_model.version}")

if __name__ == "__main__":
    hyperparameter_tuning()
    
    # MLflow UI 접속 방법 안내
    print("\nMLflow UI를 확인하려면 터미널에서 다음 명령어를 실행하세요:")
    print("mlflow ui")
    print("그리고 브라우저에서 http://localhost:5000 을 방문하세요.\n")
