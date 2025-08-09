"""
MLflow로 저장된 XGBoost 모델을 로드하고 추론하는 예제
"""

import mlflow
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def main():
    # Wine 데이터셋 로드 (테스트 데이터로 사용)
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target
    class_names = wine.target_names
    feature_names = wine.feature_names  # feature_names 명시적으로 미리 저장
    
    # 훈련 및 테스트 세트로 분할 (테스트 세트만 사용할 예정)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 테스트 데이터가 정확한 feature_names 순서와 이름을 가지도록 함
    X_test = X_test[feature_names]
    
    # 테스트 샘플 선택
    test_samples = X_test.iloc[:5]
    
    print("===== MLflow에서 XGBoost 모델 로드 및 추론 예제 =====")
    
    # 1. 모델 레지스트리에서 최신 버전의 모델 가져오기
    model_name = "wine_xgboost_classifier"
    try:
        latest_model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        print(f"\n1. 모델 '{model_name}'의 최신 버전을 성공적으로 로드했습니다.")
        
        # test_samples는 이미 올바른 특성 이름과 순서로 정렬되어 있음
        # 모델 예측을 위한 준비가 되어 있음
        
        # 일반 Python 함수로 추론
        predictions = latest_model.predict(test_samples)
        
        # 다중 클래스 확률을 클래스 인덱스로 변환
        if predictions.ndim > 1 and predictions.shape[1] > 1:  # 다중 클래스 확률인 경우
            predicted_classes = np.argmax(predictions, axis=1)
        else:  # 이미 클래스 인덱스인 경우
            predicted_classes = predictions.astype(int)
        
        # 결과 출력
        print("\n테스트 샘플에 대한 예측 결과:")
        for i, (pred_class, true_class) in enumerate(zip(predicted_classes, y_test.iloc[:5])):
            print(f"샘플 {i+1}: 예측 클래스 = {pred_class} ({class_names[pred_class]}), "
                  f"실제 클래스 = {true_class} ({class_names[true_class]})")
        
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("아직 모델을 학습하지 않았거나 모델 이름이 다를 수 있습니다.")
        print("먼저 'mlflow_example.py' 또는 'mlflow_hyperparameter_tuning.py'를 실행하여 모델을 학습하세요.")
        return
    
    # 2. 특정 모델 버전 불러오기 (버전 1을 예로 사용)
    try:
        specific_version = 1
        versioned_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{specific_version}")
        print(f"\n2. 모델 '{model_name}'의 버전 {specific_version}을 성공적으로 로드했습니다.")
        
        # 버전 모델로 예측 (오류 발생 가능성이 있으므로 별도 예외 처리 추가)
        try:
            # test_samples는 이미 올바른 순서로 정렬되어 있음
            versioned_predictions = versioned_model.predict(test_samples)
            print(f"버전 {specific_version} 모델로 예측 성공!")
        except Exception as predict_error:
            print(f"버전 {specific_version} 모델 예측 실패: {predict_error}")
    except Exception as e:
        print(f"특정 버전 모델 로드 실패: {e}")
    
    # 3. XGBoost 네이티브 형식으로 모델 로드
    try:
        native_model = mlflow.xgboost.load_model(f"models:/{model_name}/latest")
        print("\n3. 네이티브 XGBoost 형식으로 모델을 성공적으로 로드했습니다.")
        
        # XGBoost DMatrix 변환 - 모델이 기대하는 feature_names와 정확히 일치하도록 함
        dtest = xgb.DMatrix(test_samples.values, feature_names=feature_names)
        
        # 네이티브 XGBoost 추론
        native_predictions = native_model.predict(dtest)
        native_predicted_classes = np.argmax(native_predictions, axis=1)
        
        # 결과 출력
        print("\nXGBoost 네이티브 API를 사용한 예측 결과:")
        for i, (pred_class, true_class) in enumerate(zip(native_predicted_classes, y_test.iloc[:5])):
            print(f"샘플 {i+1}: 예측 클래스 = {pred_class} ({class_names[pred_class]}), "
                  f"실제 클래스 = {true_class} ({class_names[true_class]})")
        
    except Exception as e:
        print(f"네이티브 형식으로 모델 로드 실패: {e}")
    
    # 4. 실행 ID로 모델 로드 (MLflow 실행 ID를 알고 있는 경우)
    print("\n4. 실행 ID를 사용하여 모델 로드하기:")
    print("   특정 실행 ID를 사용하려면 MLflow UI에서 실행 ID를 확인하고 다음과 같이 사용하세요:")
    print("   mlflow.pyfunc.load_model('runs:/<run_id>/xgboost_model')")
    
    # 배치 추론 예시
    print("\n5. 배치 추론 예제:")
    try:
        # 이미 다른 모델을 테스트하는 데 실패했으면 이 부분을 건너뜁니다
        if 'latest_model' not in locals() or latest_model is None:
            print("   이전 모델 로드에 실패하여 배치 추론을 건너뜁니다.")
            raise Exception("이전 모델 로드 실패")
            
        # X_test는 이미 올바른 순서로 정렬되어 있음
        # 전체 테스트 데이터에 대한 예측
        batch_predictions = latest_model.predict(X_test)
        
        # 다중 클래스 확률을 클래스 인덱스로 변환
        if batch_predictions.ndim > 1 and batch_predictions.shape[1] > 1:
            batch_predicted_classes = np.argmax(batch_predictions, axis=1)
        else:
            batch_predicted_classes = batch_predictions.astype(int)
        
        # 정확도 계산
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_test, batch_predicted_classes)
        
        print(f"   배치 예측 정확도: {accuracy:.4f}")
        print("\n   분류 보고서:")
        print(classification_report(y_test, batch_predicted_classes, target_names=class_names))
    except Exception as e:
        print(f"   배치 추론 실패: {e}")

if __name__ == "__main__":
    main()
