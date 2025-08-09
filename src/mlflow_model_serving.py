"""
MLflow 모델 로딩 및 추론 예제
"""

import mlflow
import pandas as pd
from sklearn.datasets import load_iris

# MLflow 모델 불러오기 및 추론 파이프라인
def load_and_predict(model_name, model_version, data):
    """
    MLflow 모델 레지스트리에서 모델을 로드하고 예측을 수행합니다.
    
    Args:
        model_name (str): 모델 이름
        model_version (str): 모델 버전 (또는 'latest')
        data (pandas.DataFrame): 예측할 입력 데이터
    
    Returns:
        array: 예측 결과
    """
    # 모델 URI 구성 - 모델 레지스트리에서 특정 버전 또는 최신 버전 로드
    if model_version == 'latest':
        model_uri = f"models:/{model_name}/latest"
    else:
        model_uri = f"models:/{model_name}/{model_version}"
    
    # 모델 로드
    print(f"모델 '{model_uri}'를 로드합니다...")
    model = mlflow.pyfunc.load_model(model_uri)
    
    # 예측 수행
    print("예측을 수행합니다...")
    predictions = model.predict(data)
    
    return predictions

# 예측 결과를 클래스 이름으로 변환
def get_class_names(predictions):
    """
    예측된 클래스 인덱스를 Iris 클래스 이름으로 변환합니다.
    
    Args:
        predictions: 예측된 클래스 인덱스
    
    Returns:
        list: 클래스 이름 목록
    """
    # Iris 데이터셋의 클래스 이름 매핑
    class_names = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }
    
    return [class_names[pred] for pred in predictions]

if __name__ == "__main__":
    try:
        # 테스트 데이터 준비
        iris = load_iris(as_frame=True)
        test_data = iris.data.iloc[:10]  # 처음 10개 샘플만 사용
        actual_targets = iris.target[:10]
        
        print("테스트 데이터:")
        print(test_data.head())
        
        # 모델 로드 및 예측 수행
        model_name = "iris_classifier"  # mlflow_example.py에서 등록한 모델 이름
        predictions = load_and_predict(model_name, 'latest', test_data)
        
        # 예측 결과 변환 및 표시
        class_names = get_class_names(predictions)
        actual_class_names = get_class_names(actual_targets)
        
        # 결과 표시
        results = pd.DataFrame({
            'Actual': actual_class_names,
            'Predicted': class_names
        })
        
        print("\n예측 결과:")
        print(results)
        
        # 정확도 계산
        accuracy = sum(pred == actual for pred, actual in zip(predictions, actual_targets)) / len(predictions)
        print(f"\n테스트 세트 정확도: {accuracy:.4f}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("\n참고: 이 스크립트를 실행하기 전에 먼저 mlflow_example.py를 실행하여 모델을 학습하고 등록해야 합니다.")
