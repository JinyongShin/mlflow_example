"""
MLflow를 사용한 모델 배포 및 REST API 서빙 예제
"""

import os
import mlflow.pyfunc
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# 모델 이름과 스테이지 설정
MODEL_NAME = "iris_classifier"
MODEL_STAGE = "Production"  # 'None', 'Staging', 'Production', 'Archived' 중 하나

# 앱 초기화
app = Flask(__name__)

# 글로벌 변수로 모델 선언
model = None

def load_model():
    """
    MLflow 모델 레지스트리에서 모델을 로드합니다.
    """
    global model
    
    try:
        # 특정 스테이지의 모델 URI 구성
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"모델 '{model_uri}'를 로드합니다...")
        
        # MLflow 모델 로드
        model = mlflow.pyfunc.load_model(model_uri)
        print("모델이 성공적으로 로드되었습니다.")
        
        return True
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        print(f"참고: 모델 '{MODEL_NAME}'이 모델 레지스트리에 등록되어 있는지, "
              f"그리고 '{MODEL_STAGE}' 스테이지에 모델이 배포되어 있는지 확인하세요.")
        return False

@app.route('/health', methods=['GET'])
def health():
    """헬스 체크 엔드포인트"""
    return jsonify({"status": "healthy", "model": MODEL_NAME, "stage": MODEL_STAGE})

@app.route('/predict', methods=['POST'])
def predict():
    """
    JSON 입력을 받아 예측 결과를 반환합니다.
    
    요청 형식:
    {
        "sepal length (cm)": [5.1, 6.2, ...],
        "sepal width (cm)": [3.5, 2.9, ...],
        "petal length (cm)": [1.4, 4.3, ...],
        "petal width (cm)": [0.2, 1.3, ...]
    }
    
    응답 형식:
    {
        "predictions": [0, 1, 2, ...],
        "class_names": ["setosa", "versicolor", "virginica", ...],
        "probabilities": [[0.9, 0.1, 0.0], ...]  # 옵션
    }
    """
    if model is None:
        return jsonify({"error": "모델이 로드되지 않았습니다."}), 503
    
    try:
        # 요청 데이터 파싱
        data = request.get_json()
        
        # 입력 검증
        required_features = ["sepal length (cm)", "sepal width (cm)", 
                             "petal length (cm)", "petal width (cm)"]
        
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"필수 특성 '{feature}'이(가) 누락되었습니다."}), 400
        
        # DataFrame 생성
        input_data = pd.DataFrame(data)
        
        # 예측 수행
        predictions = model.predict(input_data)
        
        # 클래스 이름 매핑
        class_names = {
            0: "setosa",
            1: "versicolor",
            2: "virginica"
        }
        
        # 결과를 클래스 이름으로 변환
        class_predictions = [class_names[int(pred)] for pred in predictions]
        
        # 결과 반환
        response = {
            "predictions": predictions.tolist(),
            "class_names": class_predictions
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def promote_model_to_production(model_name, version):
    """
    모델을 Production 스테이지로 승격시킵니다.
    
    Args:
        model_name (str): 모델 이름
        version (int): 모델 버전
    """
    try:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        print(f"모델 '{model_name}' 버전 {version}을(를) Production 스테이지로 승격했습니다.")
    except Exception as e:
        print(f"모델 승격 중 오류 발생: {e}")

if __name__ == "__main__":
    # 모델 로드
    if load_model():
        # 서버 시작
        port = int(os.environ.get('PORT', 5001))
        print(f"서버가 http://localhost:{port}에서 실행 중입니다.")
        print("예측을 요청하려면 다음과 같이 curl을 사용하세요:")
        print("""
curl -X POST http://localhost:5001/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "sepal length (cm)": [5.1, 6.3, 7.0],
    "sepal width (cm)": [3.5, 2.9, 3.2],
    "petal length (cm)": [1.4, 5.6, 4.7],
    "petal width (cm)": [0.2, 1.8, 1.4]
}'
        """)
        
        app.run(host='0.0.0.0', port=port)
    else:
        print("서버를 시작할 수 없습니다. 모델 로드에 실패했습니다.")
        print("\n모델을 Production 스테이지로 승격시키려면 다음과 같이 실행하세요:")
        print("python -c \"import mlflow_model_deployment; mlflow_model_deployment.promote_model_to_production('iris_classifier', 1)\"")
