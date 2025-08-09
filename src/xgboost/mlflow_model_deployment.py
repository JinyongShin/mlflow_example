"""
MLflow로 저장된 XGBoost 모델을 REST API로 배포하는 예제
"""

import mlflow
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from waitress import serve
import json

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 전역 변수로 모델, 클래스명, 특성명 선언
model = None
class_names = None
feature_names = None

def load_model():
    """
    MLflow 모델 레지스트리에서 최신 모델을 로드합니다.
    """
    global model, class_names, feature_names
    
    # 모델 로드
    model_name = "wine_xgboost_classifier"  # 또는 "wine_xgboost_classifier_optimized"
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        print(f"모델 '{model_name}'의 최신 버전을 성공적으로 로드했습니다.")
        
        # Wine 데이터셋의 클래스명과 특성명 설정
        from sklearn.datasets import load_wine
        wine = load_wine()
        class_names = wine.target_names
        
        # 모델 시그니처 확인
        if hasattr(model, 'metadata') and hasattr(model.metadata, 'signature'):
            print("모델 시그니처가 확인되었습니다.")
            if hasattr(model.metadata.signature, 'inputs'):
                print("입력 특성:", model.metadata.signature.inputs.input_names())
        else:
            print("경고: 모델에 시그니처가 없습니다.")
        
        return True
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("아직 모델을 학습하지 않았거나 모델 이름이 다를 수 있습니다.")
        print("먼저 'src/xgboost/mlflow_example.py' 또는 'src/xgboost/mlflow_hyperparameter_tuning.py'를 실행하여 모델을 학습하세요.")
        return False

@app.route('/')
def home():
    """
    홈 엔드포인트는 API 사용 방법에 대한 간단한 안내를 제공합니다.
    """
    return """
    <h1>Wine XGBoost 분류 모델 API</h1>
    <p>이 API는 와인 데이터에 대한 XGBoost 분류 모델을 제공합니다.</p>
    <h2>API 사용 방법:</h2>
    <ul>
        <li><strong>GET /health</strong>: API 상태 확인</li>
        <li><strong>POST /predict</strong>: 예측 수행 (JSON 요청 필요)</li>
        <li><strong>GET /metadata</strong>: 모델 메타데이터 조회</li>
    </ul>
    
    <h2>예측 요청 예시 (POST /predict):</h2>
    <pre>
    {
        "data": [
            [12.82, 3.37, 2.3, 19.5, 88.0, 1.48, 0.66, 0.4, 0.97, 10.26, 0.72, 1.75, 685.0],
            [12.42, 2.55, 2.27, 22.0, 90.0, 1.68, 0.03, 1.1, 0.53, 2.3, 0.6, 1.3, 680.0]
        ]
    }
    </pre>
    """

@app.route('/health', methods=['GET'])
def health():
    """
    API 상태 확인 엔드포인트
    """
    if model is None:
        return jsonify({
            "status": "error",
            "message": "모델이 로드되지 않았습니다."
        }), 503
    
    return jsonify({
        "status": "ok",
        "message": "API가 정상 작동 중입니다."
    })

@app.route('/metadata', methods=['GET'])
def metadata():
    """
    모델 메타데이터 조회 엔드포인트
    """
    if model is None:
        return jsonify({
            "status": "error",
            "message": "모델이 로드되지 않았습니다."
        }), 503
    
    # Wine 데이터셋의 특성명 가져오기
    from sklearn.datasets import load_wine
    wine = load_wine()
    feature_names = wine.feature_names
    
    metadata = {
        "model_type": "XGBoost Classifier",
        "framework": "MLflow & XGBoost",
        "classes": class_names.tolist() if hasattr(class_names, "tolist") else class_names,
        "features": feature_names.tolist() if hasattr(feature_names, "tolist") else feature_names,
        "model_info": {
            "model_uri": model.metadata.model_uri,
            "flavor": model.metadata.flavors["python_function"]["loader_module"]
        }
    }
    
    return jsonify(metadata)

@app.route('/predict', methods=['POST'])
def predict():
    """
    예측 수행 엔드포인트
    """
    if model is None:
        return jsonify({
            "status": "error",
            "message": "모델이 로드되지 않았습니다."
        }), 503
    
    # 요청 데이터 검증
    if not request.json or 'data' not in request.json:
        return jsonify({
            "status": "error",
            "message": "유효하지 않은 요청 형식입니다. 'data' 필드가 필요합니다."
        }), 400
    
    try:
        # 요청 데이터를 pandas DataFrame으로 변환
        data = request.json['data']
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "예측할 데이터가 없습니다."
            }), 400
        
        # 모델 시그니처에서 특성명 가져오기 시도
        model_feature_names = None
        if hasattr(model, 'metadata') and hasattr(model.metadata, 'signature') and hasattr(model.metadata.signature, 'inputs'):
            try:
                model_feature_names = model.metadata.signature.inputs.input_names()
                print(f"모델 시그니처에서 특성명을 가져왔습니다: {model_feature_names}")
            except:
                print("모델 시그니처에서 특성명을 가져오는데 실패했습니다.")
        
        # 시그니처에서 가져오지 못한 경우 Wine 데이터셋에서 가져오기
        if not model_feature_names:
            from sklearn.datasets import load_wine
            wine = load_wine()
            model_feature_names = wine.feature_names
            print(f"Wine 데이터셋에서 특성명을 가져왔습니다: {model_feature_names}")
        
        # 입력 데이터를 DataFrame으로 변환
        if isinstance(data[0], list):  # 2D 리스트인 경우
            df = pd.DataFrame(data, columns=model_feature_names)
        else:  # 1D 리스트인 경우
            df = pd.DataFrame([data], columns=model_feature_names)
        
        # 예측 수행
        predictions = model.predict(df)
        
        # 다중 클래스 확률을 클래스 인덱스로 변환
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # 확률 배열이 반환된 경우
            probabilities = predictions
            predicted_classes = np.argmax(predictions, axis=1)
        else:
            # 이미 클래스 인덱스가 반환된 경우
            predicted_classes = predictions.astype(int)
            probabilities = None
        
        # 응답 구성
        response = []
        for i, pred_class in enumerate(predicted_classes):
            result = {
                "predicted_class": int(pred_class),
                "predicted_class_name": class_names[pred_class]
            }
            
            # 확률이 있는 경우 추가
            if probabilities is not None:
                probs_dict = {
                    class_names[j]: float(probabilities[i, j])
                    for j in range(len(class_names))
                }
                result["probabilities"] = probs_dict
            
            response.append(result)
        
        return jsonify({
            "status": "success",
            "predictions": response
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"예측 중 오류 발생: {str(e)}"
        }), 500

def main():
    """
    애플리케이션 메인 함수
    """
    print("===== XGBoost 모델 배포 서버 시작 =====")
    
    # 모델 로드
    if not load_model():
        print("모델 로드 실패로 서버를 시작할 수 없습니다.")
        return
    
    # 서버 호스트 및 포트 설정
    host = '0.0.0.0'
    port = 5001  # scikit-learn 예제와 포트 충돌을 피하기 위해 다른 포트 사용
    
    print(f"\n서버가 http://{host}:{port} 에서 실행 중입니다...")
    print("다음 URL로 API에 접근할 수 있습니다:")
    print(f"- API 홈: http://localhost:{port}/")
    print(f"- 상태 확인: http://localhost:{port}/health")
    print(f"- 모델 메타데이터: http://localhost:{port}/metadata")
    print("\n예측 요청 예시 (curl 명령어):")
    print(f"""curl -X POST http://localhost:{port}/predict \\
    -H "Content-Type: application/json" \\
    -d '{{
        "data": [
            [12.82, 3.37, 2.3, 19.5, 88.0, 1.48, 0.66, 0.4, 0.97, 10.26, 0.72, 1.75, 685.0]
        ]
    }}'""")
    
    print("\nCtrl+C로 서버를 종료할 수 있습니다.")
    
    # 서버 시작 (waitress를 사용한 프로덕션 서버)
    serve(app, host=host, port=port)

if __name__ == "__main__":
    main()
