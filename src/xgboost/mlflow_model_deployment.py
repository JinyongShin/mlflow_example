"""
MLflow로 저장된 XGBoost 모델을 REST API로 배포하는 고급 예제

이 스크립트는 MLflow 공식 가이드(https://mlflow.org/docs/latest/ml/traditional-ml/xgboost/guide/)를 기반으로
XGBoost 모델을 배포하고 서빙하는 고급 패턴을 구현합니다.

주요 기능:
- 모델 로드 및 배포를 위한 Flask API
- 배치 추론 및 설명 가능성
- 모델 성능 모니터링
- 실시간 예측 로깅
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from waitress import serve
import threading
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import matplotlib
matplotlib.use('Agg')  # 서버 환경에서 GUI 없이 실행하기 위해 필요
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask 애플리케이션 초기화
app = Flask(__name__)
CORS(app)  # CORS 활성화

# 전역 변수 선언
model = None
model_name = None
model_version = None
model_stage = None
class_names = None
feature_names = None
prediction_count = 0
start_time = time.time()
prediction_times = []
prediction_log = []
model_info = {}
shap_explainer = None

def load_model(model_name="wine_xgboost_classifier", version="latest", stage=None):
    """
    MLflow 모델 레지스트리에서 모델을 로드합니다.
    
    Args:
        model_name: 모델 이름
        version: 모델 버전 (문자열 또는 정수)
        stage: 모델 스테이지 ('Production', 'Staging', 'Archived', None)
    
    Returns:
        bool: 모델 로드 성공 여부
    """
    global model, model_version, model_stage, class_names, feature_names, model_info, shap_explainer
    
    try:
        # 버전 또는 스테이지를 기반으로 모델 URI 생성
        if stage:
            model_uri = f"models:/{model_name}/{stage}"
            logger.info(f"스테이지 '{stage}'의 모델 '{model_name}'을 로드합니다.")
        else:
            model_uri = f"models:/{model_name}/{version}"
            logger.info(f"버전 '{version}'의 모델 '{model_name}'을 로드합니다.")
        
        # MLflow에서 모델 로드
        model = mlflow.pyfunc.load_model(model_uri)
        
        # 모델 정보 가져오기
        client = MlflowClient()
        
        # 모델 세부 정보 가져오기
        if stage:
            latest_versions = client.get_latest_versions(model_name, stages=[stage])
            if latest_versions:
                model_version = latest_versions[0].version
                model_stage = latest_versions[0].current_stage
        else:
            if version == "latest":
                latest_versions = client.get_latest_versions(model_name)
                if latest_versions:
                    model_version = latest_versions[0].version
                    model_stage = latest_versions[0].current_stage
            else:
                model_version = version
                model_details = client.get_model_version(model_name, version)
                model_stage = model_details.current_stage
        
        # 모델 정보 저장
        model_info = {
            "name": model_name,
            "version": model_version,
            "stage": model_stage,
            "creation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "flavor": model.metadata.flavors.keys(),
        }
        
        logger.info(f"모델이 성공적으로 로드되었습니다. 버전: {model_version}, 스테이지: {model_stage}")
        
        # 모델 설명을 위한 특성명과 클래스명 설정
        try:
            # Wine 데이터셋에서 특성명과 클래스명 로드 (실제로는 모델 메타데이터에서 가져오는 것이 좋음)
            from sklearn.datasets import load_wine
            wine = load_wine()
            feature_names = wine.feature_names
            class_names = wine.target_names
            
            # 모델 시그니처에서 특성명 확인 시도
            if hasattr(model, 'metadata') and hasattr(model.metadata, 'signature'):
                if hasattr(model.metadata.signature, 'inputs'):
                    input_names = model.metadata.signature.inputs.input_names()
                    if input_names:
                        feature_names = input_names
                        logger.info(f"모델 시그니처에서 특성명을 가져왔습니다: {feature_names}")
            
            # 네이티브 XGBoost 모델 얻기 시도 (SHAP 설명을 위해)
            try:
                # MLflow 모델에서 XGBoost 모델 추출
                if "xgboost" in model.metadata.flavors:
                    xgb_model_path = model.metadata.flavors["xgboost"]["data"]
                    full_path = os.path.join(model.metadata.artifact_uri.replace("file://", ""), xgb_model_path)
                    
                    # XGBoost 모델 직접 로드
                    native_model = xgb.Booster()
                    native_model.load_model(full_path)
                    
                    # SHAP 설명기 초기화
                    shap_explainer = shap.TreeExplainer(native_model)
                    logger.info("SHAP 설명기가 초기화되었습니다.")
            except Exception as e:
                logger.warning(f"SHAP 설명기 초기화 실패: {e}")
                shap_explainer = None
            
            return True
        except Exception as e:
            logger.warning(f"특성명과 클래스명 설정 실패: {e}")
            # 기본값 설정
            feature_names = [f"feature_{i}" for i in range(13)]  # Wine 데이터셋 기준
            class_names = [f"class_{i}" for i in range(3)]       # Wine 데이터셋 기준
        return True
            
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return False

def preprocess_input(data, feature_names=None):
    """
    API 요청 데이터를 모델 입력에 적합한 형태로 변환합니다.
    
    Args:
        data: 입력 데이터 (리스트 또는 중첩 리스트)
        feature_names: 특성 이름 목록
    
    Returns:
        pandas.DataFrame: 모델 입력용 데이터프레임
    """
    try:
        # 다차원 리스트인지 확인 (배치 입력)
        is_batch = isinstance(data[0], list)
        
        if is_batch:
            # 2D 배열을 DataFrame으로 변환
            df = pd.DataFrame(data, columns=feature_names)
        else:
            # 단일 샘플을 DataFrame으로 변환
            df = pd.DataFrame([data], columns=feature_names)
        
        return df
    
    except Exception as e:
        logger.error(f"데이터 전처리 실패: {e}")
        raise ValueError(f"입력 데이터 형식이 올바르지 않습니다: {e}")

def format_prediction(y_pred, class_names=None):
    """
    모델 예측 결과를 API 응답에 적합한 형태로 변환합니다.
    
    Args:
        y_pred: 모델 예측 결과
        class_names: 클래스 이름 목록
    
    Returns:
        list: 형식화된 예측 결과
    """
    try:
        results = []
        
        # 예측값 형태 확인
        is_proba = y_pred.ndim > 1 and y_pred.shape[1] > 1
        
        for i, pred in enumerate(y_pred):
            if is_proba:
                # 확률 배열인 경우
                pred_class = np.argmax(pred)
                probabilities = {
                    class_names[j] if class_names else f"class_{j}": float(prob)
                    for j, prob in enumerate(pred)
                }
                
                result = {
                    "prediction_id": f"pred_{int(time.time())}_{i}",
                    "predicted_class": int(pred_class),
                    "predicted_class_name": class_names[pred_class] if class_names else f"class_{pred_class}",
                    "probabilities": probabilities
                }
            else:
                # 이미 클래스 인덱스인 경우
                pred_class = int(pred)
                result = {
                    "prediction_id": f"pred_{int(time.time())}_{i}",
                    "predicted_class": pred_class,
                    "predicted_class_name": class_names[pred_class] if class_names else f"class_{pred_class}"
                }
            
            results.append(result)
        
        return results
    
    except Exception as e:
        logger.error(f"예측 결과 형식화 실패: {e}")
        raise ValueError(f"예측 결과 처리 중 오류 발생: {e}")

def log_prediction(input_data, prediction, prediction_time):
    """
    예측 결과를 로깅합니다.
    
    Args:
        input_data: 입력 데이터
        prediction: 예측 결과
        prediction_time: 예측 소요 시간
    """
    global prediction_count, prediction_times, prediction_log
    
    prediction_count += 1
    prediction_times.append(prediction_time)
    
    # 최근 100개 예측만 저장
    if len(prediction_log) >= 100:
        prediction_log.pop(0)
    
    # 예측 로그 저장
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_id": prediction[0]["prediction_id"] if prediction else None,
        "input_shape": input_data.shape,
        "prediction_result": prediction[0]["predicted_class_name"] if prediction else None,
        "prediction_time_ms": prediction_time * 1000
    }
    
    prediction_log.append(log_entry)
    
    # 로그 기록
    if prediction_count % 100 == 0:
        avg_time = sum(prediction_times) / len(prediction_times)
        logger.info(f"총 {prediction_count}개 예측 처리, 평균 처리 시간: {avg_time*1000:.2f}ms")

def generate_shap_explanation(input_data):
    """
    SHAP 값을 이용해 예측에 대한 설명을 생성합니다.
    
    Args:
        input_data: 입력 데이터
    
    Returns:
        dict: 특성 기여도 및 설명 이미지 경로
    """
    if shap_explainer is None:
        return {"error": "SHAP 설명기가 초기화되지 않았습니다."}
    
    try:
        # SHAP 값 계산
        if isinstance(input_data, pd.DataFrame):
            data = xgb.DMatrix(input_data.values, feature_names=feature_names)
        else:
            data = xgb.DMatrix(input_data, feature_names=feature_names)
        
        shap_values = shap_explainer.shap_values(data)
        
        # 단일 샘플에 대한 워터폴 플롯 생성
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_explainer.expected_value, shap_values[0], feature_names=feature_names, show=False)
        
        # 이미지 저장
        explanation_image = f"shap_explanation_{int(time.time())}.png"
        plt.savefig(explanation_image)
        plt.close()
        
        # 특성 기여도 계산
        feature_contributions = {}
        for i, fname in enumerate(feature_names):
            feature_contributions[fname] = float(shap_values[0][i])
        
        # 결과 구성
        explanation = {
            "base_value": float(shap_explainer.expected_value),
            "feature_contributions": feature_contributions,
            "explanation_image": explanation_image
        }
        
        return explanation
        
    except Exception as e:
        logger.error(f"SHAP 설명 생성 실패: {e}")
        return {"error": f"설명 생성 중 오류 발생: {e}"}

@app.route('/')
def home():
    """
    홈 페이지를 제공하는 엔드포인트
    """
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>XGBoost 모델 배포 서버</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; max-width: 1000px; margin: 0 auto; }
            h1 { color: #4285f4; }
            h2 { color: #34a853; margin-top: 30px; }
            pre { background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; }
            .endpoint { background-color: #e8f0fe; padding: 10px; margin-bottom: 15px; border-radius: 4px; }
            .endpoint h3 { margin-top: 0; color: #1a73e8; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .code { font-family: monospace; background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>XGBoost 모델 배포 서버</h1>
        <p>이 서버는 MLflow로 학습된 XGBoost 모델을 API로 제공합니다.</p>
        
        <div class="endpoint">
            <h2>API 엔드포인트</h2>
            
            <h3>GET /health</h3>
            <p>서버와 모델 상태를 확인합니다.</p>
            <pre>curl http://localhost:5001/health</pre>
            
            <h3>GET /metadata</h3>
            <p>모델 메타데이터를 조회합니다.</p>
            <pre>curl http://localhost:5001/metadata</pre>
            
            <h3>GET /statistics</h3>
            <p>API 사용 통계를 조회합니다.</p>
            <pre>curl http://localhost:5001/statistics</pre>
            
            <h3>POST /predict</h3>
            <p>단일 또는 배치 예측을 수행합니다.</p>
            <pre>curl -X POST http://localhost:5001/predict \
    -H "Content-Type: application/json" \
    -d '{"data": [[12.82, 3.37, 2.3, 19.5, 88.0, 1.48, 0.66, 0.4, 0.97, 10.26, 0.72, 1.75, 685.0]]}'</pre>
            
            <h3>POST /batch_predict</h3>
            <p>대량의 데이터에 대한 배치 예측을 수행합니다.</p>
            <pre>curl -X POST http://localhost:5001/batch_predict \
    -H "Content-Type: application/json" \
    -d '{"data": [[...features...], [...features...], ...], "batch_size": 100}'</pre>
            
            <h3>POST /explain</h3>
            <p>예측에 대한 설명을 제공합니다.</p>
            <pre>curl -X POST http://localhost:5001/explain \
    -H "Content-Type: application/json" \
    -d '{"data": [[12.82, 3.37, 2.3, 19.5, 88.0, 1.48, 0.66, 0.4, 0.97, 10.26, 0.72, 1.75, 685.0]]}'</pre>
        </div>
        
        <h2>현재 모델 정보</h2>
        {% if model %}
        <table>
            <tr>
                <th>모델 이름</th>
                <td>{{ model_name }}</td>
            </tr>
            <tr>
                <th>모델 버전</th>
                <td>{{ model_version }}</td>
            </tr>
            <tr>
                <th>모델 스테이지</th>
                <td>{{ model_stage }}</td>
            </tr>
            <tr>
                <th>특성 수</th>
                <td>{{ feature_names|length }}</td>
            </tr>
            <tr>
                <th>클래스 수</th>
                <td>{{ class_names|length }}</td>
            </tr>
        </table>
        {% else %}
        <p>모델이 아직 로드되지 않았습니다.</p>
        {% endif %}
    </body>
    </html>
    '''
    return render_template_string(html, 
                                 model=model, 
                                 model_name=model_name, 
                                 model_version=model_version,
                                 model_stage=model_stage,
                                 feature_names=feature_names,
                                 class_names=class_names)

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
    
    uptime = time.time() - start_time
    
    return jsonify({
        "status": "ok",
        "message": "API가 정상 작동 중입니다.",
        "model_name": model_name,
        "model_version": model_version,
        "model_stage": model_stage,
        "uptime_seconds": uptime,
        "prediction_count": prediction_count
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
    
    # 모델 메타데이터 구성
    metadata = {
        "model_info": model_info,
        "classes": class_names,
        "features": feature_names,
        "signature": {
            "has_signature": hasattr(model.metadata, 'signature'),
            "input_schema": str(model.metadata.signature.inputs) if hasattr(model.metadata, 'signature') else None,
            "output_schema": str(model.metadata.signature.outputs) if hasattr(model.metadata, 'signature') else None
        }
    }
    
    return jsonify(metadata)

@app.route('/statistics', methods=['GET'])
def statistics():
    """
    API 사용 통계를 제공하는 엔드포인트
    """
    uptime = time.time() - start_time
    
    # 성능 지표 계산
    if prediction_times:
        avg_prediction_time = sum(prediction_times) / len(prediction_times)
        max_prediction_time = max(prediction_times)
        min_prediction_time = min(prediction_times)
        p95_prediction_time = sorted(prediction_times)[int(len(prediction_times) * 0.95)]
    else:
        avg_prediction_time = 0
        max_prediction_time = 0
        min_prediction_time = 0
        p95_prediction_time = 0
    
    stats = {
        "uptime_seconds": uptime,
        "total_predictions": prediction_count,
        "predictions_per_second": prediction_count / uptime if uptime > 0 else 0,
        "performance_metrics": {
            "avg_prediction_time_ms": avg_prediction_time * 1000,
            "max_prediction_time_ms": max_prediction_time * 1000,
            "min_prediction_time_ms": min_prediction_time * 1000,
            "p95_prediction_time_ms": p95_prediction_time * 1000
        },
        "recent_predictions": prediction_log[-10:]  # 최근 10개 예측만 표시
    }
    
    return jsonify(stats)

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
        # 예측 시작 시간
        start = time.time()
        
        # 요청 데이터 가져오기
        data = request.json['data']
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "예측할 데이터가 없습니다."
            }), 400
        
        # 데이터 전처리
        input_data = preprocess_input(data, feature_names)
        
        # 예측 수행
        predictions = model.predict(input_data)
        
        # 예측 결과 형식화
        formatted_predictions = format_prediction(predictions, class_names)
        
        # 예측 소요 시간
        prediction_time = time.time() - start
        
        # 예측 로깅
        log_prediction(input_data, formatted_predictions, prediction_time)
        
        response = {
            "status": "success",
            "predictions": formatted_predictions,
            "prediction_time_ms": prediction_time * 1000
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {e}")
        return jsonify({
            "status": "error",
            "message": f"예측 중 오류 발생: {str(e)}"
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    배치 예측 수행 엔드포인트
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
        # 예측 시작 시간
        start = time.time()
        
        # 요청 데이터 가져오기
        data = request.json['data']
        batch_size = request.json.get('batch_size', 100)  # 기본 배치 크기는 100
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "예측할 데이터가 없습니다."
            }), 400
        
        # 데이터 전처리
        input_data = preprocess_input(data, feature_names)
        
        # 결과를 저장할 리스트
        all_predictions = []
        
        # 배치 단위로 예측 수행
        for i in range(0, len(input_data), batch_size):
            batch = input_data.iloc[i:i + batch_size]
            batch_predictions = model.predict(batch)
            formatted_batch = format_prediction(batch_predictions, class_names)
            all_predictions.extend(formatted_batch)
        
        # 예측 소요 시간
        prediction_time = time.time() - start
        
        # 예측 로깅 (첫 번째 배치만)
        if all_predictions:
            log_prediction(input_data.iloc[:1], [all_predictions[0]], prediction_time / len(input_data))
        
        response = {
            "status": "success",
            "prediction_count": len(all_predictions),
            "predictions": all_predictions,
            "batch_prediction_time_ms": prediction_time * 1000
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"배치 예측 중 오류 발생: {e}")
        return jsonify({
            "status": "error",
            "message": f"배치 예측 중 오류 발생: {str(e)}"
        }), 500

@app.route('/explain', methods=['POST'])
def explain():
    """
    예측에 대한 설명을 제공하는 엔드포인트
    """
    if model is None:
        return jsonify({
            "status": "error",
            "message": "모델이 로드되지 않았습니다."
        }), 503
    
    if shap_explainer is None:
        return jsonify({
            "status": "error",
            "message": "SHAP 설명기가 초기화되지 않았습니다."
        }), 501
    
    # 요청 데이터 검증
    if not request.json or 'data' not in request.json:
        return jsonify({
            "status": "error",
            "message": "유효하지 않은 요청 형식입니다. 'data' 필드가 필요합니다."
        }), 400
    
    try:
        # 요청 데이터 가져오기
        data = request.json['data']
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "설명할 데이터가 없습니다."
            }), 400
        
        # 데이터 전처리 (단일 샘플만 허용)
        if isinstance(data[0], list):
            if len(data) > 1:
                return jsonify({
                    "status": "error",
                    "message": "설명은 한 번에 하나의 샘플에 대해서만 가능합니다."
                }), 400
            input_data = preprocess_input(data[0], feature_names)
        else:
            input_data = preprocess_input(data, feature_names)
        
        # 예측 수행
        prediction = model.predict(input_data)
        formatted_prediction = format_prediction(prediction, class_names)
        
        # SHAP 설명 생성
        explanation = generate_shap_explanation(input_data)
        
        response = {
            "status": "success",
            "prediction": formatted_prediction[0],
            "explanation": explanation
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"설명 생성 중 오류 발생: {e}")
        return jsonify({
            "status": "error",
            "message": f"설명 생성 중 오류 발생: {str(e)}"
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """
    사용 가능한 모델 목록을 제공하는 엔드포인트
    """
    try:
        client = MlflowClient()
        
        # 등록된 모든 모델 가져오기
        registered_models = client.search_registered_models()
        
        models_list = []
        for rm in registered_models:
            model_info = {
                "name": rm.name,
                "latest_versions": []
            }
            
            # 모델의 최신 버전 정보
            for mv in client.get_latest_versions(rm.name):
                version_info = {
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "creation_timestamp": datetime.fromtimestamp(mv.creation_timestamp/1000).strftime("%Y-%m-%d %H:%M:%S"),
                    "last_updated_timestamp": datetime.fromtimestamp(mv.last_updated_timestamp/1000).strftime("%Y-%m-%d %H:%M:%S"),
                    "run_id": mv.run_id
                }
                model_info["latest_versions"].append(version_info)
            
            models_list.append(model_info)
        
        return jsonify({
            "status": "success",
            "models": models_list
        })
    
    except Exception as e:
        logger.error(f"모델 목록 조회 중 오류 발생: {e}")
        return jsonify({
            "status": "error",
            "message": f"모델 목록 조회 중 오류 발생: {str(e)}"
        }), 500

@app.route('/load_model', methods=['POST'])
def api_load_model():
    """
    특정 모델을 로드하는 엔드포인트
    """
    try:
        # 요청 데이터 검증
        if not request.json:
            return jsonify({
                "status": "error",
                "message": "유효하지 않은 요청 형식입니다. JSON 데이터가 필요합니다."
            }), 400
        
        # 요청 데이터 가져오기
        model_name_req = request.json.get('model_name', 'wine_xgboost_classifier')
        version = request.json.get('version', 'latest')
        stage = request.json.get('stage')
        
        # 모델 로드
        success = load_model(model_name_req, version, stage)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"모델 '{model_name_req}'을(를) 성공적으로 로드했습니다.",
                "model_info": model_info
            })
        else:
            return jsonify({
                "status": "error",
                "message": "모델 로드에 실패했습니다."
            }), 500
    
    except Exception as e:
        logger.error(f"API를 통한 모델 로드 중 오류 발생: {e}")
        return jsonify({
            "status": "error",
            "message": f"모델 로드 중 오류 발생: {str(e)}"
        }), 500

def main():
    """
    애플리케이션 메인 함수
    """
    global model_name, start_time
    
    print("===== XGBoost 모델 배포 서버 시작 =====")
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 명령줄 인수 처리 (모델 이름, 스테이지 등)
    import argparse
    parser = argparse.ArgumentParser(description='XGBoost 모델 배포 서버')
    parser.add_argument('--model', type=str, default='wine_xgboost_classifier', help='배포할 모델 이름')
    parser.add_argument('--version', type=str, default='latest', help='모델 버전 (기본값: latest)')
    parser.add_argument('--stage', type=str, help='모델 스테이지 (기본값: None, "Production", "Staging" 등)')
    parser.add_argument('--port', type=int, default=5001, help='서버 포트 (기본값: 5001)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='서버 호스트 (기본값: 0.0.0.0)')
    
    args = parser.parse_args()
    model_name = args.model
    
    # 모델 로드
    if not load_model(model_name, args.version, args.stage):
        print("모델 로드 실패로 서버를 시작할 수 없습니다.")
        return
    
    # 서버 호스트 및 포트 설정
    host = args.host
    port = args.port
    
    print(f"\n서버가 http://{host}:{port} 에서 실행 중입니다...")
    print("다음 URL로 API에 접근할 수 있습니다:")
    print(f"- API 홈: http://localhost:{port}/")
    print(f"- 상태 확인: http://localhost:{port}/health")
    print(f"- 모델 메타데이터: http://localhost:{port}/metadata")
    print(f"- API 통계: http://localhost:{port}/statistics")
    
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