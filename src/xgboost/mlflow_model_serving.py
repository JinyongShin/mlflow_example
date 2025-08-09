"""
MLflow로 저장된 XGBoost 모델을 로드하고 추론하는 고급 예제

이 스크립트는 MLflow 공식 가이드(https://mlflow.org/docs/latest/ml/traditional-ml/xgboost/guide/)를 기반으로
XGBoost 모델을 다양한 방식으로 로드하고 추론하는 방법과 벤치마크를 제공합니다.

주요 기능:
- 다양한 모델 로드 방식
- 추론 성능 벤치마킹
- CPU/GPU 벤치마크 비교
- 다양한 추론 최적화 기법
"""

import os
import time
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import xgboost as xgb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
from mlflow.tracking import MlflowClient
from datetime import datetime
import logging
from tabulate import tabulate

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 결과 테이블 출력 형식 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def benchmark_prediction(model, X, iterations=10, batch_size=None, warm_up=True):
    """
    모델 추론 성능을 벤치마크합니다.
    
    Args:
        model: 추론에 사용할 모델
        X: 입력 데이터 (pandas DataFrame 또는 numpy array)
        iterations: 반복 횟수
        batch_size: 배치 크기 (None이면 전체 데이터셋)
        warm_up: 워밍업 수행 여부
    
    Returns:
        dict: 벤치마크 결과
    """
    # 워밍업 (첫 번째 실행은 초기화 시간 때문에 느릴 수 있음)
    if warm_up:
        if batch_size:
            batch = X.iloc[:min(batch_size, len(X))] if hasattr(X, 'iloc') else X[:min(batch_size, len(X))]
            model.predict(batch)
        else:
            model.predict(X)
    
    all_times = []
    all_throughputs = []
    
    # 벤치마크 수행
    for i in range(iterations):
        if batch_size:
            # 배치 추론
            times_per_batch = []
            num_samples_processed = 0
            
            for j in range(0, len(X), batch_size):
                batch_end = min(j + batch_size, len(X))
                batch = X.iloc[j:batch_end] if hasattr(X, 'iloc') else X[j:batch_end]
                
                start_time = time.time()
                model.predict(batch)
                end_time = time.time()
                
                batch_time = end_time - start_time
                times_per_batch.append(batch_time)
                num_samples_processed += len(batch)
            
            total_time = sum(times_per_batch)
        else:
            # 전체 데이터셋 추론
            start_time = time.time()
            model.predict(X)
            end_time = time.time()
            
            total_time = end_time - start_time
            num_samples_processed = len(X)
        
        # 처리량 계산 (samples/second)
        throughput = num_samples_processed / total_time if total_time > 0 else 0
        
        all_times.append(total_time)
        all_throughputs.append(throughput)
    
    # 결과 집계
    avg_time = sum(all_times) / len(all_times)
    avg_throughput = sum(all_throughputs) / len(all_throughputs)
    
    return {
        "avg_prediction_time_seconds": avg_time,
        "avg_samples_per_second": avg_throughput,
        "total_samples": len(X),
        "iterations": iterations,
        "batch_size": batch_size if batch_size else len(X)
    }

def load_model_many_ways():
    """
    다양한 방식으로 MLflow 모델을 로드하는 방법을 보여줍니다.
    """
    print("\n===== 다양한 방식으로 모델 로드하기 =====")
    
    # Wine 데이터셋 로드 (테스트 데이터로 사용)
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names  # feature_names 명시적으로 미리 저장
    
    # 훈련 및 테스트 세트로 분할 (테스트 세트만 사용할 예정)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 테스트 데이터가 정확한 feature_names 순서와 이름을 가지도록 함
    X_test = X_test[feature_names]
    
    # 테스트 샘플 선택
    test_samples = X_test.iloc[:5]
    
    # MLflow 클라이언트 초기화
    client = MlflowClient()
    
    print("MLflow에서 모델 로드 및 추론 방식 비교\n")
    
    try:
        # 1. 방법 1: 모델 레지스트리에서 최신 버전의 모델 가져오기 (Python 함수로)
        model_name = "wine_xgboost_classifier"
        
        try:
            # 모델 존재 확인
            latest_models = client.search_model_versions(f"name='{model_name}'")
            if not latest_models:
                raise Exception(f"모델 '{model_name}'을 찾을 수 없습니다. 먼저 'mlflow_example.py' 또는 'mlflow_hyperparameter_tuning.py'를 실행하여 모델을 생성하세요.")
            
            print(f"\n1. 모델 레지스트리에서 최신 모델 로드: models:/{model_name}/latest")
            latest_model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
            
            # 모델 타입 출력
            print(f"  모델 타입: {type(latest_model).__name__}")
            print(f"  지원되는 플레이버: {list(latest_model.metadata.flavors.keys())}")
            
            # 모델 시그니처 확인
            if hasattr(latest_model.metadata, 'signature'):
                print(f"  모델 시그니처 존재: 예")
                if hasattr(latest_model.metadata.signature, 'inputs'):
                    print(f"  입력 스키마: {latest_model.metadata.signature.inputs.input_names()}")
            else:
                print("  모델 시그니처 존재: 아니오")
            
            # 예측 수행
            predictions = latest_model.predict(test_samples)
            
            # 예측 결과 출력
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                predicted_classes = np.argmax(predictions, axis=1)
                print(f"  예측 출력 형태: {predictions.shape} (확률값)")
            else:
                predicted_classes = predictions.astype(int)
                print(f"  예측 출력 형태: {predictions.shape} (클래스 인덱스)")
            
            # 결과 테이블 생성
            results_df = pd.DataFrame({
                "예측 클래스": predicted_classes,
                "실제 클래스": y_test.iloc[:5].values
            })
            print("\n  예측 결과 (첫 5개 샘플):")
            print(tabulate(results_df, headers='keys', tablefmt='psql', showindex=False))
            
        except Exception as e:
            print(f"  최신 모델 로드 실패: {e}")
        
        # 2. 방법 2: 특정 모델 버전 로드하기
        try:
            # 특정 버전 가져오기
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                specific_version = versions[0].version  # 첫 번째 버전 사용
                print(f"\n2. 특정 모델 버전 로드: models:/{model_name}/{specific_version}")
                versioned_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{specific_version}")
                
                # 모델 타입 출력
                print(f"  모델 버전: {specific_version}")
                
                # 예측 수행
                versioned_pred = versioned_model.predict(test_samples)
                if versioned_pred.ndim > 1 and versioned_pred.shape[1] > 1:
                    versioned_classes = np.argmax(versioned_pred, axis=1)
                else:
                    versioned_classes = versioned_pred.astype(int)
                
                print(f"  예측 결과 정확도: {np.mean(versioned_classes == y_test.iloc[:5].values):.4f}")
        except Exception as e:
            print(f"  특정 버전 모델 로드 실패: {e}")
        
        # 3. 방법 3: 특정 모델 스테이지 로드하기
        try:
            # Production 스테이지의 모델 가져오기 시도
            stage = "Production"
            print(f"\n3. 특정 스테이지 모델 로드: models:/{model_name}/{stage}")
            
            # Production 단계의 모델 존재 여부 확인
            stage_versions = client.get_latest_versions(model_name, stages=[stage])
            
            if stage_versions:
                stage_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
                print(f"  '{stage}' 스테이지 모델 로드 성공")
                
                # 예측 수행
                stage_pred = stage_model.predict(test_samples)
                if stage_pred.ndim > 1 and stage_pred.shape[1] > 1:
                    stage_classes = np.argmax(stage_pred, axis=1)
                else:
                    stage_classes = stage_pred.astype(int)
                
                print(f"  예측 결과 정확도: {np.mean(stage_classes == y_test.iloc[:5].values):.4f}")
            else:
                print(f"  '{stage}' 스테이지에 모델이 없습니다. 모델을 Production 스테이지로 전환하려면:")
                print("  client.transition_model_version_stage(model_name, version, 'Production')")
        except Exception as e:
            print(f"  스테이지 모델 로드 실패: {e}")
        
        # 4. 방법 4: 네이티브 XGBoost 모델 로드
        try:
            print("\n4. 네이티브 XGBoost 모델 로드")
            native_model = mlflow.xgboost.load_model(f"models:/{model_name}/latest")
            print("  네이티브 XGBoost 모델 로드 성공")
            print(f"  모델 타입: {type(native_model).__name__}")
            
            # XGBoost DMatrix 변환
            dtest = xgb.DMatrix(test_samples.values, feature_names=feature_names)
            
            # 네이티브 XGBoost 추론
            native_pred = native_model.predict(dtest)
            native_classes = np.argmax(native_pred, axis=1)
            
            print(f"  예측 결과 정확도: {np.mean(native_classes == y_test.iloc[:5].values):.4f}")
            
            # 특성 중요도 출력
            if hasattr(native_model, 'get_score'):
                importance = native_model.get_score(importance_type='gain')
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                print("\n  특성 중요도 (상위 5개):")
                for feature, score in sorted_importance[:5]:
                    print(f"    {feature}: {score:.4f}")
        except Exception as e:
            print(f"  네이티브 모델 로드 실패: {e}")
        
        # 5. 방법 5: MLflow 실행 ID로 모델 로드
        try:
            print("\n5. MLflow 실행 ID로 모델 로드")
            
            # 모델의 마지막 실행 ID 가져오기
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                run_id = versions[0].run_id
                if run_id:
                    print(f"  실행 ID: {run_id}")
                    run_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/xgboost_model")
                    print("  실행 ID로 모델 로드 성공")
                    
                    # 예측 수행
                    run_pred = run_model.predict(test_samples)
                    if run_pred.ndim > 1 and run_pred.shape[1] > 1:
                        run_classes = np.argmax(run_pred, axis=1)
                    else:
                        run_classes = run_pred.astype(int)
                    
                    print(f"  예측 결과 정확도: {np.mean(run_classes == y_test.iloc[:5].values):.4f}")
                else:
                    print("  모델에 연결된 실행 ID가 없습니다.")
        except Exception as e:
            print(f"  실행 ID 모델 로드 실패: {e}")
        
        # 6. 방법 6: SHAP 설명 기능
        try:
            print("\n6. SHAP를 사용한 모델 설명")
            
            # 네이티브 모델을 사용하여 SHAP 설명 생성
            if 'native_model' in locals():
                explainer = shap.TreeExplainer(native_model)
                print("  SHAP 설명기 초기화 성공")
                
                # SHAP 값 계산
                shap_values = explainer.shap_values(dtest)
                if isinstance(shap_values, list):
                    # 다중 클래스 문제인 경우
                    print(f"  SHAP 값 형태: {len(shap_values)} 클래스, 각 {shap_values[0].shape}")
                    
                    # 첫 번째 샘플의 첫 번째 클래스에 대한 상위 특성
                    class_idx = 0
                    print(f"  클래스 {class_idx}에 대한 상위 특성 기여도 (첫 번째 샘플):")
                    feature_contributions = [(feature_names[i], shap_values[class_idx][0, i]) 
                                            for i in range(len(feature_names))]
                    sorted_contribs = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)
                    
                    for feature, value in sorted_contribs[:5]:
                        print(f"    {feature}: {value:.4f}")
                else:
                    # 이진 분류 또는 회귀 문제인 경우
                    print(f"  SHAP 값 형태: {shap_values.shape}")
                    
                    # 첫 번째 샘플에 대한 상위 특성
                    feature_contributions = [(feature_names[i], shap_values[0, i]) 
                                           for i in range(len(feature_names))]
                    sorted_contribs = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)
                    
                    print("  상위 특성 기여도 (첫 번째 샘플):")
                    for feature, value in sorted_contribs[:5]:
                        print(f"    {feature}: {value:.4f}")
            else:
                print("  네이티브 모델을 먼저 로드해야 SHAP 설명을 생성할 수 있습니다.")
        except Exception as e:
            print(f"  SHAP 설명 생성 실패: {e}")
            print("  SHAP 패키지를 설치하려면: pip install shap")
        
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        print("먼저 'mlflow_example.py' 또는 'mlflow_hyperparameter_tuning.py'를 실행하여 모델을 학습하세요.")

def inference_optimization():
    """
    다양한 추론 최적화 기법을 보여줍니다.
    """
    print("\n===== 추론 최적화 기법 =====")
    
    # Wine 데이터셋 로드
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    
    # 훈련 및 테스트 세트로 분할
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 테스트 데이터가 정확한 feature_names 순서를 가지도록 함
    X_test = X_test[feature_names]
    
    # 더 큰 테스트 세트 생성 (벤치마크용)
    X_large = pd.concat([X_test] * 100, ignore_index=True)
    y_large = pd.concat([y_test] * 100, ignore_index=True)
    
    print(f"테스트 데이터셋 크기: {len(X_test)} 샘플")
    print(f"확장 데이터셋 크기: {len(X_large)} 샘플")
    
    try:
        # 모델 로드
        model_name = "wine_xgboost_classifier"
        
        # 1. MLflow Python 함수 모델 로드
        print("\n1. MLflow Python 함수 모델 (기본)")
        model_mlflow = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        
        # 기본 벤치마크 (일괄 처리)
        print("  일괄 처리 벤치마크:")
        result_batch = benchmark_prediction(model_mlflow, X_large)
        print(f"  예측 시간: {result_batch['avg_prediction_time_seconds']:.4f}초")
        print(f"  처리량: {result_batch['avg_samples_per_second']:.2f} 샘플/초")
        
        # 배치 크기별 벤치마크
        batch_sizes = [1, 10, 50, 100, 500]
        batch_results = []
        
        print("\n  배치 크기별 벤치마크:")
        for batch_size in batch_sizes:
            result = benchmark_prediction(model_mlflow, X_large, batch_size=batch_size)
            batch_results.append({
                "batch_size": batch_size,
                "time": result["avg_prediction_time_seconds"],
                "throughput": result["avg_samples_per_second"]
            })
            print(f"  배치 크기 {batch_size}: {result['avg_prediction_time_seconds']:.4f}초, "
                 f"{result['avg_samples_per_second']:.2f} 샘플/초")
        
        # 2. 네이티브 XGBoost 모델 로드
        print("\n2. 네이티브 XGBoost 모델")
        try:
            model_native = mlflow.xgboost.load_model(f"models:/{model_name}/latest")
            
            # DMatrix 변환
            dtest_large = xgb.DMatrix(X_large.values, feature_names=feature_names)
            
            # 벤치마크 (일괄 처리)
            start_time = time.time()
            for _ in range(10):  # 10번 반복
                model_native.predict(dtest_large)
            avg_time = (time.time() - start_time) / 10
            throughput = len(X_large) / avg_time
            
            print(f"  예측 시간: {avg_time:.4f}초")
            print(f"  처리량: {throughput:.2f} 샘플/초")
            
            # 네이티브 vs Python 함수 모델 성능 비교
            speedup = result_batch['avg_prediction_time_seconds'] / avg_time
            print(f"\n  성능 비교: 네이티브 모델이 Python 함수 모델보다 {speedup:.2f}배 빠름")
            
        except Exception as e:
            print(f"  네이티브 모델 벤치마크 실패: {e}")
        
        # 3. GPU 가속 (가능한 경우)
        print("\n3. GPU 가속 (XGBoost 모델)")
        try:
            # GPU 사용 가능 여부 확인
            gpu_available = False
            try:
                import cupy  # GPU 지원 확인을 위한 임시 가져오기
                gpu_available = True
            except ImportError:
                pass
            
            if gpu_available:
                print("  GPU 가용성: 있음")
                
                # GPU 파라미터로 모델 설정
                params = {
                    "tree_method": "gpu_hist",
                    "gpu_id": 0
                }
                
                # 모델에 GPU 파라미터 설정 시도
                if hasattr(model_native, 'set_param'):
                    for param, value in params.items():
                        model_native.set_param(param, value)
                    
                    # GPU 벤치마크
                    start_time = time.time()
                    for _ in range(10):  # 10번 반복
                        model_native.predict(dtest_large)
                    avg_time = (time.time() - start_time) / 10
                    throughput = len(X_large) / avg_time
                    
                    print(f"  GPU 예측 시간: {avg_time:.4f}초")
                    print(f"  GPU 처리량: {throughput:.2f} 샘플/초")
                    
                    # CPU vs GPU 성능 비교
                    speedup = result_batch['avg_prediction_time_seconds'] / avg_time
                    print(f"\n  성능 비교: GPU가 CPU보다 {speedup:.2f}배 빠름")
                else:
                    print("  이 모델은 GPU 파라미터를 설정할 수 없습니다.")
            else:
                print("  GPU 가용성: 없음")
                print("  GPU 가속을 사용하려면 CUDA와 관련 패키지를 설치하세요:")
                print("  pip install cupy xgboost-gpu")
        except Exception as e:
            print(f"  GPU 벤치마크 실패: {e}")
        
        # 4. 추론 최적화 팁 출력
        print("\n4. 추론 최적화 팁")
        print("  1. 가능하면 네이티브 XGBoost 모델 사용")
        print("  2. 큰 배치 크기로 처리 (100-500 샘플)")
        print("  3. GPU 가속 활용 (가능한 경우)")
        print("  4. 모델 양자화 또는 경량화 고려")
        print("  5. tree_method='hist' 또는 tree_method='gpu_hist' 사용")
        print("  6. 추론 전용 라이브러리 (Treelite) 고려")
        
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("먼저 'mlflow_example.py' 또는 'mlflow_hyperparameter_tuning.py'를 실행하여 모델을 학습하세요.")

def model_versioning_and_promotion():
    """
    모델 버전 관리 및 승격에 대한 정보를 제공합니다.
    """
    print("\n===== 모델 버전 관리 및 승격 =====")
    
    try:
        # MLflow 클라이언트 초기화
        client = MlflowClient()
        
        # 1. 모델 버전 나열
        model_name = "wine_xgboost_classifier"
        print(f"\n1. 모델 '{model_name}'의 버전 목록:")
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"  모델 '{model_name}'에 대한 버전이 없습니다.")
            print("  먼저 'mlflow_example.py' 또는 'mlflow_hyperparameter_tuning.py'를 실행하여 모델을 생성하세요.")
            return
        
        # 버전 정보 테이블 생성
        versions_data = []
        for v in versions:
            versions_data.append({
                "버전": v.version,
                "스테이지": v.current_stage,
                "생성 시간": datetime.fromtimestamp(v.creation_timestamp/1000).strftime("%Y-%m-%d %H:%M"),
                "실행 ID": v.run_id,
                "소스": v.source
            })
        
        versions_df = pd.DataFrame(versions_data)
        print(tabulate(versions_df, headers='keys', tablefmt='psql', showindex=False))
        
        # 2. 스테이지 전환 (시뮬레이션)
        print("\n2. 모델 스테이지 전환 예시:")
        latest_version = versions[0].version
        
        print(f"  명령어: client.transition_model_version_stage('{model_name}', '{latest_version}', 'Staging')")
        print("  결과: 모델이 'Staging' 스테이지로 전환됩니다.")
        
        print(f"  명령어: client.transition_model_version_stage('{model_name}', '{latest_version}', 'Production')")
        print("  결과: 모델이 'Production' 스테이지로 전환됩니다.")
        
        # 3. 모델 등록 해제 (시뮬레이션)
        print("\n3. 모델 등록 해제 예시:")
        print(f"  명령어: client.delete_registered_model('{model_name}')")
        print("  결과: 모델과 모든 버전이 레지스트리에서 제거됩니다.")
        
        # 4. 모델 태그 추가 (시뮬레이션)
        print("\n4. 모델 태그 추가 예시:")
        print(f"  명령어: client.set_registered_model_tag('{model_name}', 'owner', 'data_science_team')")
        print("  결과: 모델에 'owner' 태그가 추가됩니다.")
        
        print(f"  명령어: client.set_model_version_tag('{model_name}', '{latest_version}', 'accuracy', '0.92')")
        print("  결과: 모델 버전에 'accuracy' 태그가 추가됩니다.")
        
        # 5. 모델 아티팩트 다운로드 (시뮬레이션)
        print("\n5. 모델 아티팩트 다운로드 예시:")
        print(f"  명령어: mlflow.xgboost.load_model(f'models:/{model_name}/latest').save_model('downloaded_model.json')")
        print("  결과: 최신 모델이 로컬 파일 'downloaded_model.json'으로 저장됩니다.")
    
    except Exception as e:
        print(f"모델 버전 관리 정보 검색 실패: {e}")

def main():
    """
    MLflow로 저장된 XGBoost 모델 서빙 예제의 메인 함수
    """
    print("===== MLflow와 XGBoost 모델 서빙 예제 =====")
    
    # 예제 실행
    load_model_many_ways()
    inference_optimization()
    model_versioning_and_promotion()
    
    print("\n===== 모든 예제가 성공적으로 실행되었습니다 =====")
    print("\nMLflow UI를 확인하려면 터미널에서 다음 명령어를 실행하세요:")
    print("mlflow ui")
    print("그리고 브라우저에서 http://localhost:5000 을 방문하세요.\n")
    print("이 예제들은 MLflow 공식 가이드를 기반으로 작성되었습니다:")
    print("https://mlflow.org/docs/latest/ml/traditional-ml/xgboost/guide/")

if __name__ == "__main__":
    main()