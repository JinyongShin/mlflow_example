"""
MLflow와 XGBoost를 활용한 고급 하이퍼파라미터 튜닝

이 스크립트는 MLflow 공식 가이드(https://mlflow.org/docs/latest/ml/traditional-ml/xgboost/guide/)를 기반으로
XGBoost 모델의 하이퍼파라미터 튜닝, 모델 비교 및 선택, 품질 검증을 수행하는 방법을 보여줍니다.

주요 기능:
- 베이지안 최적화를 통한 하이퍼파라미터 튜닝
- K-Fold 교차 검증
- 모델 품질 게이트를 통한 자동 검증
- 모델 비교 및 분석
"""

import os
import sys
import platform
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.xgboost
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)
from mlflow.models import infer_signature
import time
import optuna
from optuna.integration.mlflow import MLflowCallback
import joblib

# 재현성을 위한 랜덤 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# 환경 정보
PYTHON_VERSION = platform.python_version()
XGBOOST_VERSION = xgb.__version__
PLATFORM = platform.platform()

# MLflow 모델 평가를 위한 임계값 정의
class MetricThreshold:
    def __init__(self, threshold, greater_is_better=True):
        self.threshold = threshold
        self.greater_is_better = greater_is_better

def set_experiment_tags(task_type, tuning_method):
    """실험에 태그를 설정하여 추적을 용이하게 합니다."""
    mlflow.set_tags({
        "python_version": PYTHON_VERSION,
        "xgboost_version": XGBOOST_VERSION,
        "platform": PLATFORM,
        "random_state": RANDOM_STATE,
        "model_type": "gradient_boosting",
        "algorithm": "xgboost",
        "task": task_type,
        "tuning_method": tuning_method
    })

def plot_feature_importance(model, feature_names=None, top_n=15, importance_type="gain"):
    """
    XGBoost 모델의 특성 중요도를 시각화합니다.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(
        model,
        importance_type=importance_type,
        ax=ax,
        title=f'특성 중요도 ({importance_type} 기준)',
        xlabel='중요도 점수',
        ylabel='특성',
        max_num_features=top_n
    )
    plt.tight_layout()
    return fig

def grid_search_tuning():
    """
    그리드 서치를 사용한 XGBoost 하이퍼파라미터 튜닝 예제
    """
    print("\n=== 그리드 서치 하이퍼파라미터 튜닝 ===")
    
    # MLflow 실험 설정
    experiment_name = "xgboost-grid-search-tuning"
    mlflow.set_experiment(experiment_name)
    
    # Wine 데이터셋 로드 (분류 문제)
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    
    # 훈련 및 테스트 세트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # 하이퍼파라미터 그리드 정의
    learning_rate_list = [0.01, 0.1, 0.3]
    max_depth_list = [3, 6, 9]
    subsample_list = [0.8, 1.0]
    colsample_bytree_list = [0.8, 1.0]
    
    best_accuracy = 0.0
    best_params = {}
    best_model = None
    
    # 하이퍼파라미터 조합에 대한 그리드 서치 수행
    total_combinations = len(learning_rate_list) * len(max_depth_list) * len(subsample_list) * len(colsample_bytree_list)
    print(f"총 {total_combinations}개의 하이퍼파라미터 조합을 평가합니다...")
    
    # 진행 상황 추적
    combination_count = 0
    start_time = time.time()
    
    for learning_rate in learning_rate_list:
        for max_depth in max_depth_list:
            for subsample in subsample_list:
                for colsample_bytree in colsample_bytree_list:
                    combination_count += 1
                    print(f"조합 {combination_count}/{total_combinations} 평가 중... ", end="", flush=True)
                    
                    # MLflow 실행 시작
                    with mlflow.start_run(run_name=f"grid_search_run_{combination_count}"):
                        # 태그 설정
                        set_experiment_tags("classification", "grid_search")
                        
                        # 현재 하이퍼파라미터 조합
                        params = {
                            "objective": "multi:softprob",
                            "num_class": 3,  # Wine 데이터셋의 클래스 수
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            "subsample": subsample,
                            "colsample_bytree": colsample_bytree,
                            "tree_method": "hist",
                            "random_state": RANDOM_STATE
                        }
                        
                        # 하이퍼파라미터 로깅
                        mlflow.log_params(params)
                        
                        # K-Fold 교차 검증 수행
                        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                        cv_scores = []
                        
                        for train_idx, valid_idx in kf.split(X_train):
                            # 교차 검증용 분할
                            X_cv_train = X_train.iloc[train_idx]
                            y_cv_train = y_train.iloc[train_idx]
                            X_cv_valid = X_train.iloc[valid_idx]
                            y_cv_valid = y_train.iloc[valid_idx]
                            
                            # XGBoost용 DMatrix 변환
                            dtrain = xgb.DMatrix(X_cv_train, label=y_cv_train, feature_names=feature_names)
                            dvalid = xgb.DMatrix(X_cv_valid, label=y_cv_valid, feature_names=feature_names)
                            
                            # 모델 학습
                            model_cv = xgb.train(
                                params,
                                dtrain,
                                num_boost_round=100,
                                early_stopping_rounds=10,
                                evals=[(dvalid, 'validation')],
                                verbose_eval=False
                            )
                            
                            # 검증 세트 예측
                            y_cv_pred = model_cv.predict(dvalid)
                            y_cv_pred_class = np.argmax(y_cv_pred, axis=1)
                            
                            # 정확도 계산
                            accuracy = accuracy_score(y_cv_valid, y_cv_pred_class)
                            cv_scores.append(accuracy)
                        
                        # 교차 검증 결과 계산
                        cv_mean = np.mean(cv_scores)
                        cv_std = np.std(cv_scores)
                        
                        # 교차 검증 결과 로깅
                        mlflow.log_metrics({
                            "cv_mean_accuracy": cv_mean,
                            "cv_std_accuracy": cv_std
                        })
                        
                        # 전체 훈련 세트로 모델 훈련
                        dtrain_full = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
                        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
                        
                        eval_results = {}
                        model = xgb.train(
                            params,
                            dtrain_full,
                            num_boost_round=100,
                            early_stopping_rounds=10,
                            evals=[(dtrain_full, 'train'), (dtest, 'test')],
                            evals_result=eval_results,
                            verbose_eval=False
                        )
                        
                        # 테스트 세트에서 성능 평가
                        y_pred = model.predict(dtest)
                        y_pred_class = np.argmax(y_pred, axis=1)
                        
                        # 성능 지표 계산
                        test_accuracy = accuracy_score(y_test, y_pred_class)
                        test_f1 = f1_score(y_test, y_pred_class, average='weighted')
                        
                        # 테스트 성능 로깅
                        mlflow.log_metrics({
                            "test_accuracy": test_accuracy,
                            "test_f1": test_f1
                        })
                        
                        # 최고 성능 모델 갱신
                        if test_accuracy > best_accuracy:
                            best_accuracy = test_accuracy
                            best_params = params.copy()
                            best_model = model
                        
                        print(f"완료 - 정확도: {test_accuracy:.4f}, CV 평균: {cv_mean:.4f}")
    
    # 총 소요 시간 측정
    total_time = time.time() - start_time
    print(f"\n그리드 서치 완료! 총 소요 시간: {total_time:.2f}초")
    
    # 최적 모델 정보 출력
    print("\n최적 하이퍼파라미터:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    print(f"최고 테스트 정확도: {best_accuracy:.4f}")
    
    # 최적 모델을 별도 실행으로 로깅 및 등록
    with mlflow.start_run(run_name="grid_search_best_model"):
        # 태그 설정
        set_experiment_tags("classification", "grid_search_best")
        mlflow.set_tag("best_run", True)
        
        # 하이퍼파라미터 로깅
        mlflow.log_params(best_params)
        
        # 성능 지표 로깅
        mlflow.log_metric("test_accuracy", best_accuracy)
        
        # 특성 중요도 시각화 및 로깅
        feature_imp_fig = plot_feature_importance(best_model, feature_names)
        mlflow.log_figure(feature_imp_fig, "feature_importance.png")
        
        # 모델 서명(signature) 생성
        X_sample = X_train.iloc[:5]
        y_pred_sample = best_model.predict(xgb.DMatrix(X_sample.values, feature_names=feature_names))
        signature = infer_signature(X_sample, y_pred_sample)
        
        # 최적 모델 로깅 및 등록
        mlflow.xgboost.log_model(
            best_model, 
            "best_xgboost_model",
            signature=signature,
            input_example=X_train.iloc[:5],
            registered_model_name="wine_xgboost_grid_search_optimized"
        )
        
        print(f"최적 모델이 등록되었습니다: wine_xgboost_grid_search_optimized")
    
    return best_model, best_params

def bayesian_optimization_tuning():
    """
    Optuna를 사용한 베이지안 최적화 하이퍼파라미터 튜닝 예제
    """
    print("\n=== 베이지안 최적화 하이퍼파라미터 튜닝 ===")
    
    # MLflow 실험 설정
    experiment_name = "xgboost-bayesian-optimization"
    mlflow.set_experiment(experiment_name)
    
    # California Housing 데이터셋 로드 (회귀 문제)
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names
    
    # 훈련 및 테스트 세트로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # MLflow 콜백 설정
    mlflc = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="val_rmse", 
        create_experiment=False
    )
    
    @mlflc.track_in_mlflow()
    def objective(trial):
        """
        Optuna 최적화를 위한 목적 함수 정의
        - 목표: 검증 세트에서 RMSE 최소화
        """
        # 하이퍼파라미터 공간 정의
        params = {
            "objective": "reg:squarederror",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "tree_method": "hist",
            "random_state": RANDOM_STATE
        }
        
        # 태그 설정
        set_experiment_tags("regression", "bayesian_optimization")
        
        # K-Fold 교차 검증
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        rmse_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            # 교차 검증용 분할
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # XGBoost 모델 학습
            dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train, feature_names=feature_names)
            dval = xgb.DMatrix(X_fold_val, label=y_fold_val, feature_names=feature_names)
            
            # 조기 종료 설정
            eval_result = {}
            
            # 모델 학습
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                early_stopping_rounds=10,
                evals=[(dval, "val")],
                evals_result=eval_result,
                verbose_eval=False
            )
            
            # 검증 세트에 대한 예측 및 RMSE 계산
            y_pred = model.predict(dval)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            rmse_scores.append(rmse)
        
        # 평균 RMSE 계산
        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        
        # 추가 지표 로깅
        trial.set_user_attr("mean_rmse", mean_rmse)
        trial.set_user_attr("std_rmse", std_rmse)
        
        return mean_rmse
    
    # Optuna 연구 생성 및 최적화 실행
    study = optuna.create_study(direction="minimize", study_name="xgboost_housing_regression")
    study.optimize(objective, n_trials=20, callbacks=[mlflc])
    
    # 최적 하이퍼파라미터 가져오기
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": RANDOM_STATE
    })
    
    print("\n베이지안 최적화 완료!")
    print("\n최적 하이퍼파라미터:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    print(f"최적 검증 RMSE: {study.best_value:.4f}")
    
    # 전체 훈련 데이터로 최종 모델 훈련
    with mlflow.start_run(run_name="bayesian_opt_best_model"):
        # 태그 설정
        set_experiment_tags("regression", "bayesian_optimization_best")
        mlflow.set_tag("best_run", True)
        
        # 하이퍼파라미터 로깅
        mlflow.log_params(best_params)
        
        # 모델 훈련
        dtrain_full = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
        
        eval_results = {}
        final_model = xgb.train(
            best_params,
            dtrain_full,
            num_boost_round=100,
            early_stopping_rounds=10,
            evals=[(dtrain_full, 'train'), (dtest, 'test')],
            evals_result=eval_results,
            verbose_eval=False
        )
        
        # 테스트 성능 평가
        y_pred = final_model.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # 성능 지표 로깅
        mlflow.log_metrics({
            "test_rmse": rmse,
            "test_r2": r2
        })
        
        # 특성 중요도 시각화 및 로깅
        feature_imp_fig = plot_feature_importance(final_model, feature_names)
        mlflow.log_figure(feature_imp_fig, "feature_importance.png")
        
        # 실제 vs 예측값 시각화
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('실제 값')
        ax.set_ylabel('예측 값')
        ax.set_title('실제 vs 예측 주택 가격')
        mlflow.log_figure(fig, "predictions_scatter.png")
        
        # 잔차 플롯
        fig, ax = plt.subplots(figsize=(10, 6))
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('예측 값')
        ax.set_ylabel('잔차')
        ax.set_title('잔차 플롯')
        mlflow.log_figure(fig, "residuals.png")
        
        # 모델 서명(signature) 생성
        X_sample = X_train.iloc[:5]
        y_pred_sample = final_model.predict(xgb.DMatrix(X_sample))
        signature = infer_signature(X_sample, y_pred_sample)
        
        # 최적 모델 로깅 및 등록
        mlflow.xgboost.log_model(
            final_model, 
            "best_xgboost_model",
            signature=signature,
            input_example=X_train.iloc[:5],
            registered_model_name="housing_xgboost_bayesian_opt"
        )
        
        print(f"최적 모델이 등록되었습니다: housing_xgboost_bayesian_opt")
        print(f"테스트 RMSE: {rmse:.4f}")
        print(f"테스트 R²: {r2:.4f}")
    
    # 학습 곡선 저장
    fig, ax = plt.subplots(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study, ax=ax)
    ax.set_title("베이지안 최적화 학습 곡선")
    fig.tight_layout()
    plt.savefig("bayesian_opt_history.png")
    
    # Optuna 연구 저장
    joblib.dump(study, "housing_xgboost_study.pkl")
    
    return final_model, best_params

def model_quality_gates():
    """
    품질 게이트를 통한 모델 검증 예제
    - 모델이 배포되기 전에 특정 성능 기준을 충족하는지 확인
    - 이전 모델과 비교하여 충분한 성능 향상이 있는지 검증
    """
    print("\n=== 모델 품질 게이트 ===")
    
    # MLflow 실험 설정
    experiment_name = "xgboost-model-validation"
    mlflow.set_experiment(experiment_name)
    
    # Wine 데이터셋 로드
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    class_names = wine.target_names
    
    # 훈련, 검증, 테스트 세트 분할
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )
    
    # 베이스라인 모델 (단순 모델)
    with mlflow.start_run(run_name="baseline_model"):
        # 태그 설정
        set_experiment_tags("classification", "baseline")
        
        # 베이스라인 모델 파라미터
        baseline_params = {
            "objective": "multi:softprob",
            "num_class": len(class_names),
            "max_depth": 3,  # 단순한 모델
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": RANDOM_STATE
        }
        
        # 베이스라인 모델 학습
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        
        baseline_model = xgb.train(
            baseline_params,
            dtrain,
            num_boost_round=50,  # 적은 부스팅 라운드
            early_stopping_rounds=10,
            evals=[(dtrain, "train"), (dval, "val")],
            verbose_eval=False
        )
        
        # 검증 세트 예측
        y_pred = baseline_model.predict(dval)
        y_pred_class = np.argmax(y_pred, axis=1)
        
        # 성능 지표 계산
        accuracy = accuracy_score(y_val, y_pred_class)
        f1 = f1_score(y_val, y_pred_class, average='weighted')
        
        # 성능 지표 로깅
        mlflow.log_metrics({
            "val_accuracy": accuracy,
            "val_f1": f1
        })
        
        # 모델 로깅
        mlflow.xgboost.log_model(
            baseline_model, 
            "baseline_model",
            registered_model_name="wine_xgboost_baseline"
        )
        
        baseline_model_uri = f"runs:/{mlflow.active_run().info.run_id}/baseline_model"
        print(f"베이스라인 모델 정확도: {accuracy:.4f}")
        print(f"베이스라인 모델 F1 점수: {f1:.4f}")
    
    # 후보 모델 (더 복잡한 모델)
    with mlflow.start_run(run_name="candidate_model"):
        # 태그 설정
        set_experiment_tags("classification", "candidate")
        
        # 후보 모델 파라미터 (더 최적화됨)
        candidate_params = {
            "objective": "multi:softprob",
            "num_class": len(class_names),
            "max_depth": 6,  # 더 복잡한 모델
            "learning_rate": 0.05,  # 더 작은 학습률
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
            "gamma": 0.1,
            "tree_method": "hist",
            "random_state": RANDOM_STATE
        }
        
        # 후보 모델 학습
        candidate_model = xgb.train(
            candidate_params,
            dtrain,
            num_boost_round=200,  # 더 많은 부스팅 라운드
            early_stopping_rounds=20,
            evals=[(dtrain, "train"), (dval, "val")],
            verbose_eval=False
        )
        
        # 검증 세트 예측
        y_pred = candidate_model.predict(dval)
        y_pred_class = np.argmax(y_pred, axis=1)
        
        # 성능 지표 계산
        accuracy = accuracy_score(y_val, y_pred_class)
        f1 = f1_score(y_val, y_pred_class, average='weighted')
        
        # 성능 지표 로깅
        mlflow.log_metrics({
            "val_accuracy": accuracy,
            "val_f1": f1
        })
        
        # 모델 로깅
        mlflow.xgboost.log_model(
            candidate_model, 
            "candidate_model",
            registered_model_name="wine_xgboost_candidate"
        )
        
        candidate_model_uri = f"runs:/{mlflow.active_run().info.run_id}/candidate_model"
        print(f"후보 모델 정확도: {accuracy:.4f}")
        print(f"후보 모델 F1 점수: {f1:.4f}")
    
    # 평가 데이터 준비
    eval_data = pd.DataFrame(X_test, columns=feature_names)
    eval_data['label'] = y_test
    
    # 품질 게이트 정의 - 모델이 충족해야 하는 최소 성능 임계값
    quality_thresholds = {
        "accuracy_score": MetricThreshold(threshold=0.85, greater_is_better=True),
        "f1_score": MetricThreshold(threshold=0.80, greater_is_better=True),
        "roc_auc_score": MetricThreshold(threshold=0.85, greater_is_better=True)
    }
    
    # 개선 임계값 - 후보 모델이 베이스라인보다 얼마나 더 나아야 하는지
    improvement_thresholds = {
        "f1_score": MetricThreshold(threshold=0.02, greater_is_better=True)  # F1 점수가 2% 이상 개선되어야 함
    }
    
    try:
        print("\n모델 품질 검증 시작...")
        
        # 후보 모델 평가
        candidate_result = mlflow.evaluate(
            candidate_model_uri, 
            eval_data, 
            targets="label",
            model_type="classifier"
        )
        
        # 베이스라인 모델 평가
        baseline_result = mlflow.evaluate(
            baseline_model_uri, 
            eval_data, 
            targets="label",
            model_type="classifier"
        )
        
        print("\n품질 게이트 검증:")
        
        # 품질 게이트 검증
        try:
            mlflow.validate_evaluation_results(
                candidate_result=candidate_result,
                validation_thresholds=quality_thresholds
            )
            print("✅ 후보 모델이 모든 품질 임계값을 충족합니다.")
        except Exception as e:
            print(f"❌ 모델 검증 실패: {e}")
        
        print("\n개선 임계값 검증:")
        
        # 개선 임계값 검증
        try:
            mlflow.validate_evaluation_results(
                candidate_result=candidate_result,
                baseline_result=baseline_result,
                validation_thresholds=improvement_thresholds
            )
            print("✅ 후보 모델이 베이스라인 대비 충분히 개선되었습니다.")
        except Exception as e:
            print(f"❌ 모델 개선 검증 실패: {e}")
        
        # 최종 결과 저장
        with mlflow.start_run(run_name="model_comparison_results"):
            # 태그 설정
            set_experiment_tags("classification", "model_comparison")
            
            # 비교 결과 로깅
            mlflow.log_metrics({
                "baseline_accuracy": baseline_result.metrics["accuracy_score"],
                "candidate_accuracy": candidate_result.metrics["accuracy_score"],
                "baseline_f1": baseline_result.metrics["f1_score"],
                "candidate_f1": candidate_result.metrics["f1_score"],
                "accuracy_improvement": candidate_result.metrics["accuracy_score"] - baseline_result.metrics["accuracy_score"],
                "f1_improvement": candidate_result.metrics["f1_score"] - baseline_result.metrics["f1_score"]
            })
            
            # 비교 시각화
            metrics = ["accuracy_score", "f1_score", "roc_auc_score"]
            baseline_values = [baseline_result.metrics[m] for m in metrics]
            candidate_values = [candidate_result.metrics[m] for m in metrics]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(metrics))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], baseline_values, width, label='베이스라인 모델')
            ax.bar([i + width/2 for i in x], candidate_values, width, label='후보 모델')
            
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.set_ylabel('점수')
            ax.set_title('모델 성능 비교')
            ax.legend()
            
            plt.tight_layout()
            mlflow.log_figure(fig, "model_comparison.png")
    
    except Exception as e:
        print(f"모델 평가 중 오류 발생: {e}")

def main():
    """
    MLflow와 XGBoost를 활용한 하이퍼파라미터 튜닝 예제를 실행합니다.
    """
    print("===== MLflow와 XGBoost 하이퍼파라미터 튜닝 예제 =====")
    
    # 각 튜닝 방법 실행
    grid_search_tuning()        # 그리드 서치
    bayesian_optimization_tuning()  # 베이지안 최적화
    model_quality_gates()       # 모델 품질 게이트
    
    print("\n===== 모든 예제가 성공적으로 실행되었습니다 =====")
    print("\nMLflow UI를 확인하려면 터미널에서 다음 명령어를 실행하세요:")
    print("mlflow ui")
    print("그리고 브라우저에서 http://localhost:5000 을 방문하세요.\n")
    print("이 예제들은 MLflow 공식 가이드를 기반으로 작성되었습니다:")
    print("https://mlflow.org/docs/latest/ml/traditional-ml/xgboost/guide/")

if __name__ == "__main__":
    main()