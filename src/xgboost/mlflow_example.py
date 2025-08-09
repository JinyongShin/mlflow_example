"""
MLflow와 XGBoost 통합 가이드 예제

이 스크립트는 MLflow 공식 가이드(https://mlflow.org/docs/latest/ml/traditional-ml/xgboost/guide/)의
내용을 기반으로 XGBoost 모델의 실험 추적, 모델 관리, 배포 예제를 제공합니다.

두 가지 API 접근 방식을 모두 보여줍니다:
1. 네이티브 XGBoost API - 최대 성능과 세부 제어 가능
2. scikit-learn API - 사용하기 쉽고 sklearn 파이프라인과 통합 가능
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
import mlflow.sklearn
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_diabetes, load_digits, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)
from mlflow.models import infer_signature

# 재현성을 위한 랜덤 시드 설정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# 환경 정보
PYTHON_VERSION = platform.python_version()
XGBOOST_VERSION = xgb.__version__
PLATFORM = platform.platform()

class MLflowCallback(xgb.callback.TrainingCallback):
    """
    XGBoost 학습 과정을 MLflow에 실시간으로 로깅하기 위한 커스텀 콜백 클래스
    """
    def __init__(self):
        self.metrics_history = []
        
    def after_iteration(self, model, epoch, evals_log):
        """매 반복 학습 후 지표를 로깅합니다."""
        metrics = {}
        for dataset, metric_dict in evals_log.items():
            for metric_name, values in metric_dict.items():
                key = f"{dataset}_{metric_name}"
                metrics[key] = values[-1]  # 최신 값
        
        # MLflow에 지표 기록
        mlflow.log_metrics(metrics, step=epoch)
        self.metrics_history.append(metrics)
        
        # 체크포인트 저장 (특정 주기마다)
        if epoch > 0 and epoch % 50 == 0:
            temp_model_path = f"checkpoint_epoch_{epoch}.json"
            model.save_model(temp_model_path)
            mlflow.log_artifact(temp_model_path)
            # 임시 파일 삭제
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
                
        return False  # 학습 계속 진행

def plot_feature_importance(model, feature_names=None, top_n=15, importance_type="gain"):
    """
    XGBoost 모델의 특성 중요도를 시각화합니다.
    
    Args:
        model: 학습된 XGBoost 모델
        feature_names: 특성 이름 목록
        top_n: 표시할 상위 특성 개수
        importance_type: 중요도 유형 ('weight', 'gain', 'cover', 'total_gain', 'total_cover')
    
    Returns:
        matplotlib 그림 객체
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(
        model,
        importance_type=importance_type,
        ax=ax,
        title=f'특성 중요도 ({importance_type} 기준)',
        xlabel='중요도 점수',
        ylabel='특성',
        max_num_features=top_n,
        grid=False
    )
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    혼동 행렬을 시각화합니다.
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
    
    Returns:
        matplotlib 그림 객체
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.title('혼동 행렬')
    return fig

def set_experiment_tags():
    """실험에 태그를 설정하여 추적을 용이하게 합니다."""
    mlflow.set_tags({
        "python_version": PYTHON_VERSION,
        "xgboost_version": XGBOOST_VERSION,
        "platform": PLATFORM,
        "random_state": RANDOM_STATE,
        "model_type": "gradient_boosting",
        "algorithm": "xgboost"
    })

def example_native_api_regression():
    """
    네이티브 XGBoost API를 사용한 회귀 모델 예제
    - diabetes 데이터셋 사용
    - MLflow autologging 활용
    """
    print("\n=== 네이티브 XGBoost API 회귀 예제 ===")
    
    # MLflow 실험 설정
    mlflow.set_experiment("xgboost-native-regression")
    
    # 당뇨병 데이터셋 로드
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # 훈련 및 테스트 세트로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # XGBoost용 DMatrix 변환
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    
    # MLflow 자동 로깅 활성화
    mlflow.xgboost.autolog()
    
    # MLflow 실행 시작 및 모델 학습
    with mlflow.start_run(run_name="diabetes-regression") as run:
        print(f"MLflow 실행 시작: {run.info.run_id}")
        
        # 실험 태그 설정
        set_experiment_tags()
        
        # 모델 하이퍼파라미터 정의
        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": RANDOM_STATE
        }
        
        # 데이터셋 정보 로깅
        mlflow.log_params({
            "dataset": "diabetes",
            "dataset_size": len(X_train) + len(X_test),
            "training_size": len(X_train),
            "test_size": len(X_test),
            "n_features": X_train.shape[1],
        })
        
        # 평가 결과를 저장할 딕셔너리
        evals_result = {}
        
        # 커스텀 콜백 생성
        callback = MLflowCallback()
        
        # 모델 학습
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            early_stopping_rounds=10,
            evals=[(dtrain, "train"), (dtest, "test")],
            evals_result=evals_result,
            callbacks=[callback],
            verbose_eval=10
        )
        
        # 예측 및 평가
        y_pred = model.predict(dtest)
        
        # 평가 지표 계산 및 로깅
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metrics({
            "rmse": rmse,
            "r2": r2,
            "best_iteration": model.best_iteration,
            "best_score": model.best_score
        })
        
        # 특성 중요도 시각화 및 로깅
        feature_imp_fig = plot_feature_importance(model, feature_names)
        mlflow.log_figure(feature_imp_fig, "feature_importance.png")
        
        # 모델 서명(signature) 생성 - DataFrame을 사용하여 특성 정보 유지
        signature = infer_signature(X_train, y_pred)
        
        # 모델 로깅 - input_example에 DataFrame을 사용하여 특성 이름 정보 유지
        mlflow.xgboost.log_model(
            model, 
            "xgboost_model",
            signature=signature,
            input_example=X_train.iloc[:5],  # DataFrame 사용
            registered_model_name="diabetes_xgboost_regressor"
        )
        
        print(f"RMSE: {rmse:.4f}")
        print(f"R² 점수: {r2:.4f}")
        print(f"최적 반복 횟수: {model.best_iteration}")

def example_sklearn_api_classification():
    """
    scikit-learn API를 사용한 분류 모델 예제
    - wine 데이터셋 사용
    - sklearn 파이프라인과 통합
    """
    print("\n=== scikit-learn API 분류 예제 ===")
    
    # MLflow 실험 설정
    mlflow.set_experiment("xgboost-sklearn-classification")
    
    # Wine 데이터셋 로드
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    class_names = wine.target_names
    
    # 훈련 및 테스트 세트로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # scikit-learn API용 MLflow 자동 로깅 활성화
    mlflow.sklearn.autolog()
    
    # MLflow 실행 시작 및 모델 학습
    with mlflow.start_run(run_name="wine-classification") as run:
        print(f"MLflow 실행 시작: {run.info.run_id}")
        
        # 실험 태그 설정
        set_experiment_tags()
        mlflow.set_tag("model_flavor", "sklearn_xgboost")
        
        # 모델 생성
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(class_names),
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            tree_method="hist",
            early_stopping_rounds=10,
            eval_metric="mlogloss"
        )
        
        # 모델 학습 (평가 세트 제공)
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=10
        )
        
        # 모델 예측
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # 분류 보고서 계산 및 로깅
        report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # 클래스별 성능 지표 로깅
        for class_label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                mlflow.log_metrics({
                    f"class_{class_label}_precision": metrics["precision"],
                    f"class_{class_label}_recall": metrics["recall"],
                    f"class_{class_label}_f1": metrics["f1-score"],
                })
        
        # 혼동 행렬 시각화 및 로깅
        cm_fig = plot_confusion_matrix(y_test, y_pred, class_names)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        
        # 특성 중요도 시각화 및 로깅 (scikit-learn API는 특성 중요도 접근 방식이 다름)
        feature_importance = model.feature_importances_
        feat_imp_dict = dict(zip(feature_names, feature_importance))
        mlflow.log_dict(feat_imp_dict, "feature_importance.json")
        
        # 막대 그래프로 특성 중요도 시각화
        sorted_idx = np.argsort(feature_importance)
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('특성 중요도')
        plt.title('XGBoost 특성 중요도')
        mlflow.log_figure(fig, "feature_importance_sklearn.png")
        
        # 모델 성능 지표
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"정확도: {accuracy:.4f}")
        print(f"가중 F1 점수: {f1:.4f}")

def example_multiclass_classification():
    """
    다중 클래스 분류 예제
    - digits 데이터셋 사용 (10개 클래스)
    - 클래스별 성능 측정 및 시각화
    """
    print("\n=== 다중 클래스 분류 예제 (digits) ===")
    
    # MLflow 실험 설정
    mlflow.set_experiment("xgboost-multiclass")
    
    # Digits 데이터셋 로드
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # 훈련 및 테스트 세트로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # MLflow 자동 로깅 활성화
    mlflow.xgboost.autolog()
    
    with mlflow.start_run(run_name="digits-multiclass") as run:
        print(f"MLflow 실행 시작: {run.info.run_id}")
        
        # 실험 태그 설정
        set_experiment_tags()
        mlflow.set_tag("task", "multiclass_classification")
        
        # 데이터셋 정보 로깅
        mlflow.log_params({
            "dataset": "digits",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y))
        })
        
        # 네이티브 XGBoost API 사용
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # 모델 하이퍼파라미터
        params = {
            "objective": "multi:softprob",
            "num_class": 10,  # digits 데이터셋의 클래스 수
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE,
            "tree_method": "hist"
        }
        
        # 모델 학습
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            early_stopping_rounds=10,
            evals=[(dtrain, "train"), (dtest, "test")],
            verbose_eval=10
        )
        
        # 예측
        y_pred_proba = model.predict(dtest)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 분류 보고서 생성
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        # 클래스별 지표 로깅
        for class_label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                mlflow.log_metrics({
                    f"class_{class_label}_precision": metrics["precision"],
                    f"class_{class_label}_recall": metrics["recall"],
                    f"class_{class_label}_f1": metrics["f1-score"],
                })
        
        # 혼동 행렬 시각화 및 로깅
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('예측 클래스')
        plt.ylabel('실제 클래스')
        plt.title('혼동 행렬 (Digits 데이터)')
        mlflow.log_figure(fig, "confusion_matrix_digits.png")
        
        # 전체 성능 지표 로깅
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        mlflow.log_metrics({
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1
        })
        
        print(f"정확도: {accuracy:.4f}")
        print(f"매크로 F1 점수: {f1:.4f}")

def example_binary_classification_custom_logging():
    """
    이진 분류 예제와 커스텀 메트릭 로깅
    - make_classification으로 생성된 데이터셋 사용
    - 수동 로깅 방식 시연
    """
    print("\n=== 이진 분류 및 커스텀 로깅 예제 ===")
    
    # MLflow 실험 설정
    mlflow.set_experiment("xgboost-binary-custom")
    
    # 이진 분류용 데이터 생성
    X, y = make_classification(
        n_samples=10000, 
        n_features=20, 
        n_classes=2, 
        random_state=RANDOM_STATE
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # 훈련 및 테스트 세트로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # MLflow 자동 로깅 비활성화 (수동 로깅 예제를 위해)
    # MLflow의 이전 autolog 설정을 원복하려면 mlflow.autolog(disable=True) 사용
    
    # MLflow 실행 시작
    with mlflow.start_run(run_name="binary-custom-logging") as run:
        print(f"MLflow 실행 시작: {run.info.run_id}")
        
        # 실험 태그 설정
        set_experiment_tags()
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("logging_type", "manual")
        
        # 모델 하이퍼파라미터 정의 및 로깅
        params = {
            "objective": "binary:logistic",
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "random_state": RANDOM_STATE
        }
        
        training_config = {
            "num_boost_round": 500,
            "early_stopping_rounds": 50
        }
        
        # 파라미터 수동 로깅
        mlflow.log_params(params)
        mlflow.log_params(training_config)
        
        # 데이터셋 정보 로깅
        mlflow.log_params({
            "dataset": "synthetic_binary",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": 2,
            "pos_class_ratio": np.sum(y==1) / len(y)
        })
        
        # XGBoost용 DMatrix 변환
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
        
        # 평가 결과를 저장할 딕셔너리
        evals_result = {}
        
        # 모델 학습
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=training_config["num_boost_round"],
            evals=[(dtrain, "train"), (dtest, "test")],
            early_stopping_rounds=training_config["early_stopping_rounds"],
            evals_result=evals_result,
            verbose_eval=50
        )
        
        # 학습 과정 수동 로깅
        for epoch, (train_metrics, test_metrics) in enumerate(
            zip(evals_result["train"]["logloss"], evals_result["test"]["logloss"])
        ):
            mlflow.log_metrics(
                {"train_logloss": train_metrics, "test_logloss": test_metrics}, 
                step=epoch
            )
        
        # 예측 및 평가
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 성능 지표 계산 및 로깅
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "best_iteration": model.best_iteration,
            "best_score": model.best_score
        })
        
        # 특성 중요도 시각화 및 로깅
        feature_imp_fig = plot_feature_importance(model, feature_names)
        mlflow.log_figure(feature_imp_fig, "feature_importance.png")
        
        # 특성 중요도를 딕셔너리로 변환하여 로깅
        feature_importance = model.get_score(importance_type='gain')
        mlflow.log_dict(feature_importance, "feature_importance.json")
        
        # ROC 곡선 시각화 및 로깅
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, label=f'ROC 커브 (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC 곡선')
        ax.legend(loc="lower right")
        mlflow.log_figure(fig, "roc_curve.png")
        
        # 모델 서명(signature) 생성
        signature = infer_signature(X_train, y_pred_proba)
        
        # 모델 로깅
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost_model",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name="binary_classification_xgboost"
        )
        
        print(f"정확도: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"F1 점수: {f1:.4f}")
        print(f"최적 반복 횟수: {model.best_iteration}")

def main():
    """
    MLflow와 XGBoost를 함께 사용하는 다양한 예제를 실행합니다.
    """
    print("===== MLflow와 XGBoost 통합 예제 실행 =====")
    
    # 각 예제 실행
    example_native_api_regression()      # 네이티브 XGBoost API를 사용한 회귀 분석
    example_sklearn_api_classification()  # scikit-learn API를 사용한 분류
    example_multiclass_classification()   # 다중 클래스 분류
    example_binary_classification_custom_logging()  # 이진 분류 및 커스텀 로깅
    
    print("\n===== 모든 예제가 성공적으로 실행되었습니다 =====")
    print("\nMLflow UI를 확인하려면 터미널에서 다음 명령어를 실행하세요:")
    print("mlflow ui")
    print("그리고 브라우저에서 http://localhost:5000 을 방문하세요.\n")
    print("이 예제들은 MLflow 공식 가이드를 기반으로 작성되었습니다:")
    print("https://mlflow.org/docs/latest/ml/traditional-ml/xgboost/guide/")

if __name__ == "__main__":
    main()