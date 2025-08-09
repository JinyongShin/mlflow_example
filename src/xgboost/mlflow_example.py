"""
MLflow 실습 예제: XGBoost 모델 학습 및 추적
"""

# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow 실험 이름 설정
mlflow.set_experiment("wine-xgboost-classification")

def plot_feature_importance(model):
    """
    XGBoost 모델의 특성 중요도를 시각화합니다.
    
    Args:
        model: 학습된 XGBoost 모델
    
    Returns:
        matplotlib 그림 객체
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(
        model,
        importance_type='gain',
        ax=ax,
        title='특성 중요도 (gain 기준)',
        xlabel='중요도 점수',
        ylabel='특성',
        max_num_features=15
    )
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    혼동 행렬을 시각화합니다.
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
    
    Returns:
        matplotlib 그림 객체
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.title('혼동 행렬')
    plt.close(fig)
    return fig

def main():
    # Wine 데이터셋 로드
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    class_names = wine.target_names
    
    # 훈련 및 테스트 세트로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost용 DMatrix 변환
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    
    # MLflow 자동 로깅 활성화
    mlflow.xgboost.autolog()
    
    # MLflow 실행 시작 및 모델 학습
    with mlflow.start_run() as run:
        # 실행 시작 메시지
        print(f"MLflow 실행 시작: {run.info.run_id}")
        
        # 모델 하이퍼파라미터 정의
        params = {
            "objective": "multi:softprob",
            "num_class": len(class_names),
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",  # 빠른 학습을 위한 히스토그램 기반 알고리즘
            "random_state": 42
        }
        
        # 하이퍼파라미터 로깅
        mlflow.log_params(params)
        
        # 평가 결과를 저장할 딕셔너리
        evals_result = {}
        
        # 모델 생성 및 학습
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            early_stopping_rounds=10,
            evals=[(dtrain, "train"), (dtest, "test")],
            evals_result=evals_result,
            verbose_eval=10
        )
        
        # 학습 과정 로깅
        for epoch, (train_metric, test_metric) in enumerate(
            zip(evals_result["train"]["mlogloss"], evals_result["test"]["mlogloss"])
        ):
            mlflow.log_metrics({
                "train_logloss": train_metric,
                "test_logloss": test_metric
            }, step=epoch)
        
        # 모델 예측 및 평가
        y_pred = model.predict(dtest)
        y_pred_class = np.argmax(y_pred, axis=1)
        
        # 평가 지표 계산
        accuracy = accuracy_score(y_test, y_pred_class)
        precision = precision_score(y_test, y_pred_class, average='weighted')
        recall = recall_score(y_test, y_pred_class, average='weighted')
        f1 = f1_score(y_test, y_pred_class, average='weighted')
        
        # 평가 지표 로깅
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        
        # 특성 중요도 시각화 및 로깅
        feature_imp_fig = plot_feature_importance(model)
        mlflow.log_figure(feature_imp_fig, "feature_importance.png")
        
        # 혼동 행렬 시각화 및 로깅
        cm_fig = plot_confusion_matrix(y_test, y_pred_class, class_names)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        
        # 특성 중요도를 딕셔너리로 변환하여 로깅
        feature_importance = model.get_score(importance_type='gain')
        mlflow.log_dict(feature_importance, "feature_importance.json")
        
        # 모델 서명(signature) 생성
        # DataFrame을 직접 사용하여 특성 이름 정보 유지
        X_sample = X_train.iloc[:5]  # DataFrame을 값으로 변환하지 않음
        # DMatrix에 feature_names 명시적 제공
        y_pred_sample = model.predict(xgb.DMatrix(X_sample.values, feature_names=feature_names))
        signature = infer_signature(X_sample, y_pred_sample)
        
        # 모델 로깅 - input_example에 DataFrame을 사용하여 특성 이름 정보 유지
        mlflow.xgboost.log_model(
            model, 
            "xgboost_model",
            signature=signature,
            input_example=X_train.iloc[:5]  # numpy 배열이 아닌 DataFrame 사용
        )
        
        # 모델을 MLflow 모델 레지스트리에 등록
        model_uri = f"runs:/{run.info.run_id}/xgboost_model"
        model_name = "wine_xgboost_classifier"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        print(f"모델이 성공적으로 등록되었습니다: {model_name}, 버전: {registered_model.version}")
        print(f"정확도: {accuracy:.4f}")
        print(f"F1 점수: {f1:.4f}")

if __name__ == "__main__":
    main()
    
    # MLflow UI 접속 방법 안내
    print("\nMLflow UI를 확인하려면 터미널에서 다음 명령어를 실행하세요:")
    print("mlflow ui")
    print("그리고 브라우저에서 http://localhost:5000 을 방문하세요.\n")
