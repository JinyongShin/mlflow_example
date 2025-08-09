"""
MLflow와 함께 XGBoost 모델의 하이퍼파라미터 튜닝 수행하기
"""

import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature
import xgboost as xgb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names, top_n=15):
    """
    XGBoost 모델의 특성 중요도를 시각화합니다.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(
        model,
        importance_type='gain',
        ax=ax,
        title='특성 중요도 (gain 기준)',
        xlabel='중요도 점수',
        ylabel='특성',
        max_num_features=top_n
    )
    plt.tight_layout()
    return fig

def hyperparameter_tuning():
    """
    그리드 서치 방식으로 XGBoost 모델의 하이퍼파라미터 튜닝을 수행하고 MLflow로 모든 실험을 추적합니다.
    """
    # MLflow 실험 생성
    experiment_name = "wine-xgboost-hyperparameter-tuning"
    mlflow.set_experiment(experiment_name)
    
    # Wine 데이터셋 로드
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    
    # 훈련 및 테스트 세트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    
    combination_count = 0
    
    for learning_rate in learning_rate_list:
        for max_depth in max_depth_list:
            for subsample in subsample_list:
                for colsample_bytree in colsample_bytree_list:
                    combination_count += 1
                    print(f"조합 {combination_count}/{total_combinations} 평가 중...")
                    
                    # 현재 하이퍼파라미터 조합
                    params = {
                        "objective": "multi:softprob",
                        "num_class": 3,  # Wine 데이터셋의 클래스 수
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "subsample": subsample,
                        "colsample_bytree": colsample_bytree,
                        "tree_method": "hist",  # 빠른 학습을 위한 히스토그램 기반 알고리즘
                        "random_state": 42
                    }
                    
                    # MLflow 실행 시작
                    with mlflow.start_run(run_name=f"run_{combination_count}"):
                        # 하이퍼파라미터 로깅
                        mlflow.log_params(params)
                        
                        # XGBoost용 데이터 변환
                        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
                        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
                        
                        # K-Fold 교차 검증 수행
                        kf = KFold(n_splits=5, shuffle=True, random_state=42)
                        cv_scores = []
                        
                        for train_idx, valid_idx in kf.split(X_train):
                            # 교차 검증용 분할
                            X_cv_train = X_train.iloc[train_idx]
                            y_cv_train = y_train.iloc[train_idx]
                            X_cv_valid = X_train.iloc[valid_idx]
                            y_cv_valid = y_train.iloc[valid_idx]
                            
                            # XGBoost용 DMatrix 변환
                            d_cv_train = xgb.DMatrix(X_cv_train, label=y_cv_train, feature_names=feature_names)
                            d_cv_valid = xgb.DMatrix(X_cv_valid, label=y_cv_valid, feature_names=feature_names)
                            
                            # 모델 학습
                            model_cv = xgb.train(
                                params,
                                d_cv_train,
                                num_boost_round=100,
                                early_stopping_rounds=10,
                                evals=[(d_cv_valid, 'validation')],
                                verbose_eval=False
                            )
                            
                            # 검증 세트 예측
                            y_cv_pred = model_cv.predict(d_cv_valid)
                            y_cv_pred_class = np.argmax(y_cv_pred, axis=1)
                            
                            # 정확도 계산
                            accuracy = accuracy_score(y_cv_valid, y_cv_pred_class)
                            cv_scores.append(accuracy)
                        
                        # 교차 검증 결과 계산
                        cv_mean = np.mean(cv_scores)
                        cv_std = np.std(cv_scores)
                        
                        # 교차 검증 결과 로깅
                        mlflow.log_metric("cv_mean_accuracy", cv_mean)
                        mlflow.log_metric("cv_std_accuracy", cv_std)
                        
                        # 전체 훈련 세트로 모델 훈련
                        eval_results = {}
                        model = xgb.train(
                            params,
                            dtrain,
                            num_boost_round=100,
                            early_stopping_rounds=10,
                            evals=[(dtrain, 'train'), (dtest, 'test')],
                            evals_result=eval_results,
                            verbose_eval=False
                        )
                        
                        # 학습 과정 로깅
                        for epoch, (train_metric, test_metric) in enumerate(
                            zip(eval_results["train"]["mlogloss"], eval_results["test"]["mlogloss"])
                        ):
                            mlflow.log_metrics({
                                "train_logloss": train_metric,
                                "test_logloss": test_metric
                            }, step=epoch)
                        
                        # 테스트 세트에서 성능 평가
                        y_pred = model.predict(dtest)
                        y_pred_class = np.argmax(y_pred, axis=1)
                        
                        # 성능 지표 계산
                        test_accuracy = accuracy_score(y_test, y_pred_class)
                        test_precision = precision_score(y_test, y_pred_class, average='weighted')
                        test_recall = recall_score(y_test, y_pred_class, average='weighted')
                        test_f1 = f1_score(y_test, y_pred_class, average='weighted')
                        
                        # 테스트 성능 로깅
                        mlflow.log_metrics({
                            "test_accuracy": test_accuracy,
                            "test_precision": test_precision,
                            "test_recall": test_recall,
                            "test_f1": test_f1
                        })
                        
                        # 최고 성능 모델 갱신
                        if test_accuracy > best_accuracy:
                            best_accuracy = test_accuracy
                            best_params = params.copy()
                            best_model = model
                        
                        # 특성 중요도 시각화 및 로깅
                        feature_imp_fig = plot_feature_importance(model, feature_names)
                        mlflow.log_figure(feature_imp_fig, "feature_importance.png")
                        
                        # 특성 중요도를 딕셔너리로 변환하여 로깅
                        feature_importance = model.get_score(importance_type='gain')
                        mlflow.log_dict(feature_importance, "feature_importance.json")
                        
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
        
        # 특성 중요도 시각화 및 로깅
        feature_imp_fig = plot_feature_importance(best_model, feature_names)
        mlflow.log_figure(feature_imp_fig, "feature_importance.png")
        
        # DataFrame을 직접 사용하여 특성 이름 정보 유지
        X_sample = X_train.iloc[:5]  # DataFrame을 값으로 변환하지 않음
        # DMatrix에 feature_names 명시적 제공
        y_pred_sample = best_model.predict(xgb.DMatrix(X_sample.values, feature_names=feature_names))
        
        # 모델 서명(signature) 생성 - DataFrame을 사용하여 특성 정보 유지
        signature = infer_signature(X_sample, y_pred_sample)
        
        # 최적 모델 로깅 - input_example에 DataFrame을 사용하여 특성 이름 정보 유지
        mlflow.xgboost.log_model(
            best_model, 
            "best_xgboost_model",
            signature=signature,
            input_example=X_train.iloc[:5]  # numpy 배열이 아닌 DataFrame 사용
        )
        
        # 모델을 MLflow 모델 레지스트리에 등록
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_xgboost_model"
        model_name = "wine_xgboost_classifier_optimized"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        print(f"최적 모델이 성공적으로 등록되었습니다: {model_name}, 버전: {registered_model.version}")

if __name__ == "__main__":
    hyperparameter_tuning()
    
    # MLflow UI 접속 방법 안내
    print("\nMLflow UI를 확인하려면 터미널에서 다음 명령어를 실행하세요:")
    print("mlflow ui")
    print("그리고 브라우저에서 http://localhost:5000 을 방문하세요.\n")
