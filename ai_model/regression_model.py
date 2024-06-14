import pickle
from joblib import dump, load
import os

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.pipeline import Pipeline

from gensim.models import LdaModel
from ai_model.lda_model import LDAModel
from ai_model.utils import adjust_time, calculate_price_change, count_subdirectories

from ai_model.constants import BaseConfig, model_weights_path, RegressionModelConfig, VolaConfig

pd.options.mode.chained_assignment = None

"""
회귀 분석 모델
- 훈련 함수가 호출될 시 저장되어 있는 토픽별 가중치들을 활용하여 가장 성능이 좋은 모델 저장
- Ridge, Lasso 중 성능이 좋은 모델을 메타데이터와 함께 저장
- ex) {"model_type": "Ridge", "model_params":model.get_params()}
- topic 별 폴더에 시간대마다 회귀 모델 저장
    - topic_1
        ㄴ dictionary_1.pkl               : 해당 토픽 내의 말뭉치 사전
        ㄴ lda_weights_1                  : 해당 토픽의 lda 모델의 가중치
        ㄴ regression_weights_1m.joblib      : 1분 후 회귀 모델
        ㄴ regression_weights_5m.joblib      : 5분 후 회귀 모델
        ㄴ regression_weights_15m.joblib     : 15분 후 회귀 모델
        ㄴ regression_weights_1h.joblib      : 1시간 회귀 모델
        ㄴ regression_weights_1d.joblib      : 1일 후 회귀 모델
"""


class RegressionModel:

    def __init__(self, stock_dataset: pd.DataFrame = None, lda_model: LDAModel = None):
        """
        Args:
            stock_dataset: 종목의 1분봉 주가 데이터
        """
        self.stock_dataset = stock_dataset
        self.grouped_dfs = lda_model.get_group_df() if lda_model else None
        self.folder_count = count_subdirectories(model_weights_path)
        self.folder_prefix = "model_weights_"
        self.folder_index = self.folder_count
        self.folder_name = self.folder_prefix + str(self.folder_index)
        self.folder_path = os.path.join(model_weights_path, self.folder_name)
        print(__name__, "생성")

    def train_regression_model(self, num_topics, folder_path):
        """
        1. 사전, 코퍼스 불러오기
        2. 모델 불러오기
        3. 기존 코퍼스의 토픽 분포 획득
        4. 해당 토픽 확률 활용, 회귀 분석 실시
        """
        try:
            avg_score = []
            print("회귀 모델 학습 시작")
            for topic_idx in range(num_topics):
                # 사전, 코퍼스, 기존 lda 모델 통해서 토픽 분포 획득
                topic_distributions = self._get_topic_distributions(topic_idx=topic_idx, folder_path=folder_path)
                # 토픽 분포를 활용하여 최적의 회귀 모델 저장
                best_score_by_topic = self._get_best_performance_regression_model_and_save(
                    topic_idx=topic_idx, topic_distributions=topic_distributions, folder_path=folder_path
                )
                avg_score.append(best_score_by_topic)
            print("회귀 모델 학습 종료")
            # 모든 토픽의 결과를 하나의 데이터프레임으로 결합
            final_results_df = pd.concat(avg_score, ignore_index=True)

            # 결과 정렬 및 출력
            final_results_df = final_results_df.sort_values(by=["topic", "volatility"]).reset_index(drop=True)

            # 결과 저장
            final_results_file_name = "model_performance.csv"
            final_results_file_path = os.path.join(folder_path, final_results_file_name)
            final_results_df.to_csv(final_results_file_path, index=False)

            return final_results_df

        except Exception as e:
            print("Error of _get_num_of_topics_by_group method:", e)

    def _get_topic_distributions(self, topic_idx, folder_path):

        try:
            model_name = f"topic_{topic_idx+1}/lda_model_{topic_idx+1}.model"
            model_path = os.path.join(folder_path, model_name)
            model = LdaModel.load(model_path)

            corpus_name = f"topic_{topic_idx+1}/corpus_{topic_idx+1}.pkl"
            corpus_path = os.path.join(folder_path, corpus_name)
            with open(corpus_path, "rb") as f:
                corpus = pickle.load(f)

            return [model.get_document_topics(doc_bow) for doc_bow in corpus]

        except Exception as e:
            print("Error of _get_num_of_topics_by_group method:", e)

    def _get_best_performance_regression_model_and_save(self, topic_idx, topic_distributions, folder_path):

        try:
            model_name = f"topic_{topic_idx+1}/lda_model_{topic_idx+1}.model"
            model_path = os.path.join(folder_path, model_name)
            model = LdaModel.load(model_path)

            temp_group = self.grouped_dfs[topic_idx]
            self._get_stock_price_changes_by_date_time(temp_group=temp_group)
            self._get_topic_features(temp_group=temp_group, topic_distributions=topic_distributions, num_topics=model.num_topics)
            best_score_by_topic = self._get_best_performance_regression_model(
                temp_group=temp_group, topic_idx=topic_idx, folder_path=folder_path
            )
            return best_score_by_topic

        except Exception as e:
            print("Error of _get_num_of_topics_by_group method:", e)

    def _get_stock_price_changes_by_date_time(self, temp_group):

        try:
            temp_group.loc[:, "date_time"] = pd.to_datetime(temp_group.loc[:, "date_time"])
            temp_group.loc[:, "adjusted_time"] = temp_group.loc[:, "date_time"].apply(adjust_time)

            for minutes in VolaConfig.TIME_INTERVALS:
                temp_group.loc[:, f"vola_{minutes}m"] = temp_group.loc[:, "adjusted_time"].apply(
                    lambda x: calculate_price_change(x, minutes, self.stock_dataset)
                )

        except Exception as e:
            print("Error of _get_num_of_topics_by_group method:", e)

    def _get_topic_features(self, temp_group, topic_distributions, num_topics):

        try:
            topic_dist_list = []
            for dist in topic_distributions:
                dist_dict = dict(dist)
                topic_dist_list.append([dist_dict.get(i, 0.0) for i in range(num_topics)])

            topic_df = pd.DataFrame(topic_dist_list, columns=[f"topic_{i}" for i in range(num_topics)])

            temp_group = pd.concat([temp_group, topic_df], axis=1)

        except Exception as e:
            print("Error of _get_num_of_topics_by_group method:", e)

    def _get_best_performance_regression_model(self, temp_group, topic_idx, folder_path):

        try:
            results = []

            for vola in VolaConfig.VOLA_COLUMNS:
                # 결측값 제거
                data_clean = temp_group.dropna(subset=[vola])

                if data_clean.empty:
                    continue

                X = data_clean.drop(columns=VolaConfig.VOLA_COLUMNS + ["category", "date_time", "documents", "adjusted_time"])
                y = data_clean[vola]

                # 훈련/테스트 데이터 분할
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=BaseConfig.TEST_SIZE, random_state=BaseConfig.RANDOM_STATE
                )

                # Ridge 회귀 모델 파이프라인
                ridge_pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])

                # Lasso 회귀 모델 파이프라인
                lasso_pipeline = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso())])

                # GridSearchCV 설정
                ridge_grid = GridSearchCV(
                    ridge_pipeline, RegressionModelConfig.RIDGE_PARAMETERS, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
                )
                lasso_grid = GridSearchCV(
                    lasso_pipeline, RegressionModelConfig.LASSO_PARAMETERS, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
                )

                # print(f"GridSearchCV 설정 - ridge: {ridge_grid}")
                # print(f"GridSearchCV 설정 - lasso: {lasso_grid}")

                # 모델 훈련
                ridge_grid.fit(X_train, y_train)
                lasso_grid.fit(X_train, y_train)

                # 최적 모델 선택
                best_ridge = ridge_grid.best_estimator_
                best_lasso = lasso_grid.best_estimator_

                # print(f"GridSearchCV 최적 모델 - ridge: {best_ridge}")
                # print(f"GridSearchCV 최적 모델 - lasso: {best_lasso}")

                # 예측
                y_pred_ridge = best_ridge.predict(X_test)
                y_pred_lasso = best_lasso.predict(X_test)

                # 부호 일치 여부 확인
                ridge_correct_sign = (y_pred_ridge > 0) == (y_test > 0)
                lasso_correct_sign = (y_pred_lasso > 0) == (y_test > 0)

                # 부호 일치 정확도 계산
                ridge_sign_accuracy = accuracy_score(ridge_correct_sign, [True] * len(y_test))
                lasso_sign_accuracy = accuracy_score(lasso_correct_sign, [True] * len(y_test))

                # MAE 계산
                ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
                lasso_mae = mean_absolute_error(y_test, y_pred_lasso)

                model_name = f"topic_{topic_idx+1}/reg_model_{vola}.joblib"
                model_path = os.path.join(folder_path, model_name)
                if ridge_sign_accuracy < lasso_sign_accuracy:
                    with open(model_path, "wb") as f:
                        dump(best_lasso, f)
                else:
                    with open(model_path, "wb") as f:
                        dump(best_ridge, f)

                if ridge_sign_accuracy < lasso_sign_accuracy:
                    best_sign_accuracy = lasso_sign_accuracy
                else:
                    best_sign_accuracy = ridge_sign_accuracy

                if ridge_mae < lasso_mae:
                    best_mae = ridge_mae
                else:
                    best_mae = lasso_mae

                results.append(
                    {
                        "topic": topic_idx + 1,
                        "volatility": vola,
                        "best_sign_accuracy": best_sign_accuracy,
                        "best_mae": best_mae,
                    }
                )

            # 결과를 데이터프레임으로 변환
            results_df = pd.DataFrame(results)

            # 결과 해당 폴더에 저장ㄴ
            results_file_name = f"topic_{topic_idx+1}/model_performance.csv"
            results_file_path = os.path.join(folder_path, results_file_name)
            results_df.to_csv(results_file_path, index=False)

            return results_df

        except Exception as e:
            print("Error of _get_num_of_topics_by_group method:", e)

    def get_stock_volatilities(self, group_id, topic_distributions, folder_path):

        try:
            stock_volatilities = []

            for vola in VolaConfig.VOLA_COLUMNS:
                model_name = f"topic_{group_id+1}/reg_model_{vola}.joblib"
                model_path = os.path.join(folder_path, model_name)

                with open(model_path, "rb") as f:
                    model = load(f)

                print(f"Loaded model: {model}")

                stock_volatilities.append(model.predict(topic_distributions)[0])

            return stock_volatilities

        except Exception as e:
            print("Error of _get_num_of_topics_by_group method:", e)
