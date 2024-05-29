import pickle
import os

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from gensim.models import LdaModel
from ai_model.lda_model import LDAModel
from ai_model.preprocessor.stock_preprocessor import StockPreprocessor
from ai_model.utils import adjust_time, calculate_price_change

from tabulate import tabulate

"""
회귀 분석 모델
- 훈련 함수가 호출될 시 저장되어 있는 토픽별 가중치들을 활용하여 가장 성능이 좋은 모델 저장
- Ridge, Lasso 중 성능이 좋은 모델을 메타데이터와 함께 저장
- ex) {"model_type": "Ridge", "model_params":model.get_params()}
- topic 별 폴더에 시간대마다 회귀 모델 저장
    - topic_1
        ㄴ dictionary_1.pkl               : 해당 토픽 내의 말뭉치 사전
        ㄴ lda_weights_1                  : 해당 토픽의 lda 모델의 가중치
        ㄴ regression_weights_1m.pkl      : 1분 후 회귀 모델
        ㄴ regression_weights_5m.pkl      : 5분 후 회귀 모델
        ㄴ regression_weights_15m.pkl     : 15분 후 회귀 모델
        ㄴ regression_weights_1h.pkl      : 1시간 회귀 모델
        ㄴ regression_weights_1d.pkl      : 1일 후 회귀 모델
"""

model_weights_path = "/home/tako4/capstone/backend/Model/Backend/ai_model/model_weights"


class RegressionModel:
    def __init__(self, stock_datasets: pd.DataFrame = None, lda_model: LDAModel = None):
        """
        Args:
            stock_datasets: 종목의 1분봉 주가 데이터
        """
        self.stock_datasets = stock_datasets
        self.grouped_dfs = lda_model.get_group_df() if lda_model else None

    def train_regression_model(self, num_topics):
        """
        1. 사전, 코퍼스 불러오기
        2. 모델 불러오기
        3. 기존 코퍼스의 토픽 분포 획득
        4. 해당 토픽 확률 활용, 회귀 분석 실시
        """

        for topic_idx in range(num_topics):
            # 사전, 코퍼스, 기존 lda 모델 통해서 토픽 분포 획득
            topic_distribution = self._get_topic_distribution(topic_idx)
            print(f"{topic_idx}토픽 확률 분포: {topic_distribution}")
            # 토픽 분포를 활용하여 최적의 회귀 모델 저장
            # self._get_best_performance_regression_model_and_save(topic_idx=topic_idx, topic_distribution=topic_distribution)

    def _get_topic_distribution(self, topic_idx):

        model_name = f"topic_{topic_idx+1}/lda_model_{topic_idx+1}.model"
        model_path = os.path.join(model_weights_path, model_name)
        model = LdaModel.load(model_path)

        corpus_name = f"topic_{topic_idx+1}/corpus_{topic_idx+1}.pkl"
        corpus_path = os.path.join(model_weights_path, corpus_name)
        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)

        return [model.get_document_topics(doc_bow) for doc_bow in corpus]

    def _get_best_performance_regression_model_and_save(self, topic_idx, topic_distributions):

        model_name = f"topic_{topic_idx+1}/lda_model_{topic_idx+1}.model"
        model_path = os.path.join(model_weights_path, model_name)
        model = LdaModel.load(model_path)

        temp_group = self.grouped_dfs[topic_idx]
        self._get_stock_price_changes_by_publish_time(temp_group=temp_group)
        self._get_topic_features(temp_group=temp_group, topic_distributions=topic_distributions, num_topics=model.num_topics)
        print(tabulate(temp_group, headers="keys", tablefmt="fancy_outline"))
        # self._get_best_performance_regression_model()

    def _get_stock_price_changes_by_publish_time(self, temp_group):
        temp_group["publish_time"] = pd.to_datetime(temp_group["publish_time"])
        temp_group["adjusted_time"] = temp_group["publish_time"].apply(adjust_time)

        time_intervals = [1, 5, 15, 60, 1440]
        for minutes in time_intervals:
            temp_group[f"vola_{minutes}m"] = temp_group["adjusted_time"].apply(
                lambda x: calculate_price_change(x, minutes, self.stock_datasets)
            )

    def _get_topic_features(self, temp_group, topic_distributions, num_topics):
        topic_dist_list = []
        for dist in topic_distributions:
            dist_dict = dict(dist)
            topic_dist_list.append([dist_dict.get(i, 0.0) for i in range(num_topics)])

        topic_df = pd.DataFrame(topic_dist_list, columns=[f"topic_{i}" for i in range(num_topics)])

        temp_group = pd.concat([temp_group, topic_df], axis=1)

    def _get_best_performance_regression_model(self):
        pass

    def get_stock_volatilities(self, group_id, topic_distribution):
        pass
