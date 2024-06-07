from ai_model.lda_model import LDAModel
from ai_model.regression_model import RegressionModel

from time import time
import pandas as pd
from typing import List
from ai_model.preprocessor.similar_texts_preprocessor import SimilarTextsPreprocessor


class DataController:
    def __init__(self):
        pass

    def train_news_dataset(self, news_dataset: pd.DataFrame, stock_dataset: pd.DataFrame):
        """
        주기적으로 새로운 뉴스 데이터 + 기존 데이터 세트에 대해서 LDA 재추출 및 회귀 분석 실시 API
        Args:
            news_dataset: 뉴스 데이터 세트 [필요한 컬럼 - date_time, content(documents)]
            stock_dataset: 종목 1분봉 데이터 세트 [필요한 컬럼 - date_time, price]
        """
        start = time()
        # print("########################## Start Train News Dataset! ##########################")
        # print("########################### Remove Duplicate Texts ############################")
        stp_model = SimilarTextsPreprocessor(df=news_dataset)
        preprocessed_dataset = stp_model.preprocess()
        # print()
        # print("##################################### LDA #####################################")
        lda_model = LDAModel()
        num_topics = lda_model.train_lda_model(dataset=preprocessed_dataset)
        # print()
        # print("##################################### Reg #####################################")
        avg_score = RegressionModel(stock_dataset=stock_dataset, lda_model=lda_model).train_regression_model(num_topics=num_topics)
        # print("########################### End Train News Dataset! ###########################")
        print(f"총 훈련 소요 시간: {time() - start:.2f}")
        return avg_score

    def predict_stock_volatilities(self, text) -> List[float]:
        """
        Args:
            text: 뉴스 원문.

        Returns:
            stock_volatitlities: List[float] - 주가 변화량 리스트
        """
        group_id, topic_distributions = LDAModel().get_group_id_and_topic_distribution(text=text)
        stock_volatilities = RegressionModel().get_stock_volatilities(group_id=group_id, topic_distributions=topic_distributions)
        return stock_volatilities
