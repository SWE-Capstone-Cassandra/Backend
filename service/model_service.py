import datetime
from typing import List

from ai_model.data_controller import DataController
from model.news_prediction import NewsPrediction
from repository.news_repository import NewsRepository
from repository.prediction_repository import PredictionRepository
from repository.stock_repository import StockRepository
from service.base_service import BaseService
from service.news_service import NewsService


class ModelService(BaseService):
    """

    1. 자동 학습 요청 서비스
    2. 주가 변화량 예측 요청 서비스
    """

    def request_training(self):
        news_dataset = NewsRepository(session=self.session).get_news_dataset()
        stock_dataset = StockRepository(session=self.session).get_stock_dataset()
        try:
            DataController().train_news_dataset(news_dataset=news_dataset, stock_dataset=stock_dataset)
        except Exception as ex:
            print("무슨 에러지?", ex)

    def request_stock_volatilities(self, content: str) -> List[float]:
        return DataController().predict_stock_volatilities(text=content)

    def get_prediction_by_news_id(self, news_id: int):
        return PredictionRepository().get_news_prediction_by_news_id(news_id=news_id)

    def save_prediction(self, news_id: int, prediction: List[float]):
        news_prediction = NewsPrediction()
        news_prediction.news_id = news_id
        news_prediction.min_1 = prediction[0] if len(prediction) > 0 else 0
        news_prediction.min_5 = prediction[1] if len(prediction) > 1 else 0
        news_prediction.min_15 = prediction[2] if len(prediction) > 2 else 0
        news_prediction.min_60 = prediction[3] if len(prediction) > 3 else 0
        news_prediction.day_1 = prediction[4] if len(prediction) > 4 else 0

        res = PredictionRepository().save_news_prediction(news_prediction=news_prediction)
        return res

    def fetch_and_store_news_every_minute(self):
        time_now = datetime.datetime.now().time()
        time_now = time_now.strftime("%H%M")
        news_url_list = NewsService().get_news_list_min(item_name="삼성전자", time_now=time_now)
        for url in news_url_list:
            self.save_prediction(url=url, date="", time=time_now)
