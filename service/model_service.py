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

    def get_prediction(self, news_id: str):
        return PredictionRepository().get_news_prediction(news_id=news_id)

    def save_prediction(self, url: str, date: int, time: int):
        news_data = NewsService().get_news_data(url=url)
        # topic = LDA().get_prediction(news_data)

        # price_data = 회귀모델().get_prediction(topic)
        stock_price = StockRepository().get_stock_data_by_date(date=date, time=time)
        news_id = url - "https://v.daum.net/v/"
        news_prediction = NewsPrediction()
        news_prediction.news_id = news_id
        news_prediction.min_1 = 1
        news_prediction.min_5 = 5
        news_prediction.min_15 = 15
        news_prediction.min_60 = 60
        news_prediction.day_1 = stock_price

        res = PredictionRepository().save_news_prediction(news_prediction=news_prediction)
        return res

    def fetch_and_store_news_every_minute(self):
        time_now = datetime.datetime.now().time()
        time_now = time_now.strftime("%H%M")
        news_url_list = NewsService().get_news_list_min(item_name="삼성전자", time_now=time_now)
        for url in news_url_list:
            self.save_prediction(url=url, date="", time=time_now)
