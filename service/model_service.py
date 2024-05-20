from repository.prediction_repository import PredictionRepository
from repository.stock_repository import StockRepository
from service.news_service import NewsService
from model.news_prediction import NewsPrediction


class ModelService:
    def get_prediction(self, news_id: str):
        return PredictionRepository().get_news_prediction(news_id=news_id)

    def save_prediction(self, news_id: str, date: int, time: int):
        news_data = NewsService().get_news_data(news_id=news_id)
        topic = LDA().get_prediction(news_data)

        price_data = 회귀모델().get_prediction(topic)
        stock_price = StockRepository().get_stock_data_by_date(date=date, time=time)

        news_prediction = NewsPrediction()
        news_prediction.news_id = news_id
        news_prediction.min_1 = ""
        news_prediction.min_5 = ""
        news_prediction.min_15 = ""
        news_prediction.min_60 = ""
        news_prediction.day_1 = ""

        res = PredictionRepository().save_news_prediction(news_prediction=news_prediction)
        return res
