from sqlalchemy import select

from model.news_prediction import NewsPrediction
from repository.base_repository import BaseRepository


class PredictionRepository(BaseRepository):

    def get_news_prediction(self, news_id: int):

        stmt = select(NewsPrediction).where(NewsPrediction.news_id == news_id)
        res = self.session.execute(stmt)
        res = res.scalar()
        return res

    def save_news_prediction(self, news_prediction: NewsPrediction):

        self.session.add(news_prediction)

        return news_prediction
