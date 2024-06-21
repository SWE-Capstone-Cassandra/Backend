from typing import List

from sqlalchemy import select

from model.news_prediction import NewsPrediction
from repository.base_repository import BaseRepository


class PredictionRepository(BaseRepository):

    def get_news_prediction_by_news_id(self, news_id: int):

        stmt = select(NewsPrediction).where(NewsPrediction.news_id == news_id)
        res = self.session.execute(stmt)
        res = res.scalar()
        return res

    def save_news_prediction(self, news_prediction: NewsPrediction):

        self.session.add(news_prediction)

        return news_prediction

    def get_prediction_by_list(self, news_list: List):
        stmt = select(NewsPrediction)

        res = self.session.execute(stmt)
        res = res.scalars().all()
        return res
