from sqlalchemy import select
from config import get_session, get_engine
from model.news_prediction import NewsPrediction


class PredictionRepository:
    def __init__(self) -> None:
        self.session = get_session()
        self.engine = get_engine()

    def get_news_prediction(self, news_id: int):
        with self.session() as session:
            stmt = select(NewsPrediction).where(NewsPrediction.news_id == news_id)
            res = session.execute(stmt)
            res = res.scalar()
            return res

    def save_news_prediction(self, news_prediction: NewsPrediction):

        with self.session() as session:
            session.add(news_prediction)
            session.commit()

        return news_prediction
