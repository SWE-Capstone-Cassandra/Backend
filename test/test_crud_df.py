import datetime
import unittest

import pandas as pd
from sqlalchemy import select


class test(unittest.TestCase):

    def test_stock(self):
        session = get_session()
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        stmt = select(News).where(News.date_time < yesterday)
        all_news = session.execute(stmt)
        all_news = all_news.scalars().all()
        df = pd.DataFrame([item.__dict__ for item in all_news])
        if "_sa_instance_state" in df.columns:
            df = df.drop(columns=["_sa_instance_state"])
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        print(df)


# yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
# stmt = select(News).where(News.date_time < yesterday)
# all_news = session.execute(stmt)
# all_news = all_news.scalars().all()
# df = pd.DataFrame([item.__dict__ for item in all_news])
# if "_sa_instance_state" in df.columns:
#     df = df.drop(columns=["_sa_instance_state"])
# if "id" in df.columns:
#     df = df.drop(columns=["id"])
# print(df["date_time"].max())

# yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
# stmt = select(Stock).where(Stock.date_time < yesterday)
# all_news = session.execute(stmt)
# all_news = all_news.scalars().all()
# df = pd.DataFrame([item.__dict__ for item in all_news])
# if "_sa_instance_state" in df.columns:
#     df = df.drop(columns=["_sa_instance_state"])
# if "id" in df.columns:
#     df = df.drop(columns=["id"])
# df.sort_values(by=["date_time"])
# print(df)

if __name__ == "__main__":
    from ai_model.data_controller import DataController
    from config import get_session
    from model.news import News
    from repository.news_repository import NewsRepository
    from repository.stock_repository import StockRepository

    session = get_session()
    news_dataset = NewsRepository(session=session).get_news_dataset()
    stock_dataset = StockRepository(session=session).get_stock_dataset()

    try:
        DataController().train_news_dataset(news_dataset=news_dataset, stock_dataset=stock_dataset)
        print("done")
    except Exception as ex:
        print("무슨 에러지?", ex)
