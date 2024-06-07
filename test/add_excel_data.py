from datetime import datetime

import pandas as pd
from sqlalchemy import select

from config import create_db, get_session
from model.news import News
from model.stock import Stock


class AddExcel:
    def add_excel(self):
        create_db()
        file_path = "/home/tako4/capstone/backend/Backend/data/ybdata.csv"
        session = get_session()

        df = pd.read_csv(file_path)
        num_rows = len(df)
        # print(df)
        for i in range(1, num_rows):
            content = df.iloc[i]["content"]
            date = df.iloc[i]["date"]
            if date != "date":
                try:
                    date_time = datetime.strptime(date, "%Y. %m. %d. %H:%M")

                    news_data = News()
                    news_data.news_id = None
                    news_data.date_time = date_time
                    news_data.title = None
                    news_data.writer = None
                    news_data.content = content

                    session.add(news_data)
                except Exception as e:
                    print(e)

        session.commit()

    def to_csv(self):
        session = get_session()
        stmt = select(Stock)
        date_list = session.execute(stmt)
        date_list = date_list.scalars().all()

        df = pd.DataFrame([item.__dict__ for item in date_list])
        if "_sa_instance_state" in df.columns:
            df = df.drop(columns=["_sa_instance_state"])
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        df = df.sort_values(by=["time", "date"])
        file_path = "/home/tako4/capstone/backend/Backend/data/stock_data.csv"
        df.to_csv(file_path)


excel = AddExcel()
excel.add_excel()
