from datetime import datetime

import pandas as pd

from config import create_db, get_session
from model.news import News


class AddExcel:
    def add_excel(self):
        create_db()
        file_path = "/home/tako4/capstone/backend/Backend/data/daum_samsung_20220601000000_202206292353.csv"
        session = get_session()

        df = pd.read_csv(file_path)
        num_rows = len(df)
        # print(df)
        for i in range(num_rows):
            content = df.iloc[i]["content"]
            date = df.iloc[i]["date"]

            date = datetime.strptime(date, "%Y. %m. %d. %H:%M")
            date_str = date.strftime("%Y%m%d")
            time_str = date.strftime("%H%M")

            news_data = News()
            news_data.news_url = None
            news_data.date = int(date_str)
            news_data.time = int(time_str)
            news_data.title = None
            news_data.writer = None
            news_data.content = content

            session.add(news_data)
        session.commit()


excel = AddExcel()
excel.add_excel()
