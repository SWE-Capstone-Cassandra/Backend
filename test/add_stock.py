from datetime import datetime

import pandas as pd

from config import create_db, get_session
from model.stock import Stock


class AddStock:
    def add_excel(self):
        create_db()
        file_path = "/home/tako4/capstone/backend/Backend/data/A097950_min_chart_20240611_20220527.xlsx"
        session = get_session()

        df = pd.read_excel(file_path)
        num_rows = len(df)
        # print(df)
        for i in range(num_rows):
            date = df.iloc[i]["Date"]
            time = df.iloc[i]["Time"]
            price = df.iloc[i]["Close"]

            # date = datetime.strptime(date, "%Y. %m. %d. %H:%M")
            # date_str = date.strftime("%Y%m%d")
            # time_str = date.strftime("%H%M")

            date_time = str(date)
            time = str(time)
            stock = Stock()
            stock.date_time = datetime.strptime(date_time + time, "%Y%m%d%H%M")
            stock.price = int(price)
            stock.stock_code = "097950"
            session.add(stock)
        session.commit()

    def add_csv(self):
        file_path = "/home/tako4/capstone/backend/Backend/data/stock.csv"
        session = get_session()

        df = pd.read_csv(file_path)
        num_rows = len(df)
        # print(df)
        for i in range(num_rows):
            date_time = df.iloc[i]["date_time"]
            price = df.iloc[i]["price"]

            stock = Stock()
            stock.price = int(price)
            stock.date_time = date_time
            stock.stock_code = "005930"
            session.add(stock)
        session.commit()


excel = AddStock()
excel.add_excel()
