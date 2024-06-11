from datetime import datetime

import pandas as pd

from config import create_db, get_session
from model.stock import SamsungStock


class AddStock:
    def add_excel(self):
        create_db()
        file_path = "/home/tako4/capstone/backend/Backend/data/samsung_minute_chart_data_20220419_20240429.csv"
        session = get_session()

        df = pd.read_csv(file_path)
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
            stock = SamsungStock()
            stock.date_time = datetime.strptime(date_time + time, "%Y%m%d%H%M")
            stock.price = int(price)

            session.add(stock)
        session.commit()


excel = AddStock()
excel.add_excel()
