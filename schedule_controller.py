import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from test.testing import stock
from typing import List

from config import get_session
from service.model_service import ModelService
from service.news_service import NewsService
from utils.enum.stock_code import StockCode

FRIDAY = 0
HOUR = 60 * 60
SIX_HOUR = 60 * 60 * 6


STOCK_List = StockCode.get_list()


class ScheduleController:

    def min_data_collector(self, stock: StockCode):
        print("news collector start")
        last_min = datetime.now().minute
        while True:

            current_min = datetime.now().minute
            if not last_min == current_min:
                time.sleep(5)
                try:
                    with get_session() as session:

                        news_service = NewsService(session=session)
                        model_service = ModelService(session=session)
                        time_now = (int(datetime.today().strftime("%Y%m%d%H%M"))) * 100
                        news_min_list = news_service.get_news_list_min(item_name=stock.name, time_now=time_now)
                        print(news_min_list)
                        for news in news_min_list:
                            data = news_service.get_news_data(date_time=int(time_now / 100), url=news)
                            data.stock_code = stock.code
                            res = news_service.save_news_data(news=data)
                            prediction: List = model_service.request_stock_volatilities(content=res.content, stock_name=stock.name)
                            saved_prediction = model_service.save_prediction(news_id=res.news_id, prediction=prediction)
                            print(saved_prediction)
                        last_min = current_min
                        session.commit()
                        print("commit done")

                except Exception as e:
                    print(e)
            time.sleep(1)

    # def train_controll(self):
    #     print("train start")
    #     while True:
    #         session = get_session()
    #         current_time = datetime.now()
    #         if current_time.weekday() == FRIDAY:
    #             while True:
    #                 if current_time.time() == datetime.time(hour=17):
    #                      ModelService(session=session).request_training(stock_code=stock_code)
    #                 time.sleep(HOUR)
    #         time.sleep(SIX_HOUR)


if __name__ == "__main__":

    with ThreadPoolExecutor(max_workers=4) as executor:
        print("start")
        timer = ScheduleController()
        executor.submit(timer.min_data_collector, STOCK_List[0])
        executor.submit(timer.min_data_collector, STOCK_List[1])

        # executor.submit(timer.train_controll)
