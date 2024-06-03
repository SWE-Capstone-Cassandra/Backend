import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List

from ai_model.data_controller import DataController
from config import get_session
from repository.news_repository import NewsRepository
from repository.stock_repository import StockRepository
from service.model_service import ModelService
from service.news_service import NewsService
from model.news import News

FRIDAY = 0
HOUR = 60 * 60
SIX_HOUR = 60 * 60 * 6


class TimerMain:

    def min_data_collector(self):
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
                        news_min_list = news_service.get_news_list_min(item_name="삼성전자", time_now=time_now)
                        print(news_min_list)
                        for news in news_min_list:
                            data = news_service.get_news_data(date_time=int(time_now / 100), url=news)
                            res = news_service.save_news_data(news=data)
                            prediction: List = model_service.request_stock_volatilities(content=res.content)
                            saved_prediction = model_service.save_prediction(news_id=res.news_id, prediction=prediction)
                            print(saved_prediction)
                        last_min = current_min
                        session.commit()
                        print("commit done")

                except Exception as e:
                    print(e)
            time.sleep(1)

    def train_controll(self):
        print("train start")
        while True:
            session = get_session()
            current_time = datetime.now()
            if current_time.weekday() == FRIDAY:
                while True:
                    if current_time.time() == datetime.time(hour=17):
                        news_dataset = NewsRepository(session=session).get_news_dataset()
                        stock_dataset = StockRepository(session=session).get_stock_dataset()
                        try:
                            DataController().train_news_dataset(news_dataset=news_dataset, stock_dataset=stock_dataset)
                            print("done")
                            break
                        except Exception as ex:
                            print("무슨 에러지?", ex)
                    time.sleep(HOUR)
            time.sleep(SIX_HOUR)


if __name__ == "__main__":

    with ThreadPoolExecutor(max_workers=2) as executor:
        print("start")
        timer = TimerMain()
        executor.submit(timer.min_data_collector)
        executor.submit(timer.train_controll)
