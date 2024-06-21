import datetime
from typing import List

import pandas as pd

from ai_model.data_controller import DataController
from ai_model.utils import calculate_mape
from model.news_prediction import NewsPrediction
from repository.news_repository import NewsRepository
from repository.prediction_repository import PredictionRepository
from repository.stock_repository import StockRepository
from schema.news_prediction import NewsPredictionSchema
from service.base_service import BaseService
from service.news_service import NewsService
from service.stock_service import StockService
from utils.enum.stock_code import StockCode
from utils.enum.stock_list import StockList


class ModelService(BaseService):
    """

    1. 자동 학습 요청 서비스
    2. 주가 변화량 예측 요청 서비스
    """

    def request_training(self, stock_code: StockCode):
        news_dataset = NewsRepository(session=self.session).get_news_dataset_by_stock_code(stock_code=stock_code)
        stock_dataset = StockRepository(session=self.session).get_stock_dataset_by_stock_code(stock_code=stock_code)
        try:
            DataController().train_news_dataset(stock_name=stock_code, news_dataset=news_dataset, stock_dataset=stock_dataset)
            print("done")
        except Exception as ex:
            print("무슨 에러지?", ex)

    def request_stock_volatilities(self, content: str, stock_name: str) -> List[float]:
        return DataController().predict_stock_volatilities(text=content, stock_name=stock_name)

    def get_prediction_by_news_id(self, news_id: int) -> NewsPredictionSchema:
        res = PredictionRepository(session=self.session).get_news_prediction_by_news_id(news_id=news_id)

        price = StockRepository(session=self.session).get_stock_data_by_date(stock_code="097950", date_time=res.time)
        prediction = NewsPredictionSchema(
            min_1=res.min_1, min_5=res.min_5, min_15=res.min_15, hour_1=res.hour_1, day_1=res.day_1, stock_price=price
        )

        return prediction

    def save_prediction(self, news_id: int, prediction: List[float]):
        news_prediction = NewsPrediction()
        news_prediction.news_id = news_id
        news_prediction.min_1 = float(prediction[0]) if len(prediction) > 0 else 0
        news_prediction.min_5 = float(prediction[1]) if len(prediction) > 1 else 0
        news_prediction.min_15 = float(prediction[2]) if len(prediction) > 2 else 0
        news_prediction.hour_1 = float(prediction[3]) if len(prediction) > 3 else 0
        news_prediction.day_1 = float(prediction[4]) if len(prediction) > 4 else 0

        res = PredictionRepository(session=self.session).save_news_prediction(news_prediction=news_prediction)
        return res

    def fetch_and_store_news_every_minute(self):
        time_now = datetime.datetime.now().time()
        time_now = time_now.strftime("%H%M")
        news_url_list = NewsService().get_news_list_min(item_name="삼성전자", time_now=time_now)
        for url in news_url_list:
            self.save_prediction(url=url, date="", time=time_now)

    def get_prediction_by_list(self, news_list: List):
        return PredictionRepository(session=self.session).get_prediction_by_list(news_list=news_list)

    def request_prediction(self, start_day, end_day, stock_code: StockCode, stock: StockList):
        """
        특정기간기준으로 저장되어있는 뉴스데이터를 불러온 뒤, 그 뉴스들에 대해 예측값을 저장.
        """
        news_list = NewsService(session=self.session).get_news_by_period(start_day=start_day, end_day=end_day, stock_code=stock_code)
        for new in news_list:
            prediction: List = self.request_stock_volatilities(content=new.content, stock_name=stock.name)
            saved_prediction = self.save_prediction(news_id=new.news_id, prediction=prediction)
            print(saved_prediction)

    def get_model_accuracy(self, start_day, end_day, stock_code: StockCode):

        # 예상 시퀸스

        # prediction 데이터를 가져오기
        # 실제 주가 데이터를 가져오기
        news_id_list = NewsService(session=self.session).get_news_id_by_period(
            start_day=start_day, end_day=end_day, stock_code=stock_code
        )
        prediction_record = self.get_prediction_by_list(news_list=news_id_list)
        price_record = StockService(session=self.session).get_stock_data_by_period(
            start_day=start_day, end_day=end_day, stock_code=stock_code
        )
        prediction_df = pd.DataFrame(columns=["m1", "m5", "m15", "h1", "d1", "original"])
        prediction_df.index.name = "datetime"

        for record in price_record:
            date_time = record.date_time.replace(tzinfo=None)

            if date_time not in prediction_df.index:
                prediction_df.loc[date_time] = {"m1": [], "m5": [], "m15": [], "h1": [], "d1": [], "original": []}

            prediction_df.at[date_time, "original"].append(record.price)

        prediction_df = self.adjust_time(prediction_record=prediction_record, prediction_df=prediction_df)

        # 시간 처리하기

        for date_time in prediction_df.index:
            original_values = prediction_df.at[date_time, "original"]

            # 각 예측 값 리스트에 대해 계산 수행
            if original_values:  # original_values가 비어있지 않은 경우에만
                original = original_values[0]  # original은 첫 번째 값으로 가정
                prediction_df.at[date_time, "m1"] = [original - val for val in prediction_df.at[date_time, "m1"]]
                prediction_df.at[date_time, "m5"] = [original - val for val in prediction_df.at[date_time, "m5"]]
                prediction_df.at[date_time, "m15"] = [original - val for val in prediction_df.at[date_time, "m15"]]
                prediction_df.at[date_time, "h1"] = [original - val for val in prediction_df.at[date_time, "h1"]]
                prediction_df.at[date_time, "d1"] = [original - val for val in prediction_df.at[date_time, "d1"]]

        mape_values = {"m1": [], "m5": [], "m15": [], "h1": [], "d1": []}

        for date_time in prediction_df.index:
            original_values = prediction_df.at[date_time, "original"]
            if original_values:
                original = original_values[0]  # original은 첫 번째 값으로 가정

                for key in ["m1", "m5", "m15", "h1", "d1"]:
                    mape = calculate_mape([original] * len(prediction_df.at[date_time, key]), prediction_df.at[date_time, key])
                    if mape is not None:
                        mape_values[key].append(mape)

        # 평균 MAPE 값 계산
        avg_mape = {key: sum(values) / len(values) if values else None for key, values in mape_values.items()}
        file_path = f"/home/tako4/capstone/backend/Backend/data/result/{stock_code}/{start_day}_{end_day}.csv"

        mape_df = pd.DataFrame(list(avg_mape.items()), columns=["Interval", "MAPE"])

        # MAPE 값을 CSV 파일로 저장
        # mape_df.to_csv(file_path, index=False)
        print(mape_df)

    def adjust_time(self, prediction_record, prediction_df):
        market_close_time = datetime.datetime.strptime("15:30", "%H:%M").time()
        next_open_time = datetime.datetime.strptime("09:00", "%H:%M").time()
        try:
            for prediction in prediction_record:
                date_time: datetime.datetime = prediction.time.replace(tzinfo=None)

                new_date_time = date_time

                if date_time.time() < next_open_time:
                    new_date_time.replace(hour=9, minute=1, second=0, microsecond=0)
                    new_date_time = self.is_market_Day(new_date_time)
                    prediction_df = self.add(date_time=new_date_time, prediction=prediction, prediction_df=prediction_df)

                elif date_time.time() > market_close_time:
                    new_date_time = new_date_time + datetime.timedelta(days=1)
                    new_date_time.replace(hour=9, minute=1, second=0, microsecond=0)
                    new_date_time = self.is_market_Day(new_date_time)

                    prediction_df = self.add(date_time=new_date_time, prediction=prediction, prediction_df=prediction_df)

                elif date_time.time() > datetime.datetime.strptime("14:30", "%H:%M").time():
                    new_date_time = self.is_market_Day(new_date_time)
                    prediction_df = self.add(date_time=new_date_time, prediction=prediction, prediction_df=prediction_df)

                else:
                    # 시간조정
                    if (date_time + datetime.timedelta(minutes=1)).time() > market_close_time:
                        new_date_time = date_time + datetime.timedelta(days=1)
                        new_date_time.replace(hour=9, minute=1, second=0, microsecond=0)
                        new_date_time = self.is_market_Day(new_date_time)
                        prediction_df.at[new_date_time, "m1"].append(prediction.min_1)

                    else:
                        prediction_df.at[date_time + datetime.timedelta(minutes=1), "m1"].append(prediction.min_1)

                    if (date_time + datetime.timedelta(minutes=5)).time() > market_close_time:
                        new_date_time = date_time + datetime.timedelta(days=1)
                        new_min = (date_time + datetime.timedelta(minutes=5)).time() - market_close_time
                        new_date_time.replace(hour=9, minute=new_min, second=0, microsecond=0)
                        new_date_time = self.is_market_Day(new_date_time)
                        prediction_df.at[new_date_time, "m5"].append(prediction.min_5)

                    else:
                        prediction_df.at[date_time + datetime.timedelta(minutes=5), "m5"].append(prediction.min_5)

                    if (date_time + datetime.timedelta(minutes=15)).time() > market_close_time:
                        new_date_time = date_time + datetime.timedelta(days=1)
                        new_min = (date_time + datetime.timedelta(minutes=15)).time() - market_close_time
                        new_date_time.replace(hour=9, minute=new_min, second=0, microsecond=0)
                        new_date_time = self.is_market_Day(new_date_time)
                        prediction_df.at[new_date_time, "m15"].append(prediction.min_15)

                    else:
                        prediction_df.at[date_time + datetime.timedelta(minutes=5), "m15"].append(prediction.min_15)

                    if (date_time + datetime.timedelta(hours=1)).time() > market_close_time:
                        new_date_time = date_time + datetime.timedelta(days=1)
                        new_min = (date_time + datetime.timedelta(hours=1)).time() - market_close_time
                        new_date_time.replace(hour=9, minute=new_min, second=0, microsecond=0)
                        new_date_time = self.is_market_Day(new_date_time)
                        prediction_df.at[new_date_time, "h1"].append(prediction.hour_1)

                    else:
                        prediction_df.at[date_time + datetime.timedelta(hours=1), "h1"].append(prediction.hour_1)

                    new_date_time = date_time + datetime.timedelta(days=1)
                    new_date_time = self.is_market_Day(new_date_time)
                    prediction_df.at[new_date_time, "d1"].append(prediction.day_1)
        except Exception as e:
            print(e)

        return prediction_df

    def is_market_Day(self, date_time: datetime.datetime):
        while date_time.weekday() == 6 or date_time.weekday() == 5:
            date_time = date_time + datetime.timedelta(days=1)
        return date_time

    def add(self, date_time: datetime.datetime, prediction, prediction_df: pd.DataFrame):
        prediction_df.at[date_time + datetime.timedelta(minutes=1), "m1"].append(prediction.min_1)
        prediction_df.at[date_time + datetime.timedelta(minutes=5), "m5"].append(prediction.min_5)
        prediction_df.at[date_time + datetime.timedelta(minutes=15), "m15"].append(prediction.min_15)
        prediction_df.at[date_time + datetime.timedelta(hours=1), "h1"].append(prediction.hour_1)
        date = date_time + datetime.timedelta(days=1)
        date = self.is_market_Day(date)
        prediction_df.at[date, "d1"].append(prediction.day_1)

        return prediction_df
