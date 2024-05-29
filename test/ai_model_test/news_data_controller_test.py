import unittest
from ai_model.regression_model import RegressionModel
import pandas as pd
from ai_model.lda_model import LDAModel

import warnings
from ai_model.news_data_controller import NewsDataController

warnings.filterwarnings("ignore")

news_texts = [
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 정부의 민관합동 청년 일자리]",
    "[워싱턴=AP/뉴시스]12일(현지시간) 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
    "[워싱턴=AP/뉴시스]12일(현지시간) 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
    "[워싱턴=AP/뉴시스]12일(현지시간) 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 정부의 민관합동 청년 일자리]",
    "[워싱턴=AP/뉴시스]12일(현지시간) 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
    "[워싱턴=AP/뉴시스]12일(현지시간) 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
    "[워싱턴=AP/뉴시스]12일(현지시간) 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
] * 8

news_date = pd.to_datetime(
    [
        "2022-05-01 09:57:00",
        "2022-05-03 08:01:00",
        "2022-05-03 10:21:00",
        "2022-05-01 09:57:00",
        "2022-05-03 08:01:00",
        "2022-05-03 10:21:00",
        "2022-05-03 10:21:00",
        "2022-05-01 09:57:00",
        "2022-05-03 08:01:00",
        "2022-05-03 10:21:00",
        "2022-05-01 09:57:00",
        "2022-05-03 08:01:00",
        "2022-05-03 10:21:00",
        "2022-05-03 10:21:00",
    ]
    * 8
)

test_news_datasets = pd.DataFrame({"publish_time": news_date, "content": news_texts})

stock_date = pd.to_datetime(
    [
        "2022-05-02 09:00:00",
        "2022-05-02 09:01:00",
        "2022-05-02 09:05:00",
        "2022-05-02 09:15:00",
        "2022-05-02 10:00:00",
        "2022-05-03 09:00:00",
        "2022-05-03 09:01:00",
        "2022-05-03 09:03:00",
        "2022-05-03 09:15:00",
    ]
)
stock_price = [51231, 74520, 53210, 41230, 31251, 31251, 38251, 51251, 95251]

test_stock_datasets = pd.DataFrame({"date_time": stock_date, "price": stock_price})

test_text = "[ 문재인 대통령이 지난해 12월 27일 청와대에서 정부의 민관합동 청년 일자리]"


class TestNewsDataController(unittest.TestCase):

    # def test_train_train_news_dataset(self):
    #     NewsDataController().train_news_dataset(news_datasets=test_news_datasets, stock_datasets=test_stock_datasets)

    def test_predict_stock_volatilities(self):
        print(NewsDataController().predict_stock_volatilities(text=test_text))


if __name__ == "__main__":
    unittest.main()
