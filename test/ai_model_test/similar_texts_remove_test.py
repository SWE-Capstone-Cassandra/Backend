import pandas as pd

import unittest
from ai_model.preprocessor.similar_texts_preprocessor import SimilarTextsPreprocessor

news_texts = [
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 정부의 민관합동 청년 일자리]" * 60,
    "[워싱턴=AP/뉴시스]12일(현지시간) 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]" * 60,
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]" * 60,
] * 100

news_date = pd.to_datetime(
    [
        "2022-05-01 09:57:00",
        "2022-05-03 08:01:00",
        "2022-05-03 10:21:00",
    ]
    * 100
)

test_news_dataset = pd.DataFrame({"date_time": news_date, "content": news_texts})

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

test_stock_dataset = pd.DataFrame({"date_time": stock_date, "price": stock_price})


class TestSimilarTexts(unittest.TestCase):

    def test(self):
        print(SimilarTextsPreprocessor(test_news_dataset).preprocess())


if __name__ == "__main__":
    unittest.main()
