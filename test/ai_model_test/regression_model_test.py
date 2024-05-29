import unittest
from ai_model.regression_model import RegressionModel
import pandas as pd
from ai_model.lda_model import LDAModel

import warnings

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
] * 4

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
    * 4
)

test_news_dataset = pd.DataFrame({"publish_time": news_date, "content": news_texts})

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

test_text = "[ 문재인 대통령이 지난해 12월 27일 청와대에서 정부의 민관합동 청년 일자리]"


class TestRegressionModel(unittest.TestCase):

    # def test_train_lda_model(self):
    #     lda_model = LDAModel()
    #     num_topics = lda_model.train_lda_model(test_news_dataset)
    #     RegressionModel(stock_dataset=test_stock_dataset, lda_model=lda_model).train_regression_model(num_topics=num_topics)

    # def test_get_topic_distribution(self):
    #     lda_model = LDAModel()
    #     num_topics = lda_model.train_lda_model(test_news_dataset)
    #     print(RegressionModel(lda_model=lda_model)._get_topic_distribution(0))

    # def test_get_best_performance_regression_model_and_save(self):
    #     lda_model = LDAModel()
    #     _ = lda_model.train_lda_model(test_news_dataset)

    #     reg_model = RegressionModel(stock_dataset=test_stock_dataset, lda_model=lda_model)
    #     topic_distributions = reg_model._get_topic_distribution(topic_idx=0)
    #     reg_model._get_best_performance_regression_model_and_save(topic_idx=0, topic_distributions=topic_distributions)

    def test_get_stock_volatilities(self):
        group_id, topic_distributions = LDAModel().get_group_id_and_topic_distribution(text=test_text)
        print(RegressionModel().get_stock_volatilities(group_id=group_id, topic_distributions=topic_distributions))


if __name__ == "__main__":
    unittest.main()
