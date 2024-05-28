import unittest
from ai_model.lda_model import LDAModel

dummy_dataset = [
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 정부의 민관합동 청년 일자리]",
    "[워싱턴=AP/뉴시스]12일(현지시간) 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]",
]


class TestLdaModel(unittest.TestCase):

    def test_all_process(self):
        ldaModel = LDAModel()
        ldaModel.train_lda_model(dataset=dummy_dataset)


if __name__ == "__main__":
    unittest.main()
