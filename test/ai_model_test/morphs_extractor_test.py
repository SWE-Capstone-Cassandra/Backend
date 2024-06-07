import unittest
from ai_model.preprocessor.text_preprocessor import TextPreprocessor

news_texts = [
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 정부의 민관합동 청년 일자리]" * 60,
    "[워싱턴=AP/뉴시스]12일(현지시간) 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]" * 60,
    "[ 문재인 대통령이 지난해 12월 27일 청와대에서 미국 워싱턴 백악관 루즈벨트룸에서 열린 반]" * 60,
] * 10000


class TestMorphsExtractor(unittest.TestCase):

    # def test_basical(self):
    #     self.assertEqual(TextPreprocessor("[안녕]").preprocess(), [("안녕", "NNP")])

    def test(self):
        TextPreprocessor(news_texts).preprocess()


if __name__ == "__main__":
    unittest.main()
