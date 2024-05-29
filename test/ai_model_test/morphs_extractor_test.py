import unittest
from ai_model.text_preprocessor import TextPreprocessor


class TestMorphsExtractor(unittest.TestCase):

    def test_basical(self):
        self.assertEqual(TextPreprocessor("[안녕]").preprocess(), [("안녕", "NNP")])

    def test_stopwords(self):
        self.assertEqual(TextPreprocessor("[1231241231]").preprocess(), [])


if __name__ == "__main__":
    unittest.main()
