import unittest
from ai_model.morphs_extractor import MorphsExtractor


class TestMorphsExtractor(unittest.TestCase):

    def test_basical(self):
        self.assertEqual(MorphsExtractor("[안녕]").extract(), [("안녕", "NNP")])

    def test_stopwords(self):
        self.assertEqual(MorphsExtractor("[1231241231]").extract(), [])


if __name__ == "__main__":
    unittest.main()
