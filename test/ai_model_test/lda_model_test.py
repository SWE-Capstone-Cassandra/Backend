import unittest
from ai_model.lda_model import LDAModel


class TestLdaModel(unittest.TestCase):

    def test_preprocess_dataset(self):
        dummy_dataset = ["[안녕하세요 저는 정용한입니다]", "[반갑습니다 여러분]"]
        ldaModel = LDAModel()
        ldaModel.train_lda_model(dataset=dummy_dataset)
        self.assertEqual(list(ldaModel.df["documents"]), [["안녕 정용한"], ["반갑"]])


if __name__ == "__main__":
    unittest.main()
