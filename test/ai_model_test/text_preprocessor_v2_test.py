import pandas as pd
from collections import Counter

import unittest
from ai_model.preprocessor.text_preprocessor_v2 import TextPreprocessorV2

news_texts = pd.read_csv("data/samsung_6month_dataset.csv")

class TestTextPreprocessorV2(unittest.TestCase):

    # def test(self):
    #     TextPreprocessorV2(list(news_texts['content'].sample(frac=0.1))).preprocess() 

    def test(self):
        result = TextPreprocessorV2(list(news_texts['content'].sample(frac=0.01))).preprocess()
        # print(result)
        counter = Counter()
        
        for sublist in result:
            counter.update(sublist)
        
        print(counter)


if __name__ == "__main__":
    unittest.main()
