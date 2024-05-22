import re
from pecab import PeCab


class TextPreprocessor:
    def __init__(self, text: str):
        self.text = text

    def preprocess(self):
        self._analyze_morphs()
        self._remove_stopwords()
        self._remove_special_characters()
        return self.text

    def _analyze_morphs(self):
        pecab = PeCab()
        self.text = pecab.pos(self.text)

    def _remove_stopwords(self):
        stopwords_path = "/home/tako/Documents/yonghan/Model/stopwords-ko.txt"
        with open(stopwords_path, "r", encoding="utf-*") as file:
            stopwords = [line.strip() for line in file.readlines()]

        used_pos = ["NNG", "NNP", "NR", "NP", "VV", "VA", "VX"]

        if isinstance(self.text, str):
            self.text = eval(self.text)
        # 사용할 품사와 불용어를 기준으로 필터링
        self.text = [morph for morph in self.text if any(pos in morph[1] for pos in used_pos) and morph[0] not in stopwords]

    def _remove_special_characters(self):
        cleaned_morphs = []
        if isinstance(self.text, str):
            self.text = eval(self.text)
        for word, pos in self.text:
            cleaned_word = re.sub("[^가-힣a-zA-Z0-9]", "", word)
            if cleaned_word:
                cleaned_morphs.append((cleaned_word, pos))
        return cleaned_morphs
