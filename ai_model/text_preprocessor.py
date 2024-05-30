import re
from tqdm import tqdm
from typing import List
from konlpy.tag import Mecab
from ai_model.korean_lemmatizer_master.soylemma.lemmatizer import Lemmatizer


class TextPreprocessor:
    def __init__(self, texts: List[str] or str):
        self.texts = texts
        print(__name__, "생성")

    def preprocess(self):
        self._analyze_morphs()
        self._remove_stopwords()
        self._remove_special_characters()
        self._lemmatize_filtered_morphs()
        self._join_texts()
        print("텍스트 전처리 완료")
        return self.texts

    def _analyze_morphs(self):
        mecab = Mecab()

        if type(self.texts) is str:
            self.texts = mecab.pos(self.texts)
        else:
            self.texts = [mecab.pos(text) for text in tqdm(self.texts, desc="Analyze Morphs")]

        print("Fin analyze morphs")

    def _remove_stopwords(self):
        stopwords_path = "/home/tako4/capstone/backend/Model/stopwords-ko.txt"
        with open(stopwords_path, "r", encoding="utf-8") as file:
            stopwords = set(line.strip() for line in file.readlines())

        used_pos = ["NNG", "NNP", "NR", "NP", "VV", "VA", "VX"]

        def filter_morphs(morphs):
            return list(filter(lambda morph: any(pos in morph[1] for pos in used_pos) and morph[0] not in stopwords, morphs))

        if type(self.texts[0]) is tuple:
            self.texts = filter_morphs(self.texts)
        else:
            self.texts = [filter_morphs(text) for text in tqdm(self.texts, desc="Remove Stopwords")]

        print("Fin remove stopwords")

    def _remove_special_characters(self):
        special_char_pattern = re.compile("[^가-힣a-zA-Z0-9]")  # 정규 표현식 컴파일

        def clean_morphs(morphs):
            return [(special_char_pattern.sub("", word), pos) for word, pos in morphs if special_char_pattern.sub("", word)]

        if type(self.texts[0]) is tuple:
            self.texts = clean_morphs(self.texts)
        else:
            self.texts = [clean_morphs(text) for text in tqdm(self.texts, desc="Remove Special Characters")]

        print("Fin remove special characters")

    def _lemmatize_filtered_morphs(self):

        lemmatizer = Lemmatizer()

        lemmatize_pos = {"VV", "VA", "VX"}  # 품사 체크를 집합으로 변환

        def lemmatize_filtered_morph(morphs):
            result = [
                lemmatizer.lemmatize(word)[0][0] if pos in lemmatize_pos and lemmatizer.lemmatize(word) else word
                for word, pos in morphs
            ]
            return result

        if type(self.texts[0]) is tuple:
            self.texts = lemmatize_filtered_morph(self.texts)
        else:
            self.texts = [lemmatize_filtered_morph(text) for text in tqdm(self.texts, desc="Lemmatize Filtered Morphs")]

        print("Fin lemmatize filtered morphs")

    def _join_texts(self):
        def join_and_clean(text):
            return " ".join(text)

        if type(self.texts[0]) is str:
            self.texts = [join_and_clean(self.texts)]
        else:
            self.texts = [[join_and_clean(text)] for text in tqdm(self.texts, desc="Join Texts")]

        print("Fin join texts")
