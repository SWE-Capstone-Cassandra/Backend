import re
from typing import List
from pecab import PeCab
from ai_model.korean_lemmatizer_master.soylemma.lemmatizer import Lemmatizer


class TextPreprocessor:
    def __init__(self, texts: List[str] or str):
        self.texts = texts

    def preprocess(self):
        self._analyze_morphs()
        self._remove_stopwords()
        self._remove_special_characters()
        self._lemmatize_filtered_morphs()
        self._join_texts()
        return self.texts

    def _analyze_morphs(self):
        pecab = PeCab()

        if type(self.texts) is str:
            self.texts = pecab.pos(self.texts)
        else:
            self.texts = [pecab.pos(text) for text in self.texts]

    def _remove_stopwords(self):
        stopwords_path = "/home/tako4/capstone/backend/Model/Backend/stopwords-ko.txt"
        with open(stopwords_path, "r", encoding="utf-8") as file:
            stopwords = [line.strip() for line in file.readlines()]

        used_pos = ["NNG", "NNP", "NR", "NP", "VV", "VA", "VX"]

        def filter_morphs(morphs):
            return [morph for morph in morphs if any(pos in morph[1] for pos in used_pos) and morph[0] not in stopwords]

        if type(self.texts[0]) is tuple:
            self.texts = filter_morphs(self.texts)
        else:
            self.texts = [filter_morphs(text) for text in self.texts]

    def _remove_special_characters(self):
        def clean_morphs(morphs):
            cleaned_morphs = []
            for word, pos in morphs:
                cleaned_word = re.sub("[^가-힣a-zA-Z0-9]", "", word)
                if cleaned_word:
                    cleaned_morphs.append((cleaned_word, pos))
            return cleaned_morphs

        if type(self.texts[0]) is tuple:
            self.texts = clean_morphs(self.texts)
        else:
            self.texts = [clean_morphs(text) for text in self.texts]

    def _lemmatize_filtered_morphs(self):
        lemmatizer = Lemmatizer()

        def lemmatize_filtered_morph(morphs):
            result = []
            for word, pos in morphs:
                if pos in ["VV", "VA", "VX"]:
                    lemmatized_word = lemmatizer.lemmatize(word)
                    if not lemmatized_word:
                        result.append(word)
                    else:
                        result.append(lemmatized_word[0][0])
                else:
                    result.append(word)
            return result

        if type(self.texts[0]) is tuple:
            self.texts = lemmatize_filtered_morph(self.texts)
        else:
            self.texts = [lemmatize_filtered_morph(text) for text in self.texts]

    def _join_texts(self):
        def join_and_clean(text):
            joined_text = " ".join(word for word in text)
            return joined_text

        if type(self.texts[0]) is str:
            self.texts = [join_and_clean(self.texts)]
        else:
            self.texts = [[join_and_clean(text)] for text in self.texts]
