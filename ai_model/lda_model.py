import os
import pickle
import shutil
from time import time
from typing import List

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from tqdm import tqdm

from ai_model.constants import BaseConfig, LDAModelConfig, model_weights_path
from ai_model.text_preprocessor import TextPreprocessor

"""
LDA 모델
- 매 주기 별로 저장되어 있는 데이터로 학습 후, model_weights 폴더에 LDA model 들의 가중치 저장.
- 저장 전 폴더는 비우는 것을 원칙(로그 관련은 차후에)
- topic 별로 폴더링이 되며 해당 폴더의 구조는 다음과 같음
    - topic_1
        ㄴ dictionary_1.pkl               : 해당 토픽 내의 말뭉치 사전
        ㄴ corpus_1.pkl               : 해당 토픽 내의 코퍼스
        ㄴ lda_weights_1                  : 해당 토픽의 lda 모델의 가중치
        ㄴ regression_weights_1m.pkl      : 1분 후 회귀 모델
        ㄴ regression_weights_5m.pkl      : 5분 후 회귀 모델
        ㄴ regression_weights_15m.pkl     : 15분 후 회귀 모델
        ㄴ regression_weights_1h.pkl      : 1시간 회귀 모델
        ㄴ regression_weights_1d.pkl      : 1일 후 회귀 모델
   
#### 1. 뉴스 데이터 세트 학습     
1) 뉴스 데이터세트를 인자로 받아 자동으로 LDA 추출을 위한 하이퍼파라미터 튜닝
2) topic 개수만큼 폴더링
3) dictionary 저장
4) 각 topic 별로 LDA 하이퍼파라미터 튜닝 실시
5) 가중치 저장
"""


class LDAModel:

    def __init__(self):
        self.df = pd.DataFrame()
        self.grouped_dfs = list()
        print(__name__, "생성")

    # For Training
    ###########################################################################################
    def train_lda_model(self, dataset: pd.DataFrame):
        """
        주기적으로 LDA모델들을 학습시키기 위한 함수
        뉴스 데이터 세트를 통해 하이퍼파라미터 튜닝부터 폴더링, topic 별 lda 모델 생성, 가중치 저장까지의 프로세스 자동화

        Args:
            dataset (list of str): 뉴스 데이터 세트(원문).
        """

        self.df["date_time"] = dataset["date_time"]
        self.df["documents"] = TextPreprocessor(texts=list(dataset["content"])).preprocess()

        # print("토픽 추출 시작")
        # 토픽 추출
        num_topics = self._get_num_of_topics()
        # print("토픽 추출 완료")

        # print("재폴더링 시작")
        # # 재 폴더링(초기화)
        self._remake_folder(num_topics=num_topics, remake=True)
        # print("재폴더링 완료")

        # print("1차 LDA 모델 추출 및 저장 시작")
        # # 1차 LDA 모델 추출 및 저장
        self._create_main_lda_model_and_save(num_topics=num_topics)
        # print("1차 LDA 모델 추출 및 저장 완료")

        # print("1차 LDA 토픽 분포 추출 및 그룹화 시작")
        # # 1차 LDA 토픽 분포 추출 및 그룹화
        self._get_first_document_topics_and_grouping()
        # print("1차 LDA 토픽 분포 추출 및 그룹화 완료")

        # print("2차 LDA 모델 추출 및 저장 시작")
        # # 2차 LDA 모델 추출 및 저장
        self._create_lda_model_by_topic_and_save(num_topics=num_topics)
        # print("2차 LDA 모델 추출 및 저장 완료")

        return num_topics

    def _get_num_of_topics(self):
        """
        1차 토픽을 추출하는 함수
        """
        try:
            dictionary = corpora.Dictionary(self.df["documents"])
            corpus = [dictionary.doc2bow(text) for text in self.df["documents"]]

            best_coherence = -np.inf
            # best_perplexity = np.inf
            best_model = None

            for num_topics in tqdm(LDAModelConfig.NUM_OF_CATEGORY, desc=f"## LDA Category HyperParameter Tuning ##"):
                # start_time = time()
                model = LdaMulticore(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    random_state=BaseConfig.RANDOM_STATE,
                    passes=LDAModelConfig.PASSES,
                    iterations=LDAModelConfig.ITERATIONS,
                    workers=LDAModelConfig.WORKERS,
                )
                coherence_model = CoherenceModel(model=model, texts=list(self.df["documents"]), dictionary=dictionary, coherence="c_v")
                coherence_score = coherence_model.get_coherence()

                # perplexity_score = model.log_perplexity(corpus)

                # end_time = time()
                # elapsed_time = end_time - start_time

                # if coherence_score > best_coherence and perplexity_score < best_perplexity:
                #     best_coherence = coherence_score
                #     best_perplexity = perplexity_score
                #     best_num_topics = num_topics
                # print(
                #     f"num_topics: {num_topics}, coherence_score: {best_coherence}, perplexity: {best_perplexity} time: {elapsed_time:.2f} seconds"
                # )
                if coherence_score > best_coherence:
                    best_coherence = coherence_score
                    best_num_topics = num_topics
                    best_model = model

            # # 최적의 하이퍼파라미터 출력
            # print("Best Coherence Score - category:", best_coherence)
            # print("Best Params - category:", best_num_topics)
            for idx, topic in best_model.print_topics(-1):
                print(f"Topic: {idx} \nWords: {topic}\n")

            return best_num_topics

        except Exception as e:
            print("Error of _get_num_of_topics method:", e)

    def _get_num_of_topics_by_group(self, topic_idx, texts, dictionary, corpus):
        """
        각 토픽 그룹별 LDA 수행하는 함수
        각 토픽에 해당하는 폴더로 접근하여 저장한다
        Args:
            topic_idx: 토픽 번호
            texts: LDA에 수행될 그룹 documents
            dictionary: 그룹 dictionary
            corpus: 그룹 corpus

        Returns:
            int: 해당 그룹의 최적의 topic 수
        """
        try:
            # print(f"현재 {topic_idx}번째 그룹의 텍스트 세트:\n", texts)
            best_coherence = 0
            best_params = {}

            for num_topics in tqdm(LDAModelConfig.NUM_OF_TOPICS_BY_GROUP, desc=f"## LDA Model HyperParameter Tuning ##"):
                start_time = time()
                model = LdaMulticore(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    random_state=BaseConfig.RANDOM_STATE,
                    passes=LDAModelConfig.PASSES,
                    workers=None,
                )
                coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence="c_v")
                coherence_score = coherence_model.get_coherence()
                end_time = time()
                elapsed_time = end_time - start_time

                if coherence_score > best_coherence:
                    print(f"num_topics: {num_topics}, coherence_score: {coherence_score}, time: {elapsed_time:.2f} seconds")
                    best_coherence = coherence_score
                    best_params = {"num_topics": num_topics}

            # # 최적의 하이퍼파라미터 출력
            # print("Best Coherence Score:", best_coherence)
            # print("Best Params:", best_params)

            return best_params["num_topics"]

        except Exception as e:
            print("Error of _get_num_of_topics_by_group method:", e)

    def _remake_folder(self, num_topics, remake=True):
        try:
            if os.path.exists(model_weights_path):
                shutil.rmtree(model_weights_path)

                # 폴더 다시 만들기
                os.makedirs(model_weights_path)

            if remake:
                for i in range(num_topics):
                    new_folder_name = f"topic_{i+1}"
                    new_folder_path = os.path.join(model_weights_path, new_folder_name)
                    os.makedirs(new_folder_path)

        except Exception as e:
            print("Error of _remake_folder method:", e)

    def _create_main_lda_model_and_save(self, num_topics):

        try:
            dictionary = corpora.Dictionary(self.df["documents"])
            corpus = [dictionary.doc2bow(text) for text in self.df["documents"]]

            params = {
                "num_topics": num_topics,
                "corpus": corpus,
                "id2word": dictionary,
                "passes": LDAModelConfig.PASSES,
                "random_state": BaseConfig.RANDOM_STATE,
            }

            lda_model = LdaMulticore(**params)

            model_name = "category_lda_model.model"
            model_path = os.path.join(model_weights_path, model_name)

            lda_model.save(model_path)

            dictionary_name = "category_dictionary.pkl"
            dictionary_path = os.path.join(model_weights_path, dictionary_name)
            with open(dictionary_path, "wb") as f:
                pickle.dump(dictionary, f)

            corpus_name = "category_corpus.pkl"
            corpus_path = os.path.join(model_weights_path, corpus_name)
            with open(corpus_path, "wb") as f:
                pickle.dump(corpus, f)

        except Exception as e:
            print("Error of _create_main_lda_model_and_save method:", e)

    def _create_lda_model_by_topic_and_save(self, num_topics):

        try:
            self.grouped_dfs = [self.df[self.df["category"] == i] for i in range(1, num_topics + 1)]

            for group_idx, group_df in tqdm(enumerate(self.grouped_dfs, start=1), desc=f"## Create LDA Model by Topic ##"):
                group_texts = group_df["documents"].tolist()

                # 그룹별 단어 사전 생성
                group_dictionary = corpora.Dictionary(group_texts)
                group_corpus = [group_dictionary.doc2bow(text) for text in group_texts]

                group_idx_num_topics = self._get_num_of_topics_by_group(
                    topic_idx=group_idx, texts=group_texts, dictionary=group_dictionary, corpus=group_corpus
                )

                params = {
                    "num_topics": group_idx_num_topics,
                    "corpus": group_corpus,
                    "id2word": group_dictionary,
                    "passes": LDAModelConfig.PASSES,
                    "random_state": BaseConfig.RANDOM_STATE,
                    "workers": LDAModelConfig.WORKERS,
                    "iterations": LDAModelConfig.ITERATIONS,
                }

                model = LdaMulticore(**params)

                model_name = f"topic_{group_idx}/lda_model_{group_idx}.model"
                model_path = os.path.join(model_weights_path, model_name)

                model.save(model_path)

                dicitonary_name = f"topic_{group_idx}/dictionary_{group_idx}.pkl"
                dictionary_path = os.path.join(model_weights_path, dicitonary_name)

                with open(dictionary_path, "wb") as f:
                    pickle.dump(group_dictionary, f)

                corpus_name = f"topic_{group_idx}/corpus_{group_idx}.pkl"
                corpus_path = os.path.join(model_weights_path, corpus_name)
                with open(corpus_path, "wb") as f:
                    pickle.dump(group_corpus, f)

        except Exception as e:
            print("Error of _create_lda_model_by_topic_and_save method:", e)

    def _get_first_document_topics_and_grouping(self):

        try:
            model_name = "category_lda_model.model"
            model_path = os.path.join(model_weights_path, model_name)
            model = LdaMulticore.load(model_path)

            corpus_name = "category_corpus.pkl"
            corpus_path = os.path.join(model_weights_path, corpus_name)
            with open(corpus_path, "rb") as f:
                corpus = pickle.load(f)

            doc_topics = [model.get_document_topics(bow, minimum_probability=0) for bow in corpus]

            num_topics = model.num_topics
            for i in range(num_topics):
                self.df[f"topic_{i+1}"] = [dict(doc).get(i, 0) for doc in doc_topics]

            self.df["category"] = self.df[[f"topic_{i+1}" for i in range(num_topics)]].idxmax(axis=1)
            self.df["category"] = self.df["category"].apply(lambda x: int(x.split("_")[1]))

        except Exception as e:
            print("Error of _get_first_document_topics_and_grouping method:", e)

    ###########################################################################################

    # For Predicting
    ###########################################################################################
    def get_group_id_and_topic_distribution(self, text):
        """
        해당 텍스트의 토픽 그룹과 토픽 확률을 반환
        컨트롤러의 API 역할

        Args:
            text: 뉴스 원문.

        Returns:
            group_id: 토픽 그룹
            distribution: 해당 그룹의 토픽 분포 확률
        """

        try:
            # 텍스트 전처리
            preprocessed_text = TextPreprocessor(texts=text).preprocess()

            # 1차 가장 높은 확률의 토픽 추출
            group_id = self._get_group_id(text=preprocessed_text)

            # 해당 그룹에서 LDA 수행, 분포 추출
            topic_distribution = self._get_topic_distribution_from_group(group_id=group_id, text=preprocessed_text)

            return group_id, topic_distribution

        except Exception as e:
            print("Error of get_group_id_and_topic_distribution method:", e)

    def _get_group_id(self, text):
        try:
            print("group id 추출")
            model_name = "category_lda_model.model"
            model_path = os.path.join(model_weights_path, model_name)
            model = LdaMulticore.load(model_path)

            print(f"사용된 모델 경로: {model_path}")

            dictionary_name = "category_dictionary.pkl"
            dictionary_path = os.path.join(model_weights_path, dictionary_name)
            with open(dictionary_path, "rb") as f:
                dictionary = pickle.load(f)

            print(f"사용된 사전 경로: {dictionary_path}")

            print(text)

            bow = dictionary.doc2bow(text)

            topic_distribution = model.get_document_topics(bow, minimum_probability=0)
            group_id = max(topic_distribution, key=lambda item: item[1])[0]
            print(f"추출된 그룹 번호: {group_id}")
            return group_id

        except Exception as e:
            print("Error of _get_group_id method:", e)

    def _get_topic_distribution_from_group(self, group_id, text):

        try:
            print("2차 토픽 추출")
            model_name = f"topic_{group_id}/lda_model_{group_id}.model"
            model_path = os.path.join(model_weights_path, model_name)
            model = LdaMulticore.load(model_path)

            print(f"사용된 모델 경로: {model_path}")

            dictionary_name = f"topic_{group_id}/dictionary_{group_id}.pkl"
            dictionary_path = os.path.join(model_weights_path, dictionary_name)
            with open(dictionary_path, "rb") as f:
                dictionary = pickle.load(f)

            print(f"사용된 사전 경로: {dictionary_path}")

            bow = dictionary.doc2bow(text)

            topic_distribution = model.get_document_topics(bow, minimum_probability=0)
            print(f"추출된 토픽 분포: {topic_distribution}")
            return topic_distribution

        except Exception as e:
            print("Error of _get_topic_distribution_from_group method:", e)

    ###########################################################################################

    # For Utils
    ###########################################################################################
    def get_group_df(self) -> List[pd.DataFrame]:
        return self.grouped_dfs
