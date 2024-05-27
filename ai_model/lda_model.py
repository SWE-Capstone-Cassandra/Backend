from gensim.models import LdaModel
import pickle
import pandas as pd
from typing import List
from ai_model.text_preprocessor import TextPreprocessor
from ai_model.korean_lemmatizer_master.soylemma import Lemmatizer

"""
LDA 모델
- 매 주기 별로 저장되어 있는 데이터로 학습 후, model_weights 폴더에 LDA model 들의 가중치 저장.
- 저장 전 폴더는 비우는 것을 원칙(로그 관련은 차후에)
- topic 별로 폴더링이 되며 해당 폴더의 구조는 다음과 같음
    - topic_1
        ㄴ dictionary_1.pkl               : 해당 토픽 내의 말뭉치 사전
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

    def train_lda_model(self, dataset: List[str]):
        """
        주기적으로 LDA모델들을 학습시키기 위한 함수
        뉴스 데이터 세트를 통해 하이퍼파라미터 튜닝부터 폴더링, topic 별 lda 모델 생성, 가중치 저장까지의 프로세스 자동화

        Args:
            dataset (list of str): 뉴스 데이터 세트(원문).
        """

        self.df["documents"] = TextPreprocessor(texts=list(dataset)).preprocess()

    # TODO 최상위 폴더 생성 및 하이퍼파라미터 튜닝 작업

    def get_topic_distribution(self):
        """
        전처리된 텍스트의 토픽 분포를 반환.

        Args:
            text (list of str): 전처리된 텍스트 (형태소 분석된 단어 리스트).

        Returns:
            list of (int, float): 각 토픽의 ID와 해당 토픽의 확률.
        """
        # 전처리된 텍스트를 Bag-of-Words 형식으로 변환
        bow = self.dictionary.doc2bow(self.text)
        # 토픽 분포 계산
        topic_distribution = self.model.get_document_topics(bow, minimum_probability=0)
        return topic_distribution
