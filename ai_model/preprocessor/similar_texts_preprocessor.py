import pandas as pd


class SimilarTextsPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        print(__name__, "생성")

    def preprocess(self):
        """
        뉴스 원문의 고윳갑 중 가장 오래된 제외 제거 후, 데이터 세트를 날짜 오름차순으로 정렬하여 반환

        Return:
            df_unique - pd.DataFrame
        """
        df_sorted = self.df.sort_values(by=["content", "date_time"])
        df_unique = df_sorted.drop_duplicates(subset="content", keep="first")
        df_final = df_unique.sort_values(by="date_time")
        return df_final
