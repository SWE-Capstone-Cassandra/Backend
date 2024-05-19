class NewsRepository:
    def get_news_prediction(self, news_id: int):

        pass

    def save_news_prediction(self, news_code: int, stock_now: int):

        news_prediction = 인공지능().get_prediction(news_code, stock_now)

        self.session.add(news_prediction)

        self.session.commit()
