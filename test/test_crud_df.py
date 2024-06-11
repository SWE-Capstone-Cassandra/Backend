if __name__ == "__main__":
    from ai_model.data_controller import DataController
    from config import get_session
    from model.news import News
    from repository.news_repository import NewsRepository
    from repository.stock_repository import StockRepository

    session = get_session()
    news_dataset = NewsRepository(session=session).get_news_dataset()
    stock_dataset = StockRepository(session=session).get_stock_dataset()
    try:
        DataController().train_news_dataset(news_dataset=news_dataset, stock_dataset=stock_dataset)
        print("done")
    except Exception as ex:
        print("무슨 에러지?", ex)
