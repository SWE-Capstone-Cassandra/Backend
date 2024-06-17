from service.model_service import ModelService

if __name__ == "__main__":
    from ai_model.data_controller import DataController
    from config import get_session
    from model.news import News
    from repository.news_repository import NewsRepository
    from repository.stock_repository import StockRepository

    session = get_session()
    stock_code = ""
    ModelService(session=session).request_training(stock_code=stock_code)
