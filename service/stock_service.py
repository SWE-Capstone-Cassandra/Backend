from repository.stock_repository import StockRepository
from service.base_service import BaseService
from utils.enum.stock_code import StockCode


class StockService(BaseService):

    def get_stock_data_now(self, stock_code: StockCode):
        return StockRepository(session=self.session).get_stock_data_now(stock_code=stock_code)

    def save_stock_data(self, stock_code: StockCode, price: int):
        price = StockRepository(session=self.session).save_stock_data(stock_code=stock_code, price=price)

        return True
