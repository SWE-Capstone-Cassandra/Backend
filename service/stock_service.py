from repository.stock_repository import StockRepository
from service.base_service import BaseService


class StockService(BaseService):

    def get_stock_data_now(self):
        return StockRepository(session=self.session).get_stock_data_now()

    def save_stock_data(self, price: int):
        price = StockRepository(session=self.session).save_stock_data(price=price)

        return True
