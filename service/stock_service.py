from repository.stock_repository import StockRepository


class StockService:

    def get_stock_data_now(self, item_code: str):
        data = StockRepository().get_stock_data_now(item_code=item_code)
        return data

    def get_today_stock_data(self, item_code: str):
        pass
