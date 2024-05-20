from repository.stock_repository import StockRepository


class StockService:

    def get_stock_data_by_date(self, date: int, time: int):
        return StockRepository().get_stock_data_by_date(date=date, time=time)

    def save_stock_data(self, time: int, price: int):
        return StockRepository().save_stock_data(time=time, price=price)
