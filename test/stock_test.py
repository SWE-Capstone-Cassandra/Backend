import datetime

import win32com.client

SAMSUNG_CODE = "A005930"


class StockRepository:
    def __init__(self) -> None:
        self.inst_stock_chart = win32com.client.Dispatch("CpSysDib.StockChart")

    def get_stock_data_by_date(self, item_code: str, news_datetime: str):
        today = datetime.datetime.today().strftime("%Y%m%d")
        today = "20240212"
        self.inst_stock_chart.SetInputValue(0, item_code)  # 종목코드
        self.inst_stock_chart.SetInputValue(1, ord("2"))  # 개수로 받기
        # self.inst_stock_chart.SetInputValue(2, today)
        self.inst_stock_chart.SetInputValue(4, 1)  # 최근 1개
        self.inst_stock_chart.SetInputValue(5, [0, 1, 2])  # 요청항목 - 날짜,시간,시가
        self.inst_stock_chart.SetInputValue(6, ord("m"))  # '차트 주기 - 일/주/월
        self.inst_stock_chart.SetInputValue(7, 1)
        self.inst_stock_chart.SetInputValue(9, ord("1"))  # 수정주가 사용
        self.inst_stock_chart.BlockRequest()
        num_data = self.inst_stock_chart.GetHeaderValue(3)
        num_field = self.inst_stock_chart.GetHeaderValue(1)

        # 데이터 수집
        for i in range(num_data):
            row_data = []
            for j in range(num_field):
                row_data.append(self.inst_stock_chart.GetDataValue(j, i))
        print(today)
        print(row_data)

    def get_stock_data_now(self, item_cod: str):
        pass

    def Subscribe(self, code):
        self.objStockCur = win32com.client.Dispatch("DsCbo1.StockCur")
        handler = win32com.client.WithEvents(self.objStockCur, CpEvent)
        self.objStockCur.SetInputValue(0, code)
        handler.set_params(self.objStockCur)
        self.objStockCur.Subscribe()

    def Unsubscribe(self):
        self.objStockCur.Unsubscribe()


StockRepository().get_stock_data_by_date(item_code=SAMSUNG_CODE, news_datetime="safd")
