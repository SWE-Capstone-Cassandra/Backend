import datetime

import win32com.client


class StockRepository:
    def __init__(self) -> None:
        self.inst_stock_chart = win32com.client.Dispatch("CpSysDib.StockChart")

    def get_stock_data_now(self, item_code):
        now = datetime.datetime.now().strptime("")
        self.inst_stock_chart.SetInputValue(0, item_code)  # 종목코드
        self.inst_stock_chart.SetInputValue(1, ord("2"))  # 개수로 받기

        # inst_stock_chart.SetInputValue(2, "20240429")  # To 날짜
        # inst_stock_chart.SetInputValue(3, "20220420")  # From 날짜
        self.inst_stock_chart.SetInputValue(2, now)
        self.inst_stock_chart.SetInputValue(4, 500)  # 최근 500일치
        self.inst_stock_chart.SetInputValue(5, [0, 2])  # 요청항목 - 날짜,시가,고가,저가,종가,거래량
        self.inst_stock_chart.SetInputValue(6, ord("s"))  # '차트 주기 - 일/주/월
        self.inst_stock_chart.SetInputValue(9, ord("1"))  # 수정주가 사용
        self.inst_stock_chart.BlockRequest()

    def get_stock_data_by_date(self, item_code: str, news_datetime: str):
        self.inst_stock_chart.SetInputValue(0, item_code)  # 종목코드
        self.inst_stock_chart.SetInputValue(1, ord("2"))  # 개수로 받기
        self.inst_stock_chart.SetInputValue(2, news_datetime)
        self.inst_stock_chart.SetInputValue(4, 1)  # 최근 500일치
        self.inst_stock_chart.SetInputValue(5, [0, 2])  # 요청항목 - 날짜,시가,고가,저가,종가,거래량
        self.inst_stock_chart.SetInputValue(6, ord("M"))  # '차트 주기 - 일/주/월
        self.inst_stock_chart.SetInputValue(9, ord("1"))  # 수정주가 사용
        self.inst_stock_chart.BlockRequest()
        num_data = self.inst_stock_chart.GetHeaderValue(3)
        num_field = self.inst_stock_chart.GetHeaderValue(1)

        # 데이터 수집
        for i in range(num_data):
            row_data = []
            for j in range(num_field):
                row_data.append(self.inst_stock_chart.GetDataValue(j, i))

        print(row_data)
