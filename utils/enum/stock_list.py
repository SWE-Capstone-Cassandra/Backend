from enum import Enum


class TupleEnum(str, Enum):
    def __new__(cls, *values):
        obj = str.__new__(cls, values[1])
        obj._value_ = values
        return obj


class StockList(TupleEnum):

    SAMSUNG_ELECTRONIC = ("삼성전자", "005930")
    CJ_JAIL_JAEDANG = ("제일제당", "097950")

    @property
    def name(self):
        return self.value[0]

    @property
    def code(self):
        return self._value_[1]

    @classmethod  # 클래스 메서드로 정의
    def get_list(cls):
        return list(cls.__members__.values())
