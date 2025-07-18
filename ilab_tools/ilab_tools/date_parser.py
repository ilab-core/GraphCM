import datetime as dt

class DateParser:
    def __init__(self, formats=None):
        self.formats = formats or ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']

    #  Verilen date'in, belirlenen tarih formatlarından birine uygun olup olmadığını kontrol ederek bir datetime objesine dönüştürür. 
    def parse_date(self, date):
        for format in self.formats:
            try:
                return dt.datetime.strptime(date, format)
            except ValueError:
                pass
        raise ValueError(f"Date format is not valid: {date}")

    # datetime objesinden veya string den milisaniye yi cikartir
    @staticmethod
    def remove_ms(date):
        if isinstance(date, dt.datetime):
            return date.replace(microsecond=0)
        elif isinstance(date, str):
            try:
                parsed_date = dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
                return parsed_date.replace(microsecond=0)
            except ValueError:
                return dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        else:
            raise TypeError("Input should be datetime obje or string")
