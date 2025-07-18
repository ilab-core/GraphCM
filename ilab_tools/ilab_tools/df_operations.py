import pandas as pd
from datetime import datetime, timedelta
from string import digits
from ilab_tools.date_parser import DateParser

class DataProcessor:
    def __init__(self, df=None):
        if df is not None and not isinstance(df, pd.DataFrame):
            raise TypeError(f"Verinin tipi DataFrame olarak gelmedi, veri type: {type(df).__name__}.")
        self.df = df


    @staticmethod
    def get_latest_value(control_date, data_list):
        df = pd.DataFrame(data_list)
        df['date'] = df['date'].astype('datetime64[s]')
        df.sort_values("date", ascending=False, inplace=True)
        return df[df['date'] <= DateParser.remove_ms(control_date)].iloc[0]

    @staticmethod
    def replace_nan_with_none(data):
        if isinstance(data, pd.Series):
            return data.fillna(value=None).to_dict()
        elif pd.isna(data):
            return None
        elif isinstance(data, list):
            return [DataProcessor.replace_nan_with_none(item) for item in data]
        elif isinstance(data, dict):
            return {k: DataProcessor.replace_nan_with_none(v) for k, v in data.items()}
        elif isinstance(data, tuple):
            return tuple(DataProcessor.replace_nan_with_none(item) for item in data)
        elif isinstance(data, set):
            return {DataProcessor.replace_nan_with_none(item) for item in data}
        elif isinstance(data, frozenset):
            return frozenset(DataProcessor.replace_nan_with_none(item) for item in data)
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        elif isinstance(data, pd._libs.missing.NAType) or pd.isna(data):
            return None
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def change_column_types(self, **kwargs):
        for col, dtype in kwargs.items():
            if dtype == int:
                self.df[col] = pd.to_numeric(self.df[col])
            else:
                self.df = self.df.astype({col: dtype})
        return self.df

    # Belirtilen kolonlar icin duplike kayit arar. Eger verilmediyse tum kolonlar icin bakar.
    def data_duplicates(self, columns=None):
        if columns is None: # None ise tum kolonlar icin arar
            columns = self.df.columns.tolist()

        fail_message = None
        duplicate = self.df[self.df[columns].duplicated()][columns]
        if isinstance(columns, str):
            duplicate = {x for x in duplicate if (str(x) != 'nan' and str(x) != 'NaN')}
     
        if len(duplicate):
            if isinstance(columns, str):
                fail_message = "iLabid çoklaması tespit edildi. :warning:" + f"```{set(duplicate)} Çoklayan iLabid'ler.```"
                print(duplicate, "Çoklayan iLabid'ler")
            else:
                fail_message = "Kayıtlarda çoklama tespit edildi. :warning:"
                print(duplicate, "Çoklayan kayıtlar")
        else:
            print(f"{columns} - Duplike kayıt yoktur")
        return fail_message

            

    # Iki df arasindaki kolon farklarini tespit eder.
    @staticmethod
    def column_checker_two_df(df_today_columns, df_old_columns):
        if set(df_today_columns) != set(df_old_columns):
            return f"Dataframeler arasında kolon farkı tespit edildi."
        return None

    @staticmethod
    def n_days_control_data_diff_control(n_previous_days_control, diff_dates):
        close_days = datetime.today() - timedelta(days=n_previous_days_control)
        for d in diff_dates:
            if pd.to_datetime(d) < close_days:
                print(close_days, d)
                return f"Dataframe {n_previous_days_control} günden daha eski kayıt içeriyor."
        return None

    # Kontrol: Kolon null degerler iceriyor mu? 
    def column_null_control(self, column_name):
        if pd.isnull(self.df[column_name]).any():
            return f"{column_name}'de null kayıt mevcut."
        return None

    # df kolonlarindaki ozel karakterleri kaldirma
    def clean_dataframe_columns(self):
        self.df.columns = self.df.columns.str.replace(r'[ .\/#%@&+\n().-]', '', regex=True)
        tr_to_eng = str.maketrans("çğıöşü", "cgiosu")
        self.df.columns = self.df.columns.str.translate(tr_to_eng)
        self.df.columns = [col.lstrip(digits) for col in self.df.columns]
        return self.df.columns

    @staticmethod
    def get_percentage(value1, value2):
        return round(100 * (int(round(value1)) - int(round(value2))) / value1)
