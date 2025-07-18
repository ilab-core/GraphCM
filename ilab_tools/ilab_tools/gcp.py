import os
import json
import time
import pickle
import pandas as pd
from google.cloud import bigquery, storage
from google.oauth2 import service_account
from google.cloud.exceptions import NotFound
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
from datetime import datetime, timedelta
import gspread
import csv
from io import StringIO


class BigQueryClient:
    def __init__(self, credentials_path, project_id=None):
        self.credentials_path = credentials_path
        self.project_id = project_id
        self.client = bigquery.Client.from_service_account_json(
            credentials_path, project=project_id
        )

    def query_to_dataframe(self, query):
        job = self.client.query(query)
        return job.to_dataframe()
    
    # table_id = "your-project.your_dataset.your_table_name"
    def table_exists(self, table_id):
        try:
            self.client.get_table(table_id)
            print(f"Tablo bulundu: {table_id}")
            return True
        except NotFound:
            print(f"Tablo bulunamadı: {table_id}")
            return False

    # table_id = "your-project.your_dataset.your_table_name"
    def load_dataframe_to_table(
        self,dataframe, table_id, schema=None, overwrite=False, partition_field=None
        ):
        dataframe = dataframe[[field.name for field in schema]] if schema else dataframe
        job_config = bigquery.LoadJobConfig(schema=schema)
        job_config.create_disposition = "CREATE_IF_NEEDED"
        job_config.write_disposition = "WRITE_TRUNCATE" if overwrite else "WRITE_APPEND"
        if partition_field:
            job_config.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=partition_field,
            )
        try:
            self.client.load_table_from_dataframe(dataframe, table_id, job_config=job_config)
        except Exception as e:
            print(f"Yukleme islemi basarisiz -> {e}")
        
    def create_table_query(self, config, table_id, start_date=None, end_date=None, extra_filters=None):
        if start_date is None and end_date is None:
            query = config["bq_query_all"].format(
                config["project_id"],
                config["dataset_id"],
                table_id,
            )
        else:
            if start_date:
                start_date = start_date.strftime("%Y%m%d")
            if end_date:
                end_date = end_date.strftime("%Y%m%d")

            if start_date and end_date:
                query = config["bq_query"].format(
                    config["project_id"],
                    config["dataset_id"],
                    table_id,
                    start_date,
                    end_date,
                )
            elif start_date:
                query = config["bq_query_day"].format(
                    config["project_id"], config["dataset_id"], table_id, start_date
                )
            else: # end_date verilmis. start_date verilmemis
                raise ValueError("end_date tek basina yeterli degildir. start_date belirtilmelidir.")

        if extra_filters:
            query += f" {extra_filters}"

        print(f"Oluşturulan tablo sorgusu: {query}")
        return query


    def get_data(self, data_type, config, start_date=None, end_date=None, extra_filters=None):

        # string -> datetime
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")


        query = self.create_table_query(
            config, config[f"{data_type}_table"], start_date, end_date, extra_filters
        )

        print(f"Son sorgu: {query}")
        df = self.query_to_dataframe(query)
        return df

    def get_table_to_csv(self, path, table_name, config, data_type):
        export_path = os.path.join(
            path, f"{table_name}.{data_type}"
        )
        df = self.get_data(table_name, config, start_date=None, end_date=None)
        df.to_csv(export_path, index=False)
        print(
            f"SUCCESS - '{table_name}' tablosu CSV formatında, {export_path} konumuna kaydedildi: "
        )
        time.sleep(45)

    def run_script(self, bq_query, s_date=None):
        try:
            formatted_query = bq_query.format(s_date) if s_date else bq_query
            job = self.client.query(formatted_query)
            job.result()

            if job.state == 'DONE':
                if job.error_result:
                    error_message = f"Sorgu sonucu hata almıştır: {job.error_result}"
                    print(error_message)  
                    raise Exception(error_message)  
                else:
                    print("Başarılı şekilde sorgu çalışmıştır")
                    return True 
            else:
                error_message = f"Sorgu sonucu beklenmeyen hata almıştır: Job state: {job.state}"
                print(error_message) 
                raise Exception(error_message) 

        except Exception as e:
            error_message = f"Bir hata oluştu: {e}"
            print(error_message) 
            raise  

    def fetch_query_results_as_rows(self, query):
        query_job = self.client.query(query)
        query_job.result()

        destination = query_job.destination
        destination = self.client.get_table(destination)
        print("Sorgu sonuçları başarıyla satır formatında alındı.")
        return self.client.list_rows(destination, page_size=10000)


class CloudStorageClient:
    def __init__(self, credentials_path, project_id=None):
        with open(credentials_path) as f:
            credentials_dict = json.load(f)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict
        )
        self.client = storage.Client(project=project_id, credentials=credentials)

    def blob_exists(self, bucket_name, file_name):
        try:
            bucket = self.client.get_bucket(bucket_name)
            blob = bucket.blob(file_name)
            return blob.exists() # True
        except Exception as e:
            print(f"ERROR - Blob {file_name} bulunamadi: {str(e)}")
            return False

    def download_pickle(self, bucket_name, file_name):
        try:
            if not self.blob_exists(bucket_name, file_name):  
                return None
            else: 
                bucket = self.client.get_bucket(bucket_name)
                blob = bucket.blob(file_name) 
                data = blob.download_as_string()
                pickle_data = pickle.loads(data)
                print(f"SUCCESS - {file_name} dosyasi indirildi..")
                return pickle_data
        except (EOFError, pickle.UnpicklingError, RecursionError) as e: 
            print(f"ERROR - Pickle dosyasi okunurken hata olustu: {str(e)}")
            return []
        except Exception as e:
            print(f"ERROR - {str(e)}")
            return []

    def upload_pickle(self, bucket_name, file_name, data):
        try:
            if self.blob_exists(bucket_name, file_name):
                print(f"WARNING - Blob {file_name} zaten mevcut. Dosya üzerine yazılıyor..")
            bucket = self.client.get_bucket(bucket_name)
            blob = bucket.blob(file_name)
            serialized_data = pickle.dumps(data)
            blob.upload_from_string(serialized_data)
            print(f"SUCCESS - Pickle dosyasi {file_name} yuklendi..")
            
        except Exception as e:
            print(f"ERROR - Pickle dosyasi yuklenirken hata olustu: {str(e)}")

    def download_csv(self, bucket_name, file_name):
        try:
            if not self.blob_exists(bucket_name, file_name):
                return []
            else:
                bucket = self.client.bucket(bucket_name)
                blob = bucket.blob(file_name)
                csv_data = blob.download_as_text()
                csv_reader = csv.reader(StringIO(csv_data))
                rows = [row for row in csv_reader]
                if not rows:
                    print(f"ERROR - CSV dosyası boş: {file_name}")
                    return pd.DataFrame(), []

                header = rows[0]
                data = rows[1:]

                df = pd.DataFrame(data, columns=header)
                print(f"SUCCESS - {file_name} dosyasının içeriği alındı.")
                
                return df
        except Exception as e:
            print(f"ERROR - CSV dosyasi okunurken hata olustu: {str(e)}")
            return []

    def upload_csv(self, bucket_name, file_name, rows):
        try:
            if self.blob_exists(bucket_name, file_name):
                print(f"WARNING - Blob {file_name} zaten mevcut. Dosya üzerine yazılıyor..")
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(file_name)

            output = StringIO()
            csv_writer = csv.writer(output)
            csv_writer.writerows(rows)
            csv_data = output.getvalue()

            blob.upload_from_string(csv_data, content_type="text/csv")
            print(f"SUCCESS - CSV dosyasi {file_name} yuklendi..")
        except Exception as e:
            print(f"ERROR - CSV dosyasi yuklenirken hata olustu: {str(e)}")


    def get_bucket(self, bucket_name):
        return self.client.bucket(bucket_name)


class SpreadsheetClient:
    def __init__(self, credentials_path, scopes):
        self.credentials = ServiceAccountCredentials.from_json_keyfile_name(
            credentials_path, scopes
        )
        self.client = gspread.authorize(self.credentials)

    def get_spreadsheet_data(self, spreadsheet_name):
        try:
            sheet = self.client.open(spreadsheet_name)
            sheet_instance = sheet.get_worksheet(0)
            records = sheet_instance.get_all_values()
            df = pd.DataFrame(records)
            df = df.iloc[1:, 1:]  # index kolonu kaldirma
            header_row = df.iloc[0]
            df.columns = header_row
            df = df[1:]
            print(f"SUCCESS - '{spreadsheet_name}' isimli tablo alındı.")
            return df
        except Exception as e:
            print(
                f"ERROR - '{spreadsheet_name}' isimli tablo alınırken hata oluştu: {e}"
            )
            raise Exception(f"ERROR - Spreadsheet alınırken hata oluştu: {e}")


