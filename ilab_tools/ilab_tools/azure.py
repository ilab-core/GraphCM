import logging
import time

import pandas as pd
import pyodbc


class AzureSQLConnector:
    def __init__(self, driver, server, database, username, password):
        self.driver = driver
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.conn = None

    def get_conn(self):
        try:
            self.conn = pyodbc.connect(f"""DRIVER={self.driver};
                                           SERVER={self.server};
                                           PORT=1433;DATABASE={self.database};
                                           UID={self.username};
                                           PWD={self.password}""")
            logging.info("CONNECTED SUCCESSFULLY")
        except Exception as e:
            print(f"Auth Failed: {e}")
        return self.conn

    def get_data_with_query(self, query, chunk_sizes=1000, return_df_type=True, logging_flag=False):
        if self.conn is None:
            raise ConnectionError("ACTIVE CONNECTION NEEDED!")

        df_chunks = pd.read_sql(query, self.conn, chunksize=chunk_sizes)
        dfs = []
        chunk_iterator = 0
        start_data_collect = time.time()

        for chunk in df_chunks:
            start_chunk = time.time()
            dfs.append(chunk)
            end_chunk = time.time()
            if logging_flag:
                logging.info(
                     f"{chunk_iterator} CHUNK APPENDED WITH {(end_chunk-start_chunk) * 10**3} MS")
            chunk_iterator += 1

        end_data_collect = time.time()
        if logging_flag:
            logging.info(f"TOTAL CHUNKS: {chunk_iterator}")
            logging.info(
                f"TOTAL TIME: {(end_data_collect-start_data_collect) * 10**3} MS")
            logging.info("CHUNKS HAVE BEEN SUCCESSFULLY CONCATENATED!")

        df = pd.concat(dfs, ignore_index=True)
        return_df = df if return_df_type else dfs
        return return_df
