import enum
import pymysql
from sqlalchemy import and_, desc, insert, or_, text, create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import json

pymysql.install_as_MySQLdb()

class MysqlManager:

    def __init__(self, db_info):
        self.db_info = db_info
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False)
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)

    def _create_engine(self):
        host = self.db_info['host_name']
        username = self.db_info['user_name']
        password = self.db_info['user_password']
        db_name = self.db_info['db_name']

        return create_engine(
            f"mysql+pymysql://{username}:{password}@{host}/{db_name}",
            echo=False,
            pool_pre_ping=True
        )

    def get_session(self):
        return self.SessionLocal()

    def get_table(self, table_name):
        return self.metadata.tables.get(table_name)

    def fetch_one(self, query: str):
        with self.get_session() as session:
            try:
                result = session.execute(text(query)).scalar()
                if isinstance(result, str):
                    return json.loads(result)
                return result
            except Exception as e:
                raise Exception(f"Failed to fetch data: {e}")

    def update_data_via_query(self, query: str):
        with self.get_session() as session:
            try:
                session.execute(text(query))
                session.commit()
            except Exception as e:
                raise Exception(f"Failed to update data: {e}")
    @staticmethod
    def collect_filters(table, filters):
        filter_queries = []
        for (filter_col, filter_opt), filter_val in filters.items():
            if filter_opt == "equals":
                filter_queries.append(table.columns[filter_col] == filter_val)
            elif filter_opt == "in":
                filter_queries.append(table.columns[filter_col].in_(filter_val))
            elif filter_opt == "less_than":
                filter_queries.append(table.columns[filter_col] < filter_val)
            else:
                raise ValueError(f"Unsupported filter operation: {filter_opt}")
        return filter_queries

    @staticmethod
    def filter_by_args(table, query_obj, filters_dict: dict):
        for operator, filters in filters_dict.items():
            filter_queries = MysqlManager.collect_filters(table, filters)
            if operator == 'and':
                query_obj = query_obj.filter(and_(*filter_queries))
            elif operator == 'or':
                query_obj = query_obj.filter(or_(*filter_queries))
            else:
                raise ValueError(f"Unsupported operator: {operator}")
        return None if query_obj.count() == 0 else query_obj


    def get_rows(self, table, filters=None):
        with self.get_session() as session:
            try:
                query_obj = session.query(table)
                if filters:
                    query_obj = MysqlManager.filter_by_args(table, query_obj, filters)
                return query_obj
            except Exception as e:
                raise Exception(f"Failed to retrieve rows: {e}")

    def add_row(self, table, row: dict):
        with self.get_session() as session:
            try:
                session.execute(insert(table).values(**row))
                session.commit()
            except Exception as e:
                raise Exception(f"Failed to add row: {e}")

    def update_row(self, table, filters, new_cols_vals: dict):
        with self.get_session() as session:
            try:
                query_obj = self.get_rows(table, filters)
                if query_obj:
                        query_obj.update(new_cols_vals)
                        session.commit()
            except Exception as e:
                raise Exception(f"Failed to update row: {e}")

    def row_setdefault(self, table, filters):
        with self.get_session() as session:
            try:
                query_obj = session.query(table)
                query_obj = MysqlManager.filter_by_args(table, query_obj, filters)
                row = query_obj.first() 

                if row is None: 
                    create_values = {key[0]: value for key, value in filters["and"].items()}
                    self.add_row(table, create_values)
                else:
                    print("Row exists. No insertion..")
            except Exception as e:
                raise Exception(f"Failed to set default row: {e}")

    def get_row_info(self, table, filters):
        with self.get_session() as session:
            try:
                query_obj = self.get_rows(table, filters)
                if query_obj:
                    row = query_obj[0]
                    row_vals = list(row)
                    cols = [col.name for col in table.columns]
                    return dict(zip(cols, row_vals))
                return None
            except Exception as e:
                raise Exception(f"Failed to retrieve row info: {e}")

class ScriptStatusCodes(enum.Enum):
    not_work = 0
    working = 1
    success = 2
    fail = 3
    skipped = 4
