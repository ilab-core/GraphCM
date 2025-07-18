import functools
import time
from pymongo import MongoClient, UpdateOne
from pymongo.errors import AutoReconnect, ServerSelectionTimeoutError, ConnectionFailure
from bson import ObjectId
# from slack import send_message


class MongoDBClient:
    def __init__(
        self, host, username, password, port, auth_source, reconnection_limit=3
    ):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.auth_source = auth_source
        self.reconnection_limit = reconnection_limit
        self.client = self.connect()

    def connect(self):
        try:
            client = MongoClient(
                host=self.host,
                username=self.username,
                password=self.password,
                port=int(self.port),
                authSource=self.auth_source,
            )
            print("SUCCESS: MongoDB bağlantısı kuruldu.")
            return client
        except Exception as e:
            print(f"FAIL: MongoDB bağlantısında sorun oluştu.. {e}")
            return None

    def wait_until_reconnection(function):
        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            try:
                return function(self, *args, **kwargs)
            except (AutoReconnect, ServerSelectionTimeoutError, ConnectionFailure) as e:
                
                for trial in range(self.reconnection_limit):
                    if not self.is_active():
                        if trial != (self.reconnection_limit - 1):
                            print(
                                f"(try {trial+1}) MongoDB encountered an error; attempting to reconnect."
                            )
                            time.sleep(60)
                        else:
                            message = f"MongoDB bir hatayla karşılaştı ve geri bağlanamadı (deneme sayısı:{trial+1}) :red_circle:"
                            print(message)  # TODO: send_message(message)
                            raise e
                    else:
                        print("MongoDB has reconnected.")
                        return function(self, *args, **kwargs)

        return wrapper
  
    @wait_until_reconnection
    def get_collection(self, db_name, collection_name):
        db = self.client[db_name]
        collection = db[collection_name]
        print(f"Database: {db_name} - Collection: {collection_name} a erişildi.")
        return collection
 
    @wait_until_reconnection
    def find_documents(self, db_name, collection_name, query={}, limit=10):
        collection = self.get_collection(db_name, collection_name)
        if collection is not None:
            documents = collection.find(query).limit(limit)
            result = list(documents)
            print(f"{len(result)} doküman bulundu: ")
            for doc in result:
                print(doc)
            return result
        else:
            print("Collection bulunamadi.")
            return None

    def is_active(self):
        try:
            self.client.server_info()
            print("MongoDB aktif")
            return True
        except (ServerSelectionTimeoutError, AutoReconnect) as e:
            print(f"MongoDB aktif değil ({e})")
            self.client = self.connect()
            return self.client is not None

    @wait_until_reconnection
    def upsert_records(self, db_name, collection_name, batch_operations):
        collection = self.get_collection(db_name, collection_name)
        bulk_operations = []
        for op in batch_operations:
            op_key = op["key"]

            if isinstance(op_key, int):
                op_key = str(op_key)  
        
            try:
                object_id = ObjectId(op_key)  
            except Exception as e:
                raise ValueError(f"Invalid ObjectId format for key: {op_key}") from e
            
            bulk_operations.append(UpdateOne({"_id": object_id}, {"$set": op["val"]}, upsert=True))
        
        result = collection.bulk_write(bulk_operations)
        print(
            "Matched:",
            result.matched_count,
            "Modified:",
            result.modified_count,
            "Upserts:",
            result.upserted_count,
        )

    @wait_until_reconnection
    def get_ids_exist(self, db_name, collection_name, product_ids, batch_size=50000):
        collection = self.get_collection(db_name, collection_name)
        existing_ids = set()
        for i in range(0, len(product_ids), batch_size):
            batch = product_ids[i : i + batch_size]
            query = {"_id": {"$in": batch}}
            products = collection.find(query, {"_id": 1})
            existing_ids.update(product["_id"] for product in products)

        return existing_ids
