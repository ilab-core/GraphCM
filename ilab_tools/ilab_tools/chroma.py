import random
import threading
import time
from functools import wraps

import chromadb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity

from .metrics import count_different, ndcg_score


def wait_until_reconnection(retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    if "connect" in str(type(e).__name__).lower():
                        time.sleep(20)
                        attempt += 1
                        print(f"Attempt {attempt} failed. Retrying... \nException: {e}")
                        if attempt == retries:
                            raise TimeoutError(f"Failed after {retries} attempts.")
                    else:
                        raise Exception(f"\nException: {e}")
        return wrapper
    return decorator


class ChromaDBClient:

    def __init__(self, host, port, collection_name, connection_timeout_threshold=60):
        self.host = host
        self.port = port
        self.connection_timeout_threshold = connection_timeout_threshold
        self.client = self.get_client()
        self.collection = self.get_db_collection(collection_name)

    def connect_with_timeout(self, connection_timeout_threshold, func, *args, **kwargs):
        def target(result):
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                result[1] = e

        result = [None, None]
        thread = threading.Thread(target=target, args=(result,))
        thread.start()

        thread.join(connection_timeout_threshold)
        if thread.is_alive():
            raise TimeoutError(f"Connection attempt timed out after {connection_timeout_threshold} seconds.")
        if result[1]:
            raise result[1]
        return result[0]

    def get_client(self):
        client = self.connect_with_timeout(self.connection_timeout_threshold,
                                           chromadb.HttpClient, host=self.host, port=self.port)
        return client

    @wait_until_reconnection()
    def get_db_collection(self, collection_name):
        return self.client.get_collection(name=collection_name)

    @wait_until_reconnection()
    def create_collection(self, collection_name, metadata_configration):
        try:
            self.client.create_collection(
                name=collection_name,
                metadata=metadata_configration
            )
        except Exception as e:
            print(f'Could not create the collection.\nException: {e}')

    @wait_until_reconnection()
    def delete_records(self, ids_to_delete):
        try:
            self.collection.delete(ids=ids_to_delete)
            print(f"Succesfully deleted {len(ids_to_delete)} ids")
        except Exception as e:
            print(f'Could not delete ids: {", ".join(ids_to_delete)}.\nException: {e}')

    @wait_until_reconnection()
    def delete_collection(self, collection_name):
        self.client.delete_collection(name=collection_name)
        print(f"Succesfully deleted collection '{collection_name}'")

    @wait_until_reconnection()
    def get_collection_sizes(self):
        chromadb_collection_infos = {}
        for collection in self.client.list_collections():
            collection_name = str(collection.name)
            chromadb_collection_infos[str(collection.name)] = self.client.get_collection(name=collection_name).count()
        return chromadb_collection_infos

    @wait_until_reconnection()
    def get_collection_info(self, collection_name):
        collection = self.get_db_collection(collection=collection_name)
        return collection.metadata

    @wait_until_reconnection()
    def get_embedding(self, job_ref_no):
        job_ref_no = [job_ref_no] if type(job_ref_no) is not list else job_ref_no
        data = self.collection.get(ids=job_ref_no, include=["embeddings"])
        id_to_embedding = dict(zip(data['ids'], data['embeddings']))
        embeddings = [id_to_embedding[id] for id in job_ref_no]
        return embeddings

    @wait_until_reconnection()
    def get_all_items(self, collection_name):
        collection = self.get_db_collection(collection_name)
        all_items = collection.get(include=["embeddings"])
        all_items_dict = dict(zip(all_items['ids'], all_items['embeddings']))
        return all_items_dict

    @wait_until_reconnection()
    def search(self,
               job_ref_no,
               metadata_filter=None,
               result_metadata_keys=[],
               n_results=20,
               distance_threshold=None,
               not_exist_ok=False):
        job_ref_no = [job_ref_no] if type(job_ref_no) is not list else job_ref_no
        list_filtered_results = []
        list_metadata_cols = list(self.collection.get(include=['metadatas'], limit=1)['metadatas'][0].keys())
        list_mismatched_metadata_cols = [elem for elem in result_metadata_keys if elem not in list_metadata_cols]
        if not list_mismatched_metadata_cols:
            for index_job_ref_no in job_ref_no:
                try:
                    embedding = self.get_embedding(job_ref_no=index_job_ref_no)
                except Exception:
                    if not_exist_ok:
                        list_filtered_results.append(None)
                        continue
                    else:
                        raise KeyError(f"Could not find id: {index_job_ref_no}")

                results = self.collection.query(
                    query_embeddings=embedding,
                    where=metadata_filter,
                    n_results=(n_results + 1)
                )

                for result_metadata in results['metadatas']:
                    result_filtered_response = [{key: item[key] for key in result_metadata_keys}
                                                for item in result_metadata]
                    for i, item in enumerate(result_filtered_response):
                        item['id'] = results['ids'][0][i]
                        item['distance'] = results['distances'][0][i]
                    if distance_threshold is not None:
                        result_filtered_response = [result
                                                    for result in result_filtered_response
                                                    if float(result["distance"]) < float(distance_threshold)]
                result_filtered_response = [item
                                            for item in result_filtered_response
                                            if item['id'] != index_job_ref_no]
                list_filtered_results.append(result_filtered_response)
            return list_filtered_results
        else:
            raise ValueError(f'"Could not find requested columns: {list_mismatched_metadata_cols}')

    @wait_until_reconnection()
    def add_items(self, df_embedding, batch_size, metadata_columns, id_column):
        df_embedding[id_column] = df_embedding[id_column].astype(str)
        df_embedding = df_embedding.fillna('None')

        embeddings = []
        ids = []
        metadatas = []
        batch_index = 0
        batch_round = 0
        start_full_load = time.time()

        for _, row in df_embedding.iterrows():
            embedding = row['Embedding'].tolist()[0]
            id = row[id_column]
            metadata = {col: row[col] for col in metadata_columns}
            embeddings.append(embedding)
            ids.append(id)
            metadatas.append(metadata)

            if (batch_index + 1) % batch_size == 0:
                start_add_batch = time.time()
                self.collection.upsert(embeddings=embeddings, ids=ids, metadatas=metadatas)
                embeddings.clear()
                ids.clear()
                metadatas.clear()
                end_add_batch = time.time()
                print(f"Batch {batch_round} uploaded in {round((end_add_batch - start_add_batch) * 10**3, 2)} ms")
                batch_round += 1

            batch_index += 1
        if embeddings:
            self.collection.upsert(embeddings=embeddings, ids=ids, metadatas=metadatas)
            print(f"All rows uploaded in {round((time.time() - start_full_load) * 10**3, 2)} ms")

    @wait_until_reconnection()
    def download_collection_data(self, collection_name, file_path=None, rename_columns={'ids': 'id', 'embeddings': 'Embedding'}):
        list_collection_data = []
        file_path_with_format = None if file_path else file_path + '.pkl'
        collection = self.get_db_collection(collection=collection_name)
        data = collection.get()
        for data_column in ['ids', 'embeddings', 'metadatas']:
            df = pd.DataFrame(data[data_column])
            list_collection_data.append(df)
        final_df = pd.concat(list_collection_data)
        final_df.rename(columns=rename_columns, inplace=True)
        if file_path_with_format:
            try:
                final_df.to_pickle(file_path_with_format)
                print(f'Succesfully exported to {file_path_with_format}')
            except Exception as e:
                print(f'Could not export.\nException: {e}')
        return final_df

    def evaluate(self, collection_name, test_size, n_results, tolerance):
        all_embeddings_dict = self.get_all_items(collection_name)
        all_embeddings = list(all_embeddings_dict.values())
        all_ids = list(all_embeddings_dict.keys())

        if test_size <= 1:
            num_ids_to_test = int(len(all_ids) * test_size)
        else:
            num_ids_to_test = int(test_size)
        print(f"Number of ids to test: {num_ids_to_test}")
        ids_to_test = random.sample(all_ids, num_ids_to_test)

        chroma_results = []
        chroma_time = []
        cosine_results = []
        cosine_time = []
        for job_ref_no in ids_to_test:
            try:
                embedding = self.get_embedding(job_ref_no=job_ref_no)
            except Exception:
                raise KeyError(f"Could not find id: {job_ref_no}")

            start_time = time.time()
            chroma_similars = self.search(job_ref_no, n_results=n_results)
            total_time = time.time() - start_time
            chroma_result = {item['id']: 1 - item['distance'] for item in chroma_similars[0]}
            chroma_time.append(total_time)
            chroma_results.append(chroma_result)

            start_time = time.time()
            cosine_similars = cosine_similarity(embedding, all_embeddings)[0]
            total_time = time.time() - start_time
            cosine_result = {all_ids[idx]: cosine_similars[idx]
                             for idx in np.argsort(cosine_similars)[::-1][1:n_results + 1]}
            cosine_time.append(total_time)
            cosine_results.append(cosine_result)

        precisions = [precision_score(list(cosine_results[i].keys()), list(chroma_results[i].keys()), average='macro', zero_division=0)
                      for i in range(num_ids_to_test)]
        recalls = [recall_score(list(cosine_results[i].keys()), list(chroma_results[i].keys()), average='macro', zero_division=0)
                   for i in range(num_ids_to_test)]
        ndcgs = [ndcg_score(list(cosine_results[i].keys()), list(chroma_results[i].keys()), n_results)
                 for i in range(num_ids_to_test)]
        diffs = [count_different(list(cosine_results[i].values()), list(chroma_results[i].values()), tolerance)
                 for i in range(num_ids_to_test)]

        avg_chroma_time = sum(chroma_time) / len(chroma_time)
        avg_cosine_time = sum(cosine_time) / len(cosine_time)
        avg_precision = round(sum(precisions) / len(precisions), 2)
        avg_recall = round(sum(recalls) / len(recalls), 2)
        avg_ndcg = round(sum(ndcgs) / len(ndcgs), 2)
        sum_diffs = sum(diffs)
        avg_diffs = round(sum_diffs / len(diffs), 2)

        print(f"Average Chroma (search + filter) time: {avg_chroma_time}")
        print(f"Average Cosine Similarity time: {avg_cosine_time}")
        print(f"Precision: {avg_precision}")
        print(f"Recall: {avg_recall}")
        print(f"NDCG: {avg_ndcg}")
        print(f"Total Count of Different Similarities: {sum_diffs}, Tolerance: {tolerance}")
        print(f"Mean of Different Similarities Per Search: {avg_diffs}, Tolerance: {tolerance}")

        return avg_precision, avg_recall, avg_ndcg, sum_diffs, avg_diffs
