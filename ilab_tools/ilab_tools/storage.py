import pickle

import google
from google.cloud import storage


def storage_connection(creds_info, bucket_name):
    my_creds = google.oauth2.service_account.Credentials.from_service_account_info(creds_info)
    storage_client = storage.Client(credentials=my_creds)
    bucket = storage_client.get_bucket(bucket_name)
    return bucket


def load_pickle(creds_info, bucket_name, folder_path, filename):
    my_bucket = storage_connection(creds_info, bucket_name)
    blob = my_bucket.blob(folder_path + filename)
    if not blob.exists():
        return False
    with blob.open(mode='rb') as f:
        content = pickle.load(f)
    return content


def save_pickle(creds_info, bucket_name, folder_path, filename, file):
    my_bucket = storage_connection(creds_info, bucket_name)
    blob = my_bucket.blob(folder_path + filename)
    with blob.open(mode='wb') as f:
        pickle.dump(file, f)
