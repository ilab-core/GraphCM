import datetime as dt

import faiss
import numpy as np

from ilab_tools.storage import load_pickle, save_pickle


class FaissIndex():
    def __init__(self, verbose=True):
        self.index_ = None
        self.pid_to_data = dict()
        self.idx_to_pid = dict()
        self.verbose = verbose

    def make(self, pid_to_data):
        self.pid_to_data = pid_to_data
        self.idx_to_pid = {idx: pid for idx, pid in enumerate(list(pid_to_data.keys()))}
        embeddings = np.array(list(self.pid_to_data.values()))
        self.index_ = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        self.index_.add(embeddings)
        return self

    def get_pids(self):
        return list(self.pid_to_data.keys())

    def get_embeddings(self):
        return self.pid_to_data

    def pid_exists(self, pid):
        return pid in self.pid_to_data

    def get_embedding(self, pid, reshape=(1, -1)):
        if not self.pid_exists(pid):
            return None
        if reshape:
            return self.pid_to_data[pid].reshape(reshape)
        return self.pid_to_data[pid]

    def search(self, pid, k):
        if not self.pid_exists(pid):
            return dict()
        S, IX = self.index_.search(self.get_embedding(pid), k)
        similar_pids = [self.idx_to_pid[i] for i in IX[0]]
        return dict(zip(similar_pids, S[0]))

    def load_from_gcs(self, creds, bucket, folder, filename):
        read_data = load_pickle(creds, bucket, folder, filename)
        self.index_ = faiss.deserialize_index(read_data["index_"])
        self.pid_to_data = read_data["pid_to_data"]
        self.idx_to_pid = read_data["idx_to_pid"]
        if self.verbose:
            print(f"{self._now()} FaissIndex başarıyla yüklendi: {folder}{filename}")
            print(f"{self._now()} Bulunan item sayısı: {len(read_data['idx_to_pid'])}")
        return self

    def save_to_gcs(self, creds, bucket, folder, filename):
        write_data = dict()
        write_data["index_"] = faiss.serialize_index(self.index_)
        write_data["pid_to_data"] = self.pid_to_data
        write_data["idx_to_pid"] = self.idx_to_pid
        save_pickle(creds, bucket, folder, filename, write_data)
        if self.verbose:
            print(f"{self._now()} Bulunan item sayısı: {len(write_data['idx_to_pid'])}")
            print(f"{self._now()} FaissIndex başarıyla kaydedildi: {folder}{filename}")
        return self

    def _now(self, format='%Y-%m-%d %H:%M:%S'):
        return dt.datetime.now().strftime(format)
