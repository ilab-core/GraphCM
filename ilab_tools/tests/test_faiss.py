import random

import faiss
import numpy as np
import pytest
from sklearn.metrics import precision_score
from sklearn.metrics.pairwise import cosine_similarity

from ilab_tools.faiss import FaissIndex
from ilab_tools.storage import storage_connection


def test_make_index(embeddings):
    index1 = FaissIndex(verbose=False).make(embeddings["type_1"])
    assert isinstance(index1, FaissIndex)
    assert isinstance(index1.index_, faiss.swigfaiss_avx2.IndexFlat)
    assert len(index1.get_embeddings()) == len(embeddings["type_1"])
    assert len(index1.get_pids()) == len(embeddings["type_1"])

    index2 = FaissIndex(verbose=False).make(embeddings["type_2"])
    assert isinstance(index2, FaissIndex)
    assert isinstance(index2.index_, faiss.swigfaiss_avx2.IndexFlat)
    assert len(index2.get_embeddings()) == len(embeddings["type_2"])
    assert len(index2.get_pids()) == len(embeddings["type_2"])


def test_load_index(config):
    index1 = FaissIndex(verbose=False).load_from_gcs(config["gcp_creds"], config["gcs_bucket"],
                                                     "tests/faiss_indexes/", "LaBSE_embddings.index.pkl")
    assert isinstance(index1, FaissIndex)
    assert isinstance(index1.index_, faiss.swigfaiss_avx2.IndexFlatIP)
    assert index1.pid_to_data[random.choice(index1.get_pids())].shape == (768,)
    assert sorted(list(index1.pid_to_data.keys())) == sorted(list(index1.idx_to_pid.values()))

    index2 = FaissIndex(verbose=False).load_from_gcs(config["gcp_creds"], config["gcs_bucket"],
                                                     "tests/faiss_indexes/", "CLIP_embddings.index.pkl")
    assert isinstance(index2, FaissIndex)
    assert isinstance(index2.index_, faiss.swigfaiss_avx2.IndexFlatIP)
    assert index2.pid_to_data[random.choice(index2.get_pids())].shape == (512,)
    assert sorted(list(index2.pid_to_data.keys())) == sorted(list(index2.idx_to_pid.values()))


def test_load_save_index(config, indexes):
    indexes["type_1"].save_to_gcs(config["gcp_creds"], config["gcs_bucket"],
                                  "tests/faiss_indexes/", "temp_LaBSE_embddings.index.pkl")
    loaded_index1 = FaissIndex().load_from_gcs(config["gcp_creds"], config["gcs_bucket"],
                                               "tests/faiss_indexes/", "temp_LaBSE_embddings.index.pkl")
    assert isinstance(loaded_index1, FaissIndex)
    assert isinstance(loaded_index1.index_, faiss.swigfaiss_avx2.IndexFlatIP)

    my_bucket = storage_connection(config["gcp_creds"], config["gcs_bucket"])
    path = "tests/faiss_indexes/temp_LaBSE_embddings.index.pkl"
    blob = my_bucket.blob(path)
    assert blob.exists()
    # cleaning up so that the test can be run again.
    blob.delete()

    with pytest.raises(Exception):
        FaissIndex.load_from_gcs(config["gcp_creds"], config["gcs_bucket"],
                                 "tests/faiss_indexes/", "temp_LaBSE_embddings.index.pkl")


def test_search(embeddings, indexes):
    keys_1 = list(embeddings["type_1"].keys())
    random_key_1 = random.choice(keys_1)
    f1 = indexes["type_1"].search(random_key_1, k=50)
    c1 = cosine_similarity(embeddings["type_1"][random_key_1].reshape(1, -1),
                           np.array(list(embeddings["type_1"].values())))
    c1 = [keys_1[ix] for ix in np.argsort(c1[0])[::-1][:50]]

    assert precision_score(c1, list(f1.keys()), average='macro', zero_division=0) >= 0.85
    assert precision_score(c1[:20], list(f1.keys())[:20], average='macro', zero_division=0) >= 0.85

    image_keys = list(embeddings["type_2"].keys())
    random_key_image = random.choice(image_keys)
    f2 = indexes["type_2"].search(random_key_image, k=100)
    c2 = cosine_similarity(embeddings["type_2"][random_key_image].reshape(1, -1),
                           np.array(list(embeddings["type_2"].values())))
    c2 = {image_keys[ix]: c2[0][ix] for ix in np.argsort(c2[0])[::-1][:100]}

    diff_in_100 = sum(abs(a - b) > 0.000001 for a, b in zip(list(c2.values()), list(f2.values())))
    diff_in_50 = sum(abs(a - b) > 0.000001 for a, b in zip(list(c2.values())[:50], list(f2.values())[:50]))

    assert diff_in_100 < 2 or precision_score(list(c2.keys()), list(f2.keys()), average='macro', zero_division=0) >= 0.7
    assert diff_in_50 < 2 or precision_score(list(c2.keys())[:50], list(
        f2.keys())[:50], average='macro', zero_division=0) >= 0.6
