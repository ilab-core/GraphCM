import json
import os

import pytest
from dotenv import load_dotenv

from ilab_tools.faiss import FaissIndex
from ilab_tools.storage import load_pickle

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path, override=True)


@pytest.fixture
def config():
    config_data = {}
    config_data["gcp_creds"] = json.loads(os.getenv("GCLOUD_SERVICE_KEY"))
    config_data["gcs_bucket"] = "ilab_tools"
    return config_data


@pytest.fixture
def embeddings(config):
    emb = {}
    emb["type_1"] = load_pickle(config["gcp_creds"], config["gcs_bucket"],
                                "tests/embeddings/", "LaBSE_embeddings.pkl")
    emb["type_2"] = load_pickle(config["gcp_creds"], config["gcs_bucket"],
                                "tests/embeddings/", "CLIP_embeddings.pkl")
    return emb


@pytest.fixture
def indexes(config):
    indexes = {}
    indexes["type_1"] = FaissIndex(verbose=False).load_from_gcs(config["gcp_creds"], config["gcs_bucket"],
                                                                "tests/faiss_indexes/", "LaBSE_embddings.index.pkl")
    indexes["type_2"] = FaissIndex(verbose=False).load_from_gcs(config["gcp_creds"], config["gcs_bucket"],
                                                                "tests/faiss_indexes/", "CLIP_embddings.index.pkl")
    return indexes
