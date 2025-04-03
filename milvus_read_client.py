import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
import numpy as np
import logging
from typing import List, Tuple, Union

# Load .env variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("milvus-client")


def get_milvus_connection(store_id: int) -> Tuple[str, str]:
    # Example: 10 stores per shard
    shard_index = (store_id - 1) // 10 + 1
    host = os.getenv(f"MILVUS_SHARD_{shard_index}_HOST", os.getenv("MILVUS_LOCAL_HOST", "localhost"))
    port = os.getenv(f"MILVUS_SHARD_{shard_index}_PORT", os.getenv("MILVUS_LOCAL_PORT", "19530"))
    return host, port


class MilvusReIDClient:
    def __init__(self, store_id: int, collection_name: str = "person_embeddings"):
        self.store_id = store_id
        self.collection_name = collection_name

        host, port = get_milvus_connection(store_id)
        logger.info(f"[store_id={store_id}] Connecting to Milvus at {host}:{port}...")
        connections.connect("default", host=host, port=port)
        logger.info("Connected to Milvus.")

        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        if self.collection_name in [col.name for col in Collection.list()]:
            logger.info(f"Collection '{self.collection_name}' exists. Loading...")
            return Collection(self.collection_name)
        else:
            logger.info(f"Collection '{self.collection_name}' not found. Creating...")
            fields = [
                FieldSchema(name="track_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
                FieldSchema(name="store_id", dtype=DataType.INT64),
                FieldSchema(name="camera_id", dtype=DataType.INT64),
                FieldSchema(name="timestamp", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields, description="Person ReID Embeddings")
            collection = Collection(name=self.collection_name, schema=schema)
            collection.create_index(
                field_name="embedding",
                index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
            )
            return collection

    def insert_embedding(
        self,
        track_id: int,
        embedding: List[float],
        camera_id: int,
        timestamp: int
    ):
        data = [[track_id], [embedding], [self.store_id], [camera_id], [timestamp]]
        logger.info(f"Inserting embedding for track_id={track_id}, store_id={self.store_id}")
        self.collection.insert(data)
        self.collection.flush()

    def search_embedding(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        if isinstance(query_embedding, list):
            query_embedding = np.array([query_embedding])
        elif isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.reshape(1, -1)

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        expr = f"store_id == {self.store_id}"

        logger.info(f"Searching for top {top_k} similar embeddings in store {self.store_id}...")
        results = self.collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["track_id", "camera_id", "timestamp"]
        )

        matches = []
        for hit in results[0]:
            matches.append((hit.entity.get("track_id"), hit.distance))
        return matches
