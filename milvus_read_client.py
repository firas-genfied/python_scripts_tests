import os
from dotenv import load_dotenv
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np
import logging
from typing import List, Tuple, Union
from config import get_milvus_host_port  # Import the helper
from collections import defaultdict

# Load .env variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("milvus-client")

class MilvusReIDClient:
    def __init__(self, store_id: int,collection_name: str = "person_embeddings"):
        self.collection_name = collection_name

        host, port = get_milvus_host_port(store_id)
        logger.info(f"[store_id={store_id}] Connecting to Milvus at {host}:{port}...")
        connections.connect("default", host=host, port=port)
        logger.info("Connected to Milvus.")

        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        if self.collection_name in utility.list_collections():
            logger.info(f"Collection '{self.collection_name}' exists. Loading...")
            collection = Collection(self.collection_name)
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

        logger.info(f"Loading collection '{self.collection_name}' into memory...")
        collection.load()
        return collection


    def insert_embedding(
        self,
        track_id: int,
        embedding: List[float],
        store_id: int,
        camera_id: int,
        timestamp: int
    ):
        data = [[track_id], [embedding], [store_id], [camera_id], [timestamp]]
        logger.info(f"Inserting embedding for track_id={track_id}, store_id={store_id}")
        self.collection.insert(data)
        self.collection.flush()

    def search_embedding(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 5,
        store_filter: int = None
    ) -> List[Tuple[int, float]]:
        if isinstance(query_embedding, list):
            query_embedding = np.array([query_embedding])
        elif isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.reshape(1, -1)

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        expr = f"store_id == {store_filter}" if store_filter is not None else ""

        logger.info(f"Searching for top {top_k} similar embeddings...")
        results = self.collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["track_id", "store_id", "camera_id", "timestamp"]
        )

        matches = []
        for hit in results[0]:
            matches.append((hit.entity.get("track_id"), hit.distance))

        return matches
    
    def get_all_track_features(self, store_id) -> dict[int, List[np.ndarray]]:
        """
        Returns a dictionary where keys are track_ids and values are lists of embeddings (np.ndarrays),
        filtered by the current store_id.
        """
        expr = f"store_id == {store_id}"
        logger.info(f"Querying all embeddings for store_id={store_id} from collection '{self.collection_name}'...")

        try:
            results = self.collection.query(
                expr=expr,
                output_fields=["track_id", "embedding"],
                limit=100_000  # adjust if needed
            )
        except Exception as e:
            logger.error(f"Failed to query embeddings: {e}")
            return {}

        feature_map = defaultdict(list)
        for row in results:
            track_id = row["track_id"]
            embedding = np.array(row["embedding"], dtype=np.float32)
            feature_map[track_id].append(embedding)

        return dict(feature_map)
    
    def get_all_track_ids(self, store_id: int) -> List[int]:
        """
        Fetch all unique track_ids in the collection for a given store_id.
        Assumes the collection has a 'track_id' and 'store_id' field.
        """
        try:
            collection = Collection(self.collection_name)

            # Ensure index is loaded
            collection.load()

            expr = f"store_id == {store_id}"
            output_fields = ["track_id"]
            results = collection.query(expr, output_fields=output_fields)

            # Extract unique track_ids
            track_ids = list({r["track_id"] for r in results})
            return track_ids
        except Exception as e:
            logger.error(f"Error fetching track_ids for store_id {store_id} from Milvus: {e}")
            return []
    
    def delete_track(self, track_id: int, store_id: int):
        """
        Delete all embeddings associated with a given track_id and store_id from the collection.
        """
        expr = f"track_id == {track_id} and store_id == {store_id}"
        logger.info(f"Deleting records for track_id={track_id}, store_id={store_id} from collection '{self.collection_name}'...")
        try:
            self.collection.delete(expr)
            self.collection.flush()
            logger.info(f"Successfully deleted records for track_id={track_id}, store_id={store_id}")
        except Exception as e:
            logger.error(f"Failed to delete track_id={track_id}, store_id={store_id}: {e}")

