import redis
# from redis.commands.search.field import TagField, VectorField
# from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
import requests
from utils import dotdict
import dspy
from typing import List, Optional
from config.vectordb import REDIS_INDEX_NAME, HUGGINGFACE_EMBEDDING_MODEL_URL, HUGGINGFACE_EMBEDDING_API_HEADERS


class DSPythonicRMClient(dspy.Retrieve):
    def __init__(self, url: str, port:int = None, k:int = 3):
        super().__init__(k=k)

        self.url = f"{url}:{port}"
        self.r = self._connect_to_redis()

    def _connect_to_redis(self):
        """Connect to Redis instance."""
        r = redis.from_url(self.url)
        return r
    
    def _query_embedding_api(self, payload):
        """Query Hugging Face Embedding API."""
        response = requests.post(HUGGINGFACE_EMBEDDING_MODEL_URL, headers={"Authorization": "Bearer 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"}, json=payload)
        return response.json()

    def _get_input_embedding(self, input):
        """Get input embedding from Hugging Face Embedding API and return as bytes for Redis search."""
        output = self._query_embedding_api({
            "inputs": input,
        })
        return np.array(output, dtype=np.float32).tobytes()
    
    def _format_output(self, k_docs_output):
        """Format output from Redis search for formats accepted by dspy"""
        top_k_docs = [dotdict({'long_text': document_class.content}) for document_class in k_docs_output]
        return top_k_docs
    
    def query_redis(self, query:str, k:int):
        """Query Redis instance."""
        redis_query = (
            Query(f"*=>[KNN {k} @content_vector $vec as score]")
            .sort_by("score")
            .return_fields("source", "content")
            .paging(0, k)
            .dialect(k)
        )
        query_params = {
            "vec": self._get_input_embedding(query)
        }
        k_docs = self.r.ft(REDIS_INDEX_NAME).search(redis_query, query_params).docs
        return self._format_output(k_docs)

    def forward(self, query_or_queries:str, k:Optional[int]) -> dspy.Prediction:
        """Forward pass for the Retrieve module."""
        response = self.query_redis(query_or_queries, k)
        return response 
    
