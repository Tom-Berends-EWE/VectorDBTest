__all__ = ['PineconeConnection']

from typing import Generator

import numpy as np
from pinecone import Pinecone

from connection import Connection
from res import resource_path, load_connection_config
from util import convert_raw_embedding_to_vec

_CONFIG = load_connection_config(resource_path('.env.pinecone'))

_EMBEDDING_VEC_DIMS = int(_CONFIG['EMBEDDING_VEC_DIMS'])

_API_KEY = _CONFIG['API_KEY']
_INDEX_NAME = _CONFIG['INDEX_NAME']


def _scrape_query_response(query_response) -> Generator[tuple[str, int], None, None]:
    for matching_qualifier in query_response['matches']:
        qualifier = matching_qualifier['metadata']['qualifier']
        score = 1 - matching_qualifier['score']

        yield qualifier, score


class PineconeConnection(Connection):
    def __init__(self):
        super().__init__()
        self._index = None

    def _create_connection(self):
        return Pinecone(_API_KEY)

    def setup(self):
        self._index = self._connection.Index(_INDEX_NAME)

    def upload_qualifier_data(self, qualifier_data: dict[str, str]) -> None:
        self._index.upsert(
            vectors=[
                {
                    'id': str(hash(qualifier_name)),
                    'values': convert_raw_embedding_to_vec(raw_qualifier_embedding_vec),
                    'metadata': {'qualifier': qualifier_name}
                } for qualifier_name, raw_qualifier_embedding_vec in qualifier_data.items()
            ]
        )

    def retrieve_matches(self, query_embedding_vec: np.array, k: int) -> list[tuple[str, float]]:
        query_response = self._index.query(
            vector=query_embedding_vec.tolist(),
            top_k=k,
            include_metadata=True
        )
        return list(_scrape_query_response(query_response))

    def close(self, *args):
        if self._index:
            self._index.__exit__(*args)
