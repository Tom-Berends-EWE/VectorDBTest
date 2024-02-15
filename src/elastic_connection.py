__all__ = ['ElasticConnection']

from typing import Sequence, Any, Generator

import numpy as np
import urllib3
from elastic_transport import SecurityWarning, ObjectApiResponse
from elasticsearch import Elasticsearch

from connection import Connection
from res import resource_path, load_connection_config
from util import convert_raw_embedding_to_vec

_CONFIG = load_connection_config(resource_path('.env.elastic'))

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(SecurityWarning)

_EMBEDDING_VEC_DIMS = _CONFIG['EMBEDDING_VEC_DIMS']
_INDEX_NAME = _CONFIG['INDEX_NAME']

_ENDPOINT = _CONFIG['ENDPOINT']
_API_KEY = _CONFIG['API_KEY']


def _setup_index_cmd_body():
    return {
        'mappings': {
            'properties': {
                'qualifier-vector': {
                    'type': 'dense_vector',
                    'dims': _EMBEDDING_VEC_DIMS,
                    'index': True,
                    'similarity': 'cosine'
                }
            },
            'qualifier': {
                'type': 'text'
            }
        }
    }


def _post_cmd_body(qualifiers: dict[str, str]) -> Sequence[dict[str, Any]]:
    post_cmd_mappings = list()

    for idx, (intent, embedding) in enumerate(qualifiers.items()):
        post_cmd_mappings.append(
            {
                'index': {
                    '_id': str(idx + 1)
                }
            }
        )
        post_cmd_mappings.append(
            {
                'qualifier': intent,
                'qualifier-vector': convert_raw_embedding_to_vec(embedding),
            }
        )

    return post_cmd_mappings


def _search_cmd_body(query_embedding_vec, k) -> dict[str, Any]:
    return {
        'knn': {
            'field': 'qualifier-vector',
            'query_vector': query_embedding_vec,
            'k': k,
            'num_candidates': 100
        },
        'fields': ['qualifier'],
    }


def _scrape_search_request_response(response: ObjectApiResponse) -> Generator[tuple[str, int], None, None]:
    for matching_qualifier in response['hits']['hits']:
        intent = matching_qualifier['fields']['qualifier']
        score = matching_qualifier['_score']

        yield intent, float(score)


class ElasticConnection(Connection):
    def _create_connection(self) -> Elasticsearch:
        return Elasticsearch(_ENDPOINT, api_key=_API_KEY, verify_certs=False)

    def _perform_search_request(self, query_embedding_vec: np.array, k: int) -> ObjectApiResponse:
        return self._connection.search(index=_INDEX_NAME, body=_search_cmd_body(query_embedding_vec, k))

    def setup(self):
        self._connection.index(index=_INDEX_NAME, body=_setup_index_cmd_body())

    def upload_qualifier_data(self, qualifier_data: dict[str, str]) -> None:
        self._connection.bulk(index=_INDEX_NAME, refresh=True, body=_post_cmd_body(qualifier_data))

    def retrieve_matches(self, query_embedding_vec: np.array, k: int) -> list[tuple[str, float]]:
        search_request_response = self._perform_search_request(query_embedding_vec, k)
        return list(_scrape_search_request_response(search_request_response))
