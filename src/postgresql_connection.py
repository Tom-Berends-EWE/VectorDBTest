__all__ = ['LocalPostgreSQLConnection', 'RemotePostgreSQLConnection']

from abc import ABC
from typing import Any

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from connection import Connection
from res import resource_path, load_connection_config

_LOCAL_CONFIG = load_connection_config(resource_path('.env.postgresql.local'))
_REMOTE_CONFIG = load_connection_config(resource_path('.env.postgresql.remote'))
_CONFIG = dict(_LOCAL_CONFIG.items() & _REMOTE_CONFIG.items())

_EMBEDDING_VEC_DIMS = _CONFIG['EMBEDDING_VEC_DIMS']


class PostgreSQLConnection(Connection, ABC):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self._config = config

    def _create_connection(self):
        return psycopg2.connect(self._config['CONNECTION_STRING'], password=self._config['PASSWORD'])

    def _execute(self, query_func, *args, **kwargs):
        with self._connection.cursor() as cursor:
            result = query_func(cursor, *args, **kwargs)

        return result

    def _setup_vector_extension(self, cursor):
        cursor.execute('CREATE EXTENSION IF NOT EXISTS vector;')
        if self._config['PG_SIMILARITY_SUPPORT']:
            cursor.execute('CREATE EXTENSION IF NOT EXISTS pg_similarity;')
        register_vector(self._connection)

    def _setup_qualifier_table(self, cursor):
        cursor.execute('DROP TABLE IF EXISTS qualifiers;'
                       'CREATE TABLE IF NOT EXISTS qualifiers (id bigserial primary key,'
                       'intent text,'
                       f'embedding vector({_EMBEDDING_VEC_DIMS}));')

    def _retrieve_matches(self, cursor, query_embedding_vec, k):
        cursor.execute(
            f'SELECT intent, embedding <=> %s as similarity_score FROM qualifiers ORDER BY embedding <=> %s LIMIT {k}',
            (query_embedding_vec, query_embedding_vec))
        return cursor.fetchall()

    def _upload_qualifier_data(self, cursor, qualifier_data):
        qualifier_data = zip(qualifier_data.keys(), qualifier_data.values())
        execute_values(cursor, 'INSERT INTO qualifiers (intent, embedding) VALUES %s', qualifier_data)

    def setup(self) -> None:
        self._execute(self._setup_vector_extension)
        self._execute(self._setup_qualifier_table)

    def upload_qualifier_data(self, qualifier_data: dict[str, str]) -> None:
        self._execute(self._upload_qualifier_data, qualifier_data)

    def retrieve_matches(self, query_embedding_vec: np.array, k: int) -> list[tuple[str, float]]:
        return self._execute(self._retrieve_matches, query_embedding_vec, k)


class LocalPostgreSQLConnection(PostgreSQLConnection):
    def __init__(self):
        super().__init__(_LOCAL_CONFIG)


class RemotePostgreSQLConnection(PostgreSQLConnection):
    def __init__(self):
        super().__init__(_REMOTE_CONFIG)
