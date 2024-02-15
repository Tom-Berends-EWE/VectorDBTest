__all__ = ['Connection', 'LocalPostgreSQLConnection', 'RemotePostgreSQLConnection', 'ElasticConnection',
           'PineconeConnection']

import os
import sys

from postgresql_connection import LocalPostgreSQLConnection, RemotePostgreSQLConnection
from elastic_connection import ElasticConnection
from pinecone_connection import PineconeConnection


def _load_connection():
    if len(sys.argv) > 1:
        class_name = sys.argv[1]
    elif 'CONNECTION' in os.environ:
        class_name = os.environ['CONNECTION']
    else:
        raise RuntimeError('Missing connection name ( command parameter or environment variable )')

    try:
        return globals()[class_name]
    except LookupError:
        raise RuntimeError(f'No connection named: {class_name}')


Connection = _load_connection()
