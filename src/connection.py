__all__ = ['Connection']

from abc import ABC, abstractmethod

import numpy as np


class Connection(ABC):
    __slots__ = ['_connection']

    @abstractmethod
    def _create_connection(self):
        pass

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def upload_qualifier_data(self, qualifier_data: dict[str, str]) -> None:
        pass

    @abstractmethod
    def retrieve_matches(self, query_embedding_vec: np.array, k: int) -> list[tuple[str, float]]:
        pass

    def close(self, *args):
        pass

    def __enter__(self):
        self._connection = self._create_connection()
        if hasattr(self._connection, '__enter__'):
            self._connection.__enter__()
        return self

    def __exit__(self, *args):
        if hasattr(self._connection, '__exit__'):
            return self._connection.__exit__(*args)
        elif hasattr(self._connection, 'close'):
            self._connection.close()
        else:
            # INFO LOG
            self.close(*args)
