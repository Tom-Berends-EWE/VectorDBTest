__all__ = ['convert_raw_embedding_to_vec']

import numpy as np


def convert_raw_embedding_to_vec(raw_embedding: str) -> np.array:
    return np.fromstring(raw_embedding.removeprefix('[').removesuffix(']'), sep=',')
