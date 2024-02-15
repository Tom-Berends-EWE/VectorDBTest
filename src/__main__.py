import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from connections import Connection
from res import *
from util import convert_raw_embedding_to_vec

load_env()


def postprocess_embedding(embedding_vec, mu, U):
    embedding_vec_tilde = embedding_vec - mu
    embedding_vec_projection = np.zeros(len(embedding_vec))

    for component in U:
        embedding_vec_jj = np.dot(component, embedding_vec)
        embedding_vec_projection += embedding_vec_jj * component

    embedding_vec_prime = embedding_vec_tilde - embedding_vec_projection
    embedding_vec_prime = embedding_vec_prime / np.linalg.norm(embedding_vec_prime)
    return embedding_vec_prime


def load_qualifier_data():
    qualifiers_df = ISOTROPIC_QUALIFIERS.load()
    return {intent: embedding for intent, embedding in zip(qualifiers_df['Intent'], qualifiers_df['Embedding'])}


def postprocess_qualifier_embeddings():
    qualifiers_df = QUALIFIERS.load()
    qualifier_embeddings = qualifiers_df['Embedding']
    qualifier_embedding_vecs = [convert_raw_embedding_to_vec(qualifier_embedding) for qualifier_embedding in
                                qualifier_embeddings]

    qualifier_embedding_vecs = np.array(qualifier_embedding_vecs)

    mu = np.mean(qualifier_embedding_vecs, axis=0)

    qualifier_embedding_vecs_tilde = qualifier_embedding_vecs - mu

    pca = PCA()
    pca.fit(qualifier_embedding_vecs_tilde)

    D = 15

    qualifier_embeddings_new = list()
    N = len(qualifier_embedding_vecs)
    U = pca.components_
    U = U[D:, :]
    for n in range(N):
        qualifier_embedding_vec = qualifier_embedding_vecs[n]
        qualifier_embedding_vec_prime = postprocess_embedding(qualifier_embedding_vec, mu, U)
        qualifier_embedding_vec_prime_r = repr(qualifier_embedding_vec_prime).removeprefix('array(').removesuffix(')')
        qualifier_embeddings_new.append(qualifier_embedding_vec_prime_r)

    qualifiers_df['Embedding'] = pd.Series(qualifier_embeddings_new)
    ISOTROPIC_QUALIFIERS.save(qualifiers_df, index=False)
    PCA_RESULTS.save(mu=mu, U=U)


def main():
    postprocess_qualifier_embeddings()
    with Connection() as connection:
        connection.setup()

        qualifier_data = load_qualifier_data()
        connection.upload_qualifier_data(qualifier_data)

        customer_inputs_df = CUSTOMER_INPUTS.load()
        pca_results = PCA_RESULTS.load()
        mu, U = pca_results['mu'], pca_results['U']
        for user_input, embedding in zip(customer_inputs_df['userInput'], customer_inputs_df['Embedding']):
            embedding_vec = convert_raw_embedding_to_vec(embedding)
            embedding_vec = postprocess_embedding(embedding_vec, mu, U)

            best_matching_qualifiers = connection.retrieve_matches(embedding_vec, int(os.environ['K']))
            best_matching_qualifier, cos_distance = best_matching_qualifiers[0]

            low = 0.3
            high = 1.2
            cos_distance = (cos_distance - low) * 2 / (high - low)
            cos_similarity = 1 - cos_distance

            confidence = (cos_similarity + 1) / 2 * 100
            print(f'{user_input} -> {best_matching_qualifier} - {confidence}% # {best_matching_qualifiers}')


if __name__ == '__main__':
    main()
