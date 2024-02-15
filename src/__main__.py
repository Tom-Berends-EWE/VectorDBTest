import os

from connections import Connection
from data_processing import *
from res import *
from util import convert_raw_embedding_to_vec

load_env()

K = int(os.environ['K'])


def load_qualifier_data():
    qualifiers_df = ISOTROPIC_QUALIFIERS.load()
    return {intent: embedding for intent, embedding in zip(qualifiers_df['Intent'], qualifiers_df['Embedding'])}


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

            best_matching_qualifiers = connection.retrieve_matches(embedding_vec, K)
            best_matching_qualifier, cos_distance = best_matching_qualifiers[0]

            confidence = estimate_confidence(cos_distance)
            print(f'{user_input} -> {best_matching_qualifier} - {confidence}% # {best_matching_qualifiers}')


if __name__ == '__main__':
    main()
