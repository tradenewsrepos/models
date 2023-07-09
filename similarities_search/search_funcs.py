from typing import Tuple, List

import faiss
import numpy as np


def similarities_search(
    db_ids: np.array,
    db_vectors: np.array,
    query: np.array,
    k_nearest: int = 3,
) -> Tuple[List, List]:
    """

    :param db_ids:
    :param db_vectors: - вектор размерностью shape (num_texts, 768)
    :param query: - вектор размерностью shape (num_texts, 768)
    :param k_nearest: сколько матчей нужно передать
    :return:
    """
    dim = db_vectors.shape[1]

    assert dim == query.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)

    index.add_with_ids(db_vectors, db_ids)

    similarities, similarities_ids = index.search(query, k=k_nearest)
    similarities = np.around(np.clip(similarities, 0, 1), decimals=4)
    return similarities_ids[0].tolist(), similarities[0].tolist()
