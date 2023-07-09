import pickle
from typing import Dict, AnyStr, Any, Tuple

import numpy as np
from sqlalchemy import select
import datetime
import pytz

from .models import (
    TradeNewsEmbeddings,
    TradeNewsRelevant,
    TradeNewsForApproval,
)


def get_relevant_data(session) -> Dict[AnyStr, Any]:
    query = select(
        TradeNewsRelevant.id,
        TradeNewsRelevant.title,
        TradeNewsRelevant.article_ids,
    )
    result = session.execute(query)
    result = result.fetchall()

    return result


def get_approval_data(session) -> Dict[AnyStr, Any]:
    query = select(
        TradeNewsForApproval.id,
        TradeNewsForApproval.title,
        TradeNewsForApproval.article_ids,
    )
    result = session.execute(query)
    result = result.fetchall()
    return result


# def get_embeddings(session, excluded_id) -> Tuple[Dict, np.array]:
#     """
#     Метод берет из базы вектора, меняет их тип float16 -> float32,
#     т.к. faiss не работает с fp16
#     :param session:
#     :param excluded_id - id вектора, дубликат которого мы хотим найти, исключаем,
#     чтобы не рассматривать тот же текст
#     :return: id векторов в базе и соответствующие векторы
#     """
#     query = select(
#         TradeNewsEmbeddings.id,
#         TradeNewsEmbeddings.embedding,
#     ).where(TradeNewsEmbeddings.id != excluded_id)
#     result = session.execute(query)
#     result = result.fetchall()
#     ids = {i: row[0] for i, row in enumerate(result)}
#     vectors = np.stack([pickle.loads(row[1]) for row in result])
#     vectors = vectors.astype(np.float32)
#     return ids, vectors

def get_filter_date():
    return datetime.datetime.now(pytz.utc) - datetime.timedelta(days=30)

def get_embeddings(session, excluded_i) -> Tuple[Dict, np.array]:
    """
    Метод берет из базы вектора, меняет их тип float16 -> float32,
    т.к. faiss не работает с fp16
    :param session:
    :param excluded_id - id вектора, дубликат которого мы хотим найти, исключаем,
    чтобы не рассматривать тот же текст
    :return: id векторов в базе и соответствующие векторы
    """
    query = select(
        TradeNewsEmbeddings.id,
        TradeNewsEmbeddings.embedding
    ).where(TradeNewsEmbeddings.date_added > get_filter_date())
    result = session.execute(query)
    result = result.fetchall()
    ids = {i: row[0] for i, row in enumerate(result)}
    vectors = np.stack([pickle.loads(row[1]) for row in result])
    vectors = vectors.astype(np.float32)
    return ids, vectors

def insert_embedding(
    session,
    uuid,
    article_ids: str,
    vector: np.array,
    model: str,
    date_time,
):
    vector_byte = pickle.dumps(vector)
    embedding = TradeNewsEmbeddings(
        id=uuid,
        article_id=article_ids,
        embedding=vector_byte,
        model=model,
        date_added=date_time,
    )
    session.add(embedding)
