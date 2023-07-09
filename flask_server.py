import os
import subprocess
from typing import Dict
from fastapi import FastAPI
from pydantic import BaseModel
import time
import numpy as np
from clf_news_server import Inferer as ClfNewsInferer
from clf_server import (
    TokenInferer as TokenClfInferer,
    TextInferer as TextClfInferer,
)
from er_server import Inferer as ErInferer
from ner_server import Inferer as NerInferer
from similarities_search.search_funcs import similarities_search
from similarities_search.db.config import Session
from similarities_search.db.data import (
    get_embeddings,
)
from clf_language import LangInferer


class Text(BaseModel):
    text: str


class TextSimiliar(BaseModel):
    text_id: str
    text: str


app = FastAPI()

if os.getenv("PATH_NER"):
    path_ner = os.getenv("PATH_NER")
else:
    path_ner = (
        r"C:\Users\Alex\Documents\Рабкота ранхигс\MULTI_NER_AND_ER"
        r"\OKPD2\sber_bert_large_089_ner_rured_plus_cased"
    )

if os.getenv("PATH_ER"):
    path_er = os.getenv("PATH_ER")
else:
    path_er = (
        r"C:\Users\Alex\Documents"
        r"\Рабкота ранхигс\MULTI_NER_AND_ER\OKPD2"
        r"\sberbank-ai_ruBert-large_cased_0956_ER_typed_entity_marker"
    )
if os.getenv("PATH_CLF_NEWS"):
    path_clf_news = os.getenv("PATH_CLF_NEWS")
else:
    path_clf_news = (
        r"C:\Users\Alex\Documents\Рабкота ранхигс\MULTI_NER_AND_ER"
        r"\OKPD2\news_rubert_072"
    )

if os.getenv("PATH_WORD_PROD_CLF"):
    path_clf = os.getenv("PATH_WORD_PROD_CLF")
    clf_model_name = os.listdir(path_clf)[0]
else:
    path_clf = "/clf_o/ru_products_clf_v2.pkl"

if os.getenv("PATH_TEXT_SPEC_PROD_CLF"):
    path_specprod = os.getenv("PATH_TEXT_SPEC_PROD_CLF")
    spec_prod_model_name = os.listdir(path_specprod)[0]

if os.getenv("PATH_LANG"):
    path_lang = os.getenv("PATH_LANG")
    lang_model_name = os.listdir(path_lang)[0]


clf_news = ClfNewsInferer(task="text-classification", path=path_clf_news)
ner = NerInferer(task="ner", path=path_ner)
er = ErInferer(path_ner=path_ner, path_er=path_er)
token_clf_prod = TokenClfInferer(path=path_clf, model=clf_model_name)
text_clf_prod = TextClfInferer(path=path_specprod, model=spec_prod_model_name)
lang_detection = LangInferer(path=path_lang, model=lang_model_name)

with open("./s3_models.txt", "r") as file:
    readed_file = file.read()
    models_titles = readed_file.split("\n")
models_titles = [model.replace(".zip", "") for model in models_titles]

models: Dict[str, dict] = {
    "ner": {"model": None, "get_data": None, "launch_function": ner.infer},
    "word_clf": {
        "model": None,
        "get_data": None,
        "launch_function": token_clf_prod.infer,
    },
    "clf_news": {
        "model": None,
        "get_data": None,
        "launch_function": clf_news.infer,
    },
    "relation_extraction": {
        "model": None,
        "get_data": None,
        "launch_function": er.infer,
    },
    "text_clf": {
        "model": None,
        "get_data": None,
        "launch_function": text_clf_prod.infer,
    },
    "feature_extractor": {
        "model": None,
        "get_data": None,
        "launch_function": clf_news.embed_bert_pool,
    },
    "lang_clf": {
        "model": None,
        "get_data": None,
        "launch_function": lang_detection.infer,
    },
}


# TODO добавить функцию для тестирования моделей на новых данных
@app.post("/infer/{app_name}")
def infer(app_name: str, text: Text):
    """[summary]
    requests.post(f"{url}/infer/ner", json={"text": "РАНХиГС"})
    requests.post(f"{url}/infer/clf", json={"text": "Мясные консервы"})
    requests.post(f"{url}/infer/relation_extraction",
    json={"text":
          "Деньги на приобретение топлива Киев получил от Всемирного банка"})
    requests.post(f"{url}/infer/clf_news", json={"text": "Новость про что-то"})

    Args:
        app_name (str): [description]
        text (Text): [description]

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """
    preds = None
    if app_name == "ner":
        preds = models[app_name]["launch_function"](text.text)
    elif app_name == "word_clf":
        preds = models[app_name]["launch_function"](text.text)
    elif app_name == "clf_news":
        preds = models[app_name]["launch_function"](text.text)
    elif app_name == "relation_extraction":
        preds = models[app_name]["launch_function"](text.text)
    elif app_name == "text_clf":
        preds = models[app_name]["launch_function"](text.text)
    elif app_name == "lang_clf":
        preds = models[app_name]["launch_function"](text.text)
    else:
        raise Exception("not supported")

    return preds


@app.post("/get_similiar")
def infer(text: TextSimiliar):
    """
    text.text_id - uuid новости, дубликат которой мы хотим найти.
    id нужен, чтобы не сравнивать новость с самой собой
    text.text - текст новости, дубликат которой мы хотим найти
    """
    start_time = time.monotonic()
    # TODO async function
    with Session() as session:
        id_uuids, db_vectors = get_embeddings(session, text.text_id)

    assert len(id_uuids) == len(db_vectors)

    ids = np.array([*id_uuids.keys()])
    # TODO async function
    query_embedding = models["feature_extractor"]["launch_function"](text.text)

    query_embedding = np.array(query_embedding, dtype=np.float32)
    similarities_ids, similarities = similarities_search(
        ids, db_vectors, query_embedding
    )
    similarities_uuid = [*map(id_uuids.get, similarities_ids)]
    end_time = time.monotonic()
    duration = end_time - start_time

    return {
        "duration": duration,
        "similarities_ids": similarities_uuid,
        "similarities": similarities,
    }


@app.post("/embedding")
def get_embedding(text: Text):
    """
    Возвращает вектор текста shape= и метаинформацию
    """
    model_name = models_titles[2]
    start_time = time.monotonic()
    embedding = models["feature_extractor"]["launch_function"](text.text)[0]
    embedding = np.array(embedding, dtype=np.float16).tolist()
    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"прошло времени: {duration}")

    return {
        "duration": duration,
        "text": text,
        "embedding": embedding,
        "model": model_name,
    }


@app.get("/models_names")
def get_models_names():
    """
    Возвращает словарь с названиями моделей из файла s3_models.
    """
    return {
        "ner": models_titles[0],
        "relation_extraction": models_titles[1],
        "clf_news": models_titles[2],
        "word_clf": models_titles[3],
        "text_clf": models_titles[4],
        "lang_clf": models_titles[5],
    }


@app.post("/train/{app_name}")
def train(app_name: str):
    process = subprocess.Popen(["git", "clone", "ner_model"])
    process = subprocess.Popen(["./train_ner.sh", "./brat", "ner_model"])
    process.wait()
    return "Trained"
