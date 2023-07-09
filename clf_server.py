import os
import cloudpickle
import re
import pymorphy2
from typing import Dict, AnyStr


morph = pymorphy2.MorphAnalyzer()


def pymorphy2_lemmatize(text):
    text = re.sub(
        '[.,:«»;%©?*,!@#$%^&()\t\n]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', " ", text
    )  # deleting symbols
    text = " ".join(morph.parse(word)[0].normal_form for word in text.split())
    return text


class LemmaTokenizer:
    def __init__(self):
        self.lem = pymorphy2_lemmatize

    def __call__(self, doc):
        return [self.lem(t) for t in doc.split()]


class TokenInferer:
    # TODO добавить условия для выбора класса ГС
    """
    Класс работает с моделью, обученной предсказывать класс товара по СМТК
    Например,
    text = ['нефть']
    ответ:
    array(['33 - Нефть, нефтепродукты и аналогичные материалы'], dtype='<U212')
    """

    def __init__(
        self,
        path: str,
        model: str,
    ):
        model = os.path.join(path, model)
        with open(model, "rb") as file:
            self.inferer = cloudpickle.load(file)
        self.hs_keywords = [
            'золото монетарное',
            'монетарное',
            'золотая монета',
            'золотой резерв',
        ]


    def infer(self, text: AnyStr) -> Dict:
        """
        Правило определяет есть ли в товаре одно из слов в списке. Если есть то товар
        относится к классу 'Товары ГС07, выходящие за рамки охвата СМТК'.
        Такой подход увеличивает False Positive для этого класса. Другого подхода пока нет, т.к.
        мало данных для этого класса
        """
        contains_hs_prods = [w for w in self.hs_keywords if w in text]
        if contains_hs_prods:
            prediction = 'Товары ГС07, выходящие за рамки охвата СМТК'
        else:
            prediction = self.inferer.predict([text])[0]
        return {
                "text": text,
                "class": prediction,
            }


class TextInferer:
    """
    Класс работает с моделью, обученной предсказывать класс
    "93 - Специальные операции и товары, не классифицированные по типу" по СМТК
    На вход модели подается текст новости. На выходе:
    1, если новость относится к разделу
     "93 - Специальные операции и товары, не классифицированные по типу"
    0 - если новость не относится к этому классу.

    """

    def __init__(
        self,
        path: AnyStr,
        model: AnyStr,
    ):
        model = os.path.join(path, model)
        with open(model, "rb") as file:
            self.inferer = cloudpickle.load(file)

    def infer(self, text: AnyStr) -> Dict:
        prediction = self.inferer.predict([text])[0]
        print("prediction: ", prediction)
        return {
            "class": prediction,
        }
