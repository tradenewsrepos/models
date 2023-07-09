import pymorphy2
import re

morph = pymorphy2.MorphAnalyzer()


def pymorphy2_lemmatize(text):
    text = re.sub('[.,:«»;%©?*,!@#$%^&()\t\n]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)  # deleting symbols
    text = " ".join(morph.parse(word)[0].normal_form for word in text.split())
    return text


class LemmaTokenizer:
    def __init__(self):
        self.lem = pymorphy2_lemmatize

    def __call__(self, doc):
        return [self.lem(t) for t in doc.split()]