import os.path

import fasttext


class LangInferer:
    def __init__(self, path: str, model: str):
        self.path = os.path.join(path, model)
        self.inferer = fasttext.load_model(self.path)

    def infer(self, text: str) -> str:
        pred = self.inferer.predict(text)
        pred_lang = pred[0][0].split("__")[-1]
        return {"language": pred_lang}
