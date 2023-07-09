from razdel import tokenize
from transformers import pipeline


class Inferer:
    def __init__(
        self, task: str, path: str, model="bert-base-multilingual-cased"
    ):

        self.inferer = pipeline(
            task,
            model=path,
            tokenizer=path,
            ignore_labels=['O'], 
            aggregation_strategy="first"
        )

    def infer(self, text):
        print(text)
        print(self.inferer(text))

        return self.inferer(text)

    def infer_old(self, text):
        preds = self.inferer(text)
        for p in preds:
            p["score"] = [p["score"]]
        joined_preds = []
        for p in preds:
            if not joined_preds:
                joined_preds.append(p)
            elif p["word"].startswith("##") or p["entity"].startswith("I-"):
                joined_preds[-1]["end"] = p["end"]
                joined_preds[-1]["score"].append(p["score"][0])
            else:
                joined_preds.append(p)
        tokens = list(tokenize(text))
        token_i = 0
        for pred in joined_preds:
            for i in range(token_i, len(tokens)):
                if tokens[i].start >= pred["start"]:
                    print(tokens[i], pred["start"])
                    if pred["end"] < tokens[i].stop:
                        pred["end"] = tokens[i].stop
                    token_i = i
                    break

        for p in joined_preds:
            p["word"] = text[p["start"] : p["end"]]
            p["entity"] = p["entity"][2:]
            p["score"] = sum(p["score"]) / len(p["score"])
            p["start"] = int(p["start"])
            p["end"] = int(p["end"])
        return joined_preds
