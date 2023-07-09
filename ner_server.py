from transformers import pipeline
from typing import List

class Inferer:
    def __init__(
        self, task: str, path: str, model="bert-base-multilingual-cased"
    ):

        self.inferer = pipeline(
            task,
            model=path,
            tokenizer=path,
            ignore_labels=['O'],
            aggregation_strategy="first",
        )

    def infer(self, text):
        result: List = self.inferer(text)
        # плохие манеры но fastapi не может в numpy

        new_list = []
        for i in result:
            new_list.append(
                {
                    "entity_group": i["entity_group"],
                    "score": i["score"].item(),
                    "word": i["word"],
                    "start": int(i["start"]),
                    "end": int(i["end"]),
                }
            )
        return new_list

