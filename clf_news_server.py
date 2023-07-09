from transformers import pipeline, AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F


class Inferer:
    def __init__(self, task: str, path: str, model="bert-base-multilingual-cased"):
        self.inferer = pipeline(
            task,
            model=path,
            tokenizer=path,
            function_to_apply="sigmoid",
            return_all_scores=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path, )
        self.model = AutoModel.from_pretrained(path)


    def infer(self, text):
        result = self.inferer(text, truncation=True)[0]

        result = [
            {"label": "COVID-19", "score": 0.0},
            {"label": "международные отношения", "score": 0},
            {"label": "Россия", "score": 0.0},
            {"label": "Социологические опросы", "score": 0.0},
            {"label": "аналитика", "score": 0.0},
            {"label": "военная тематика", "score": 0.0},
            {"label": "меры поддержки", "score": 0.0},
            {"label": "мнения", "score": 0.0},
            {"label": "политика", "score": 0.0},
            {"label": "не по теме", "score": result[0]["score"]},
            {"label": "другие отношения", "score": result[1]["score"]},
            {"label": "торговля", "score": result[2]["score"]},
            {"label": "проекты", "score": result[3]["score"]},
            {"label": "санкции", "score": result[4]["score"]},
            {"label": "инвестиция", "score": result[5]["score"]},
        ]

        return result


    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed_bert_pool(self, text):
        """
        Вектор текста рассчитывается аналогично расчету вектора в библиотеке sentence-transformers
        В результирующем векторе учитываются все векторы скрытого слоя токенов предложения.
        """
        # Tokenize sentences
        encoded_input = self.tokenizer(
            text, max_length=500, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
