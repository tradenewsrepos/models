import argparse
from threading import Lock

from transformers import AutoTokenizer
from transformers import pipeline

from utils import to_batches_by_razdel

MUTEX = Lock()
subj_ners = [
    "ORGANIZATION",
    "PERSON",
    "GROUP",
    "FAMILY",
    "GPE",
    "COUNTRY",
    "CITY",
    "REGION",
    "DATE",
    "PROFESSION",
    "PRODUCT",
    "WEBSITE",
    "FAC",
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "PERCENT",
    "MONEY",
    "PRICE",
    "EVENT",
    "CURRENCY",
    "LAW",
    "LOCATION",
    "INVESTMENT_PROJECT",
    "TRADE_AGREEMENT",
    "ECONOMIC_SECTOR",
]

extra_obj_ners = [
    "PRODUCT",
    "PRICE",
    "EVENT",
    "PROFESSION",
    "COUNTRY",
    "ORGANIZATION",
    "GPE",
    "LOCATION",
    "CITY",
    "LAW",
    "PERSON",
    "DATE",
    "GROUP",
    "FAC",
    "REGION",
    "CARDINAL",
    "FAMILY",
    "PERCENT",
    "MONEY",
]

misc_ners = [
    "B-EVENT",
    "B-ORDINAL",
    "B-QUANTITY",
    "B-MONEY",
    "B-PERCENT",
    "B-LANGUAGE",
    "B-LAW",
    "B-FAC",
]

all_ners = subj_ners + extra_obj_ners + misc_ners

subj_ners = [
    "ORGANIZATION",
    "PERSON",
    "GROUP",
    "FAMILY",
    "GPE",
    "COUNTRY",
    "CITY",
    "REGION",
    "DATE",
    "PROFESSION",
    "PRODUCT",
    "WEBSITE",
    "FAC",
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "PERCENT",
    "MONEY",
    "PRICE",
    "EVENT",
    "CURRENCY",
    "LAW",
    "LOCATION",
    "INVESTMENT_PROJECT",
    "TRADE_AGREEMENT",
    "ECONOMIC_SECTOR",
]
all_ners = [
    "ORGANIZATION",
    "PERSON",
    "GROUP",
    "FAMILY",
    "GPE",
    "COUNTRY",
    "CITY",
    "REGION",
    "DATE",
    "PROFESSION",
    "PRODUCT",
    "WEBSITE",
    "FAC",
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "PERCENT",
    "MONEY",
    "PRICE",
    "EVENT",
    "CURRENCY",
    "LAW",
    "LOCATION",
    "INVESTMENT_PROJECT",
    "TRADE_AGREEMENT",
    "ECONOMIC_SECTOR",
]

RULES_ER = {
    "SELLS_TO": {"subj": ["PRODUCT", "ORGANIZATION"], "obj": []},
    "WORKS_AS": {"subj": ["PERSON"], "obj": []},
    "WORKPLACE": {"subj": ["PERSON"], "obj": []},
    "COSTS": {
        "subj": ["PRODUCT"],
        "obj": ["QUANTITY", "ORDINAL", "CARDINAL", "PERCENT", "PRICE"],
    },
    "OWNERSHIP": {
        "subj": ["COUNTRY", "ORGANIZATION", "PERSON", "REGION"],
        "obj": ["CITY", "FACILITY", "ORGANIZATION"],
    },
    "DATE_TAKES_PLACE_ON": {"subj": ["EVENT"], "obj": ["DATE"]},
    "TAKES_PLACE_IN": {"subj": ["EVENT"], "obj": ["COUNTRY", "CITY"]},
    "RESULT": {"subj": ["EVENT"], "obj": ["LAW"]},
    "ORGANIZES": {
        "subj": ["ORGANIZATION", "COUNTRY", "CITY", "GPE", "GROUP"],
        "obj": ["EVENT"],
    },
    "EVENT_TAKES_PART_IN": {
        "subj": ["ORGANIZATION", "COUNTRY", "CITY", "GPE", "PERSON", "GROUP"],
        "obj": ["EVENT"],
    },
    "SIGNED_ON": {"subj": ["LAW", "TRADE_AGREEMENT"], "obj": ["DATE"]},
    "SIGNED_BY": {
        "subj": ["LAW", "TRADE_AGREEMENT"],
        "obj": ["COUNTRY", "ORGANIZATION", "GPE", "PERSON", "GROUP"],
    },
    "SUBJECT": {
        "subj": ["LAW", "TRADE_AGREEMENT", "INVESTMENT_PROJECT"],
        "obj": ["PRODUCT", "INVESTMENT_PROJECT", "FAC", "ORGANIZATION"],
    },
    "SUBEVENT_OF": {"subj": ["EVENT"], "obj": ["EVENT"]},
}


# позволяет склеивать сломаные сущности
def second_agg_v2(
    tags,
    mass=[
        "QUANTITY",
        "DATE",
        "ORDINAL",
        "CARDINAL",
        "PERCENT",
        "PRICE",
        "GPE",
        "ORGANIZATION",
        "FAC",
    ],
):
    new_ans = list()
    end_token = list()
    for i, tag in enumerate(tags):
        if len(end_token) == 0:
            end_token = tag.copy()
        else:
            if (
                (end_token["end"] == tag["start"])
                and (end_token["entity_group"] == tag["entity_group"])
                and (tag["entity_group"] in mass)
            ):
                end_token["word"] += tag["word"]
                end_token["end"] = tag["end"]
            elif (
                (end_token["end"] + 1 == tag["start"])
                and (end_token["entity_group"] == tag["entity_group"])
                and (tag["entity_group"] in mass)
            ):
                end_token["word"] += " " + tag["word"]
                end_token["end"] = tag["end"]
            else:
                new_ans.append(end_token)
                end_token = tag.copy()

    if len(new_ans) < 1:
        return new_ans

    if new_ans[-1] != end_token:
        new_ans.append(end_token)
    return new_ans


class Processor:
    def __init__(self, input_format, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.new_tokens = []
        self.input_format = input_format
        if self.input_format == "entity_marker":
            self.new_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
        self.tokenizer.add_tokens(self.new_tokens)
        if self.input_format not in (
            "REBEL",
            "entity_mask",
            "entity_marker",
            "entity_marker_punct",
            "typed_entity_marker",
            "typed_entity_marker_punct",
        ):
            raise Exception("Invalid input format!")

    def tokenize(self, tokens, subj_type, obj_type, ss, se, os, oe):
        """
        Implement the following input formats:
            - entity_mask: [SUBJ-NER], [OBJ-NER].
            - entity_marker: [E1] subject [/E1], [E2] object [/E2].
            - entity_marker_punct: @ subject @, # object #.
            - typed_entity_marker: [SUBJ-NER] subject [/SUBJ-NER], [OBJ-NER] obj [/OBJ-NER]
            - typed_entity_marker_punct: @ * subject ner type * subject @, # ^ object ner type ^ object #
        """
        sents = []
        input_format = self.input_format
        if input_format == "entity_mask":
            subj_type = "[SUBJ-{}]".format(subj_type)
            obj_type = "[OBJ-{}]".format(obj_type)
            for token in (subj_type, obj_type):
                if token not in self.new_tokens:
                    self.new_tokens.append(token)
                    self.tokenizer.add_tokens([token])
        elif input_format == "typed_entity_marker":
            subj_start = "[SUBJ-{}]".format(subj_type)
            subj_end = "[/SUBJ-{}]".format(subj_type)
            obj_start = "[OBJ-{}]".format(obj_type)
            obj_end = "[/OBJ-{}]".format(obj_type)

            MUTEX.acquire()
            try:
                for token in (subj_start, subj_end, obj_start, obj_end):
                    if token not in self.new_tokens:
                        self.new_tokens.append(token)
                        self.tokenizer.add_tokens([token])
            finally:
                MUTEX.release()

        elif input_format == "typed_entity_marker_punct":
            subj_type = self.tokenizer.tokenize(subj_type.replace("_", " ").lower())
            obj_type = self.tokenizer.tokenize(obj_type.replace("_", " ").lower())

        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)

            if input_format == "entity_mask":
                if ss <= i_t <= se or os <= i_t <= oe:
                    tokens_wordpiece = []
                    if i_t == ss:
                        new_ss = len(sents)
                        tokens_wordpiece = [subj_type]
                    if i_t == os:
                        new_os = len(sents)
                        tokens_wordpiece = [obj_type]

            elif input_format == "entity_marker":
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ["[E1]"] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ["[/E1]"]
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ["[E2]"] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ["[/E2]"]

            elif input_format == "entity_marker_punct":
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ["@"] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ["@"]
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ["#"] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ["#"]

            elif input_format == "typed_entity_marker":
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = [subj_start] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + [subj_end]
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = [obj_start] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + [obj_end]

            elif input_format == "typed_entity_marker_punct":
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = (
                        ["@"] + ["*"] + subj_type + ["*"] + tokens_wordpiece
                    )
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ["@"]
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = (
                        ["#"] + ["^"] + obj_type + ["^"] + tokens_wordpiece
                    )
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ["#"]

            elif input_format == "REBEL":
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ["@"] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ["@"]
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ["*"]

            sents.extend(tokens_wordpiece)
        return sents


def is_pair_exist(tags, subj_ners, extra_obj_ners, misc_ners):
    subj_entities = [x in subj_ners for x in tags]
    obj_entities = [x in extra_obj_ners for x in tags]
    misc_entities = [x in misc_ners for x in tags]
    if any(obj_entities) and any(subj_entities):
        return True
    elif len([x for x in subj_entities if x]) > 1:
        return True
    elif any(subj_entities) and any(misc_entities):
        return True
    else:
        return False


def get_subj_and_obj_pairs(tags, commutative=True):
    first_loop = [i for i, tag in enumerate(tags) if tag in subj_ners]
    seen = set()
    for subj_start in first_loop:
        idx = 1
        while subj_start + idx < len(tags) and tags[subj_start + idx].startswith("I-"):
            idx += 1
        subj_end = subj_start + idx - 1
        second_loop = [
            i for i, tag in enumerate(tags) if tag in all_ners and i != subj_start
        ]
        for obj_start in second_loop:
            idx = 1
            while obj_start + idx < len(tags) and tags[obj_start + idx].startswith(
                "I-"
            ):
                idx += 1
            obj_end = obj_start + idx - 1
            if commutative:
                entities = tuple(sorted([subj_start, obj_start]))
                if entities in seen:
                    continue
                else:
                    seen.add(entities)

            yield (subj_start, subj_end, tags[subj_start]), (
                obj_start,
                obj_end,
                tags[obj_start],
            )


def get_subj_and_obj_pairs_2(tags, commutative=True, min_score=0.7):
    first_loop = [
        (i, tag)
        for i, tag in enumerate(tags)
        if tag["entity_group"] in subj_ners and tag["score"] > min_score
    ]
    seen = set()
    for (subj_start, subj_tag) in first_loop:
        subj_end = subj_start + 1
        second_loop = [
            (i, tag)
            for i, tag in enumerate(tags)
            if tag["entity_group"] in all_ners
            and i != subj_start
            and tag["score"] > min_score
        ]
        for (obj_start, obj_tag) in second_loop:
            obj_end = obj_start + 1
            if commutative:
                entities = tuple(sorted([subj_start, obj_start]))
                if entities in seen:
                    continue
                else:
                    seen.add(entities)
            yield (subj_start, subj_end, subj_tag["entity_group"]), (
                obj_start,
                obj_end,
                obj_tag["entity_group"],
            )


class Inferer:
    def __init__(
        self,
        path_ner: str,
        path_er: str,
        aggregation_strategy: str = "first",
        commutative_er_model: bool = False,
        min_score_er_model=0.8,
        min_score_ner_model=0.3,
        model_max_length: int = 180,
        max_num_sentences: int = 1,
        test_batch_size=16,
        device="cpu",
    ):
        self.commutative_er_model = commutative_er_model
        self.min_score_ner_model = min_score_ner_model
        self.min_score_er_model = min_score_er_model
        self.model_max_length = model_max_length
        self.max_num_sentences = max_num_sentences

        self.inferer_ner = pipeline(
            "ner",
            model=path_ner,
            tokenizer=path_ner,
            ignore_labels=[],
            aggregation_strategy=aggregation_strategy,
        )
        self.path_ner = path_ner

        self.inferer_er = pipeline(
            "text-classification",
            model=path_er,
            tokenizer=path_er,
        )
        self.path_er = path_ner
        self.tokenizer_er = AutoTokenizer.from_pretrained(path_er, model_max_length=500)
        self.processor = Processor("typed_entity_marker", self.tokenizer_er)

    def infer(self, text):
        args = argparse.Namespace()
        args.max_seq_length = 512
        args.input_format = "typed_entity_marker"

        batch_ans = []
        batches_ner = [
            (i, self.inferer_ner(short_text))
            for (i, short_text) in to_batches_by_razdel(
                text,
                ner_path=self.path_ner,
                model_max_length=self.model_max_length,
                max_num_sentences=self.max_num_sentences,
            )
            if short_text.strip() != ""
        ]

        product_ners = [
            x
            for (_, batch_ner) in batches_ner
            for x in batch_ner
            if x["entity_group"] == "PRODUCT" and x["score"] > 0.7
        ]
        if not product_ners:
            return {"relations": [], "entities": []}

        total_ners = []
        for (offset, batch_ner) in batches_ner:
            for ner in batch_ner:
                if ner["entity_group"] == "O":
                    continue
                new_ner = {
                    "word": ner["word"],
                    "entity": ner["entity_group"],
                    "start": ner["start"] + offset,
                    "end": ner["end"] + offset,
                    "score": ner["score"].item(),
                }
                total_ners.append(new_ner)

        for (text_offset, train_train_) in batches_ner:
            ###
            # warning!!!
            train_train_ = second_agg_v2(train_train_)
            ###
            train_train_tokens = [i["word"] for i in train_train_]
            train_train_tags = [i["entity_group"] for i in train_train_]
            if len(train_train_tokens) < 2:
                continue

            # if the relation model is not commutative, then you
            # should pass commutative=False to the get_subj_and_obj_pairs_2
            text_to_class = []
            ent_to_class = []
            for sample_id, samples in enumerate(
                get_subj_and_obj_pairs_2(
                    train_train_,
                    commutative=self.commutative_er_model,
                    min_score=self.min_score_ner_model,
                )
            ):
                obj_start = samples[0][0]
                subj_start = samples[1][0]
                new_train = {
                    "id": sample_id,
                    "subj": train_train_tokens[subj_start],
                    "subj_type": samples[1][2],
                    "obj": train_train_tokens[obj_start],
                    "obj_type": samples[0][2],
                    "relation": "no_relation",
                    "obj_end": samples[0][1],
                    "obj_start": obj_start,
                    "obj_charend": int(train_train_[obj_start]["end"]) + text_offset,
                    "obj_charstart": int(train_train_[obj_start]["start"])
                    + text_offset,
                    "stanford_ner": train_train_tags,
                    "subj_end": samples[1][1],
                    "subj_start": subj_start,
                    "subj_charend": int(train_train_[subj_start]["end"]) + text_offset,
                    "subj_charstart": int(train_train_[subj_start]["start"])
                    + text_offset,
                    "token": train_train_tokens,
                }
                pre_text = self.processor.tokenize(
                    train_train_tokens,
                    samples[0][2],
                    samples[1][2],
                    samples[0][0],
                    samples[0][1] - 1,
                    samples[1][0],
                    samples[1][1] - 1,
                )
                clean_text = " ".join(pre_text[:500]).replace(" ##", "")
                text_to_class.append(clean_text)
                ent_to_class.append(new_train)

            re_ans = self.inferer_er(text_to_class)
            for ent, re in zip(ent_to_class, re_ans):
                if re["score"] < self.min_score_er_model:
                    pass
                elif re["label"] == "no_relation":
                    pass
                elif re["label"] in RULES_ER.keys():
                    if ent["subj_type"] in RULES_ER[re["label"]]["subj"]:
                        ent_sec = ent
                        ent_sec["relation"] = re["label"]
                        ent_sec["prob"] = re["score"]
                        batch_ans.append(ent_sec)
                else:
                    ent_sec = ent
                    ent_sec["relation"] = re["label"]
                    ent_sec["prob"] = re["score"]
                    batch_ans.append(ent_sec)

        if not batch_ans:
            return {"relations": [], "entities": []}
        return {"relations": batch_ans, "entities": total_ners}
