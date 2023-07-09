import re
import select
import sys
from collections import Counter
from glob import glob
from typing import Any, Dict, List, Set, Tuple

import pymorphy2
import razdel
import torch
from transformers import AutoTokenizer
from country_lemmas import change_dict

morph = pymorphy2.MorphAnalyzer()


def get_annotation_from_file(
        annotation_filename: str, with_relations_only: bool = False
) -> List[dict]:
    with open(annotation_filename) as f:
        annotation_str: str = f.read()
    annotation_values: List[dict] = []
    if with_relations_only and "Arg1:" not in annotation_str:
        return []
    annotation: List = annotation_str.split("\n")
    annotation = [a.strip() for a in annotation]
    annotation = [a for a in annotation if a]
    annotation = [a.split("\t") for a in annotation]
    annotation_entries: dict = dict()
    ners = [a for a in annotation if a[0][0] == "T"]
    relations = [a for a in annotation if a[0][0] == "R"]
    annotation = ners + relations
    for a in annotation:
        annotation_entry: dict = dict()
        if a[0].startswith("T"):
            annotation_type = "NER"
        elif a[0].startswith("R"):
            annotation_type = "Relation"
        else:
            annotation_type = "Unknown"
        annotation_entry["type"] = annotation_type
        annotation_entry["code"] = annotation_code = a[0]

        annotation_body = a[1]
        annotation_body = annotation_body.replace("Arg1:", "").replace(
            "Arg2:", ""
        )
        annotation_body = annotation_body.split()
        annotation_entry["ann_type"] = annotation_body[0]
        if annotation_type == "NER":
            if len(a) <= 2:
                continue
            annotation_entry["start"] = int(annotation_body[1])
            annotation_entry["end"] = int(annotation_body[2])
            annotation_entry["text"] = a[2]
        else:
            for item_id, _ in enumerate(("subj", "obj")):
                item_id += 1
                for position in ("start", "end"):
                    annotation_entry[
                        f"{item_id}_{position}"
                    ] = annotation_entries[annotation_body[item_id]][position]
            annotation_entry["subj_ref"] = annotation_body[1]
            annotation_entry["obj_ref"] = annotation_body[2]
        annotation_entries[annotation_code] = annotation_entry
    annotation_values = list(annotation_entries.values())
    return annotation_values


def open_annotation() -> List[str]:
    with open("brat/data/annotation.conf") as f:
        annotation_conf = f.readlines()

    relations = []
    started = False
    for line in annotation_conf:
        if not started:
            if line == "[relations]\n":
                started = True
            continue
        else:
            if line == "[events]\n":
                break
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            relations.append(line)
    return relations


def get_allowed_relation_matches() -> Tuple[
    Dict[str, set], Dict[str, Dict[str, set]]
]:
    relations = open_annotation()
    cleaned_relations = [l.split() for l in relations]

    allowed_matches: Dict[str, set] = dict()
    relations_dict: Dict[str, Dict[str, set]] = dict()
    for r in cleaned_relations:
        relation_type = r[0]
        r = [l.split(":")[1] for l in r[1:3]]
        if not r:
            continue
        key_str, value_str = r
        keys = key_str.replace(",", "").split("|")
        values = set(value_str.split("|"))
        relations_dict[relation_type] = dict()
        relations_dict[relation_type]["keys"] = set(keys)
        relations_dict[relation_type]["values"] = values
        for key in keys:
            if key in allowed_matches:
                allowed_matches[key] = allowed_matches[key].union(values)
            else:
                allowed_matches[key] = values
    return allowed_matches, relations_dict


def get_allowed_types_for_relations():
    relations = open_annotation()
    relations = [r.split() for r in relations]


def get_brat_annotations(
        path="brat/data/relations/Экономика/*/*.ann", with_relations_only=False
) -> Dict[str, list]:
    annotations = glob(path)

    # key_ners = set(["PROFESSION", "GROUP"])

    attach_flag = False
    text_data: Dict[str, list] = dict()

    for annotation_filename in annotations:
        try:
            annotation_values = get_annotation_from_file(
                annotation_filename, with_relations_only=with_relations_only
            )
        except (KeyError, ValueError, IndexError) as ex:
            print(ex, annotation_filename)
        if with_relations_only:
            if len(annotation_values) <= 3:
                continue
        else:
            if not annotation_values:
                continue
        text_filename = annotation_filename.replace(".ann", ".txt")
        with open(text_filename) as f:
            text = f.read()
        sents = list(razdel.sentenize(text))
        prev_sent = None
        text_data[annotation_filename] = []
        for sent in sents:
            if attach_flag and prev_sent:
                sent.text = prev_sent.text + " " + sent.text
                sent.start = prev_sent.start
                attach_flag = False
                prev_sent = None
            sent_tokens = list(razdel.tokenize(sent.text))
            sent_ners = ["O" for t in sent_tokens]
            sent_brat_codes = ["O" for t in sent_tokens]
            sent_data: Dict = dict()
            sent_data["token"] = [t.text for t in sent_tokens]
            sent_data["text"] = sent.text
            sent_data["relation"] = False
            sent_data["relations"] = []
            for _, value in enumerate(annotation_values):
                annotation_code = value["code"]
                if value["type"] == "NER":
                    start = value["start"]
                    end = value["end"]
                    ner_text = value["text"]
                    ner_type = value["ann_type"]
                    overlaps = check_pair_in_range(
                        start, end, range(sent.start, sent.stop)
                    )
                    if overlaps:
                        for s_t_i, s_t in enumerate(sent_tokens):
                            token_overlaps = check_pair_in_range(
                                start,
                                end,
                                range(
                                    s_t.start + sent.start,
                                    s_t.stop + sent.start,
                                ),
                            )
                            if token_overlaps:
                                if ner_text.startswith(s_t.text):
                                    sent_ners[s_t_i] = "B-" + ner_type
                                else:
                                    sent_ners[s_t_i] = "I-" + ner_type
                                sent_brat_codes[s_t_i] = annotation_code
                if value["type"] == "Relation":
                    t1_overlaps = check_pair_in_range(
                        value["1_start"],
                        value["1_end"],
                        range(sent.start, sent.stop),
                    )
                    t2_overlaps = check_pair_in_range(
                        value["2_start"],
                        value["2_end"],
                        range(sent.start, sent.stop),
                    )
                    if t1_overlaps and t2_overlaps:
                        sent_data["relation"] = True
                        relation: Dict = dict()
                        relation["relation"] = value["ann_type"]
                        relation["code"] = annotation_code
                        for entry in ("subj", "obj"):
                            codes = [
                                i
                                for i, s in enumerate(sent_brat_codes)
                                if s == value[f"{entry}_ref"]
                            ]
                            if not codes:
                                continue
                            relation[f"{entry}_type"] = (
                                sent_ners[codes[0]]
                                    .replace("B-", "")
                                    .replace("I-", "")
                            )
                            relation[f"{entry}_brat_code"] = value[
                                f"{entry}_ref"
                            ]
                            relation[f"{entry}_start"] = codes[0]
                            relation[f"{entry}_end"] = codes[-1]
                        if relation:
                            sent_data["relations"].append(relation)
                    elif t1_overlaps or t2_overlaps:
                        prev_sent = sent
                        attach_flag = True
            sent_data["stanford_ner"] = sent_ners
            sent_data["brat_codes"] = sent_brat_codes
            sent_data["start"] = sent.start
            sent_data["stop"] = sent.stop
            # all_data.append(sent_data)
            text_data[annotation_filename].append(sent_data)
    return text_data


def convert_ner_to_bio(ner: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    bio_ner = []
    for word_i, (word, tag) in enumerate(ner):
        if tag != "O":
            if word_i == 0:
                tag = "B-" + tag
            else:
                if ner[word_i - 1][1] == tag:
                    tag = "I-" + tag
                else:
                    tag = "B-" + tag
        bio_ner.append((word, tag))
    return bio_ner


def combine_bio_ners(ner: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    words: List[str] = []
    tags: List[str] = []
    last_word = ""
    last_tag = ""
    for _, (word, tag) in enumerate(ner):
        short_tag = tag.replace("B-", "").replace("I-", "")
        if tag == "O":
            if last_word:
                words.append(last_word)
                tags.append(last_tag)
                last_word = ""
                last_tag = ""
            words.append(word)
            tags.append(tag)
        elif tag.startswith("I-"):
            if short_tag == last_tag:
                last_word += f" {word}"
            else:
                words.append(last_word)
                tags.append(last_tag)
                last_word = word
                last_tag = short_tag
        elif tag.startswith("B-"):
            if last_word:
                words.append(last_word)
                tags.append(last_tag)
            last_word = word
            last_tag = short_tag
    if last_word:
        words.append(last_word)
        tags.append(last_tag)
    output = list(zip(words, tags))
    return output


def get_bio_subjects(ner: List[Tuple[str, str]]) -> List[dict]:
    subjects: List[dict] = []
    subject: Dict = {"tokens": [], "token_ids": [], "tag": ""}
    for word_i, (word, tag) in enumerate(ner):
        short_tag = tag.replace("B-", "").replace("I-", "")
        if tag == "O":
            if subject["tokens"]:
                subjects.append(subject)
                subject = {"tokens": [], "token_ids": [], "tag": ""}
        elif tag.startswith("I-"):
            if short_tag == subject["tag"]:
                subject["tokens"].append(word)
                subject["token_ids"].append(word_i)
            else:
                if subject["tokens"]:
                    subjects.append(subject)
                subject = {
                    "tokens": [word],
                    "token_ids": [word_i],
                    "tag": short_tag,
                }
        elif tag.startswith("B-"):
            if subject["tokens"]:
                subjects.append(subject)
            subject = {
                "tokens": [word],
                "token_ids": [word_i],
                "tag": short_tag,
            }
    return subjects


def construct_sequence_relation_samples(sentence: Dict[str, Any]):
    """[summary]
    post('http://10.8.0.3:3333', json={"token": [], ner=['O'] * len(...),
    label=[... B-[SELF] ...]})
    Arguments:
        annotation {List[Dict[str, Any]]} -- [description]
    """
    subjects = get_bio_subjects(
        list(zip(sentence["token"], sentence["stanford_ner"]))
    )
    for subject in subjects:
        subject["token"] = sentence["token"]
        subject["ner"] = sentence["stanford_ner"]
        subject["label"] = ["O" for s in sentence["token"]]
        subject["label"][subject["token_ids"][0]] = "B-[SELF]"
        for token_id in subject["token_ids"][1:]:
            subject["label"][token_id] = "I-[SELF]"
    return subjects


def clean_error_annotations():
    "cleaning merged lines like T2T3"
    files = glob("brat/data/relations/Экономика/*/*.ann")
    for file in files:
        with open(file) as f:
            r = f.read()
        if re.search(r"T\d+T\d+", r):
            r = re.sub(r"T(\d+)T(\d+)", r"T\g<1>\nT\g<2>", r)
            with open(file, "w") as f:
                f.write(r)
    # remove empty lines (containing only \n)
    for file in files:
        with open(file) as f:
            lines = f.readlines()
        orig_length = len(lines)
        if orig_length == 0:
            continue
        lines = [l.strip() for l in lines[:-1] if l.strip()] + [lines[-1]]
        new_length = len(lines)
        if orig_length != new_length:
            with open(file, "w") as f:
                f.write("\n".join(lines))


def replace_gpes():
    files = glob("brat/data/relations/Экономика/*/*.ann")
    counter = Counter()
    for file in files:
        with open(file) as f:
            lines = f.readlines()
        if not lines:
            continue
        changed = False
        to_remove = set()
        for line_i, line in enumerate(lines):
            if line.startswith("T"):
                word = line.strip().split("\t")[-1]
                ner_type = line.split("\t")[1].split()[0]
                # if ner_type in {"LOCATION", "GPE"}:
                if ner_type in {"NATIONALITY", "NORP"}:
                    words = word.split()
                    lemmas = [morph.parse(w)[0].normal_form for w in words]
                    lemma = " ".join(lemmas)
                    counter[lemma] += 1
                    # continue
                    for key, value in change_dict.items():
                        if lemma in value:
                            changed = True
                            if key == "REMOVE":
                                to_remove.add(line_i)
                            else:
                                line = line.replace("NATIONALITY", key)
                                line = line.replace("NORP", key)
                                # line = line.replace("LOCATION", key)
                                # line = line.replace("GPE", key)
                                lines[line_i] = line
                                # line_changes.append([old_line, line])
                            break
        if changed:
            lines = [l for l_i, l in enumerate(lines) if l_i not in to_remove]
            new_text = "".join(lines)
            # changes.append([old_text, new_text])
            with open(file, "w") as f:
                f.write(new_text)


def filter_relevant_ners(stanford_ner: List[str]) -> List[Dict]:
    ners = []
    last_ner: Dict = dict()
    for tag_i, tag in enumerate(stanford_ner):
        if tag.startswith("B-") or tag == "O":
            if last_ner:
                ners.append(last_ner)
            if tag.startswith("B-"):
                last_ner = {
                    "type": tag.replace("B-", ""),
                    "token_ids": [tag_i],
                }
            else:
                last_ner = dict()
        else:
            # sometimes models dont learn the BIO-format
            # and may output something like
            # O, I-ORG, I-ORG, I-PERSON, O
            short_tag = tag.replace("I-", "")
            if last_ner and last_ner["type"] == short_tag:
                last_ner["token_ids"].append(tag_i)
            else:
                if last_ner:
                    ners.append(last_ner)
                last_ner = {"type": short_tag, "token_ids": [tag_i]}
    return ners


def input_with_timeout(prompt, timeout=5, *args, **kwargs):
    """[Input with timeout; Unix only]
    https://stackoverflow.com/a/15533404/

    Arguments:
        prompt {[type]} -- [description]

    Keyword Arguments:
        timeout {int} -- [description] (default: {5})

    Returns:
        [type] -- [description]
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        # expect stdin to be line-buffered
        output = sys.stdin.readline().rstrip("\n")
        if output != "1":
            output = "0"
        return output
    return "0"


def relation_approvement(sentence: dict, relation: dict) -> bool:
    """[input() based manual annotation for releations]

    Arguments:
        sentence {dict} -- [a dict of this format]
            {'token': [
                'В',
                'Госдуме',
                'предложили',
                'запретить',
                '.'],
            'relation': True,
            'relations': [{
                'relation': 'WORKPLACE',
                'subj_type': 'PERSON',
                'subj_brat_code': 'T4',
                'subj_start': 15,
                'subj_end': 16,
                'obj_type': 'ORGANIZATION',
                'obj_brat_code': 'T3',
                'obj_start': 12,
                'obj_end': 14}],
            'stanford_ner': [
                'O',
                'B-ORGANIZATION',
                'O',
                'O'
                'O'],
            'brat_codes': [
                'O',
                'T1',
                'O',
                'O',
                'O'],
            'start': 0,
            'stop': 245}
        relation {dict} -- [a dict of this format]
            {
                'label': 'WORKPLACE',
                'relation': 'WORKPLACE',
                'subj_text': 'Бессонов',
                'subj_type': 'PERSON',
                'subj_brat_code': 'T4',
                'subj_start': 16,
                'subj_end': 16,
                'obj_text': 'по обороне',
                'obj_type': 'ORGANIZATION',
                'obj_brat_code': 'T3',
                'obj_start': 13,
                'obj_end': 14}
    Returns:
        bool -- [True or False, based on input() '1' or '0']
    """
    print(" ".join(sentence["token"]))
    max_logits = None
    if "logits" in relation:
        max_logits = max(relation["logits"].values())
    print(
        relation["subj_text"],
        relation["obj_text"],
        relation["label"],
        max_logits,
    )
    allowed_inputs = {"1", "0"}
    user_input = ""
    while user_input not in allowed_inputs:
        user_input = input_with_timeout("Do you approve the annotation? ", 5)
        if user_input not in allowed_inputs:
            print("user input should be one of ", allowed_inputs)
    return bool(int(user_input))


def annotation_relation_dict(role_dict: dict, sent: dict, roletext: str):
    """[converts output of filter_relevant_ners to brat_typed relation]

    Arguments:
        role_dict {dict} -- [a dictionary with keys `type` and `token_ids`]
            {'type': 'ORGANIZATION', 'token_ids': [13, 14]}
        sent {dict} -- [a sent in tacred format]
            {'token': [
                'В',
                'Госдуме',
                'предложили',
                'запретить',
                '.'],
            'relation': True,
            'relations': [{
                'relation': 'WORKPLACE',
                'subj_type': 'PERSON',
                'subj_brat_code': 'T4',
                'subj_start': 15,
                'subj_end': 16,
                'obj_type': 'ORGANIZATION',
                'obj_brat_code': 'T3',
                'obj_start': 12,
                'obj_end': 14}],
            'stanford_ner': [
                'O',
                'B-ORGANIZATION',
                'O',
                'O'
                'O'],
            'brat_codes': [
                'O',
                'T1',
                'O',
                'O',
                'O'],
            'start': 0,
            'stop': 245}
        roletext {str} -- [a string 'subj' or 'obj']

    Returns:
        [type] -- [description]
    """
    text = " ".join([sent["token"][t] for t in role_dict["token_ids"]])
    brat_code = sent["brat_codes"][role_dict["token_ids"][0]]
    relation_dict = {
        f"{roletext}_text": text,
        f"{roletext}_type": role_dict["type"],
        f"{roletext}_brat_code": brat_code,
        f"{roletext}_start": role_dict["token_ids"][0],
        f"{roletext}_end": role_dict["token_ids"][-1],
        f"{roletext}_token_ids": role_dict["token_ids"],
    }
    return relation_dict


def write_relations_to_file(
        relations: List[Dict], filename: str, last_relation_code: int
):
    relation_txt = ""
    for relation_i, f_relation in enumerate(relations):
        relation_string = "R{}\t{} Arg1:{} Arg2:{}\t\n".format(
            relation_i + 1 + last_relation_code,
            f_relation["label"],
            f_relation["subj_brat_code"],
            f_relation["obj_brat_code"],
        )
        relation_txt += relation_string
    if relation_txt:
        # file_codes = set([f["subj_brat_code"] for f in relations])
        print("http://10.8.0.2:8001/index.xhtml#/" + filename[10:-4])
        with open(filename, "r") as f:
            text = f.read()
        if text.endswith("\n\n"):
            text = text[:-1]
        with open(filename, "w") as f:
            f.write(text + relation_txt)


def get_last_relation_code(annotation):
    last_code = 0
    for sent in annotation:
        if sent["relations"]:
            relations = [r for r in sent["relations"] if "code" in r]
            if not relations:
                continue
            sent_code = max(int(r["code"][1:]) for r in relations)
            if sent_code > last_code:
                last_code = sent_code
    return last_code


def get_last_ner_code(annotation):
    last_code = 0
    for sent in annotation:
        if not sent["brat_codes"]:
            continue
        codes = [int(code[1:]) for code in sent["brat_codes"] if len(code) > 1]
        if codes:
            sent_code = max(codes)
        else:
            sent_code = 0
        if sent_code > last_code:
            last_code = sent_code
    return last_code


def workplace_annotation(text_data: Dict):
    triplets: Set[str] = set()
    triplets_counter: Counter = Counter()
    for filename, annotation in text_data.items():
        file_relations: List[Dict] = []
        max_relation_code = get_last_relation_code(annotation)
        for sent in annotation:
            if len(sent["relations"]) != 1:
                continue

            relation = sent["relations"][0]
            if relation.get("subj_type") != "PERSON":
                continue
            if relation.get("obj_type") != "PROFESSION":
                continue
            relevant_ners = filter_relevant_ners(sent["stanford_ner"])
            # is_gap = False
            for rel_ner_i in range(2, len(relevant_ners)):
                rel_ner = relevant_ners[rel_ner_i]
                if rel_ner["type"] != "PERSON":
                    continue
                if relevant_ners[rel_ner_i - 2]["type"] != "PROFESSION":
                    continue
                if relevant_ners[rel_ner_i - 1]["type"] not in (
                        "COUNTRY",
                        "CITY",
                        "ORGANIZATION",
                        "GPE",
                        "REGION",
                ):
                    continue
                subj = rel_ner
                obj = relevant_ners[rel_ner_i - 1]
                pre_obj = relevant_ners[rel_ner_i - 2]
                pre_obj_text = " ".join(
                    [sent["token"][t] for t in pre_obj["token_ids"]]
                )
                pre_obj_text = " ".join(
                    [
                        morph.parse(w)[0].normal_form
                        for w in pre_obj_text.split()
                    ]
                )

                relation = {"label": "WORKPLACE", "relation": "WORKPLACE"}
                relation.update(annotation_relation_dict(subj, sent, "subj"))
                relation.update(annotation_relation_dict(obj, sent, "obj"))
                obj_text = " ".join(
                    [
                        morph.parse(w)[0].normal_form
                        for w in relation["obj_text"].split()
                    ]
                )
                subj_text = " ".join(
                    [
                        morph.parse(w)[0].normal_form
                        for w in relation["subj_text"].split()
                    ]
                )
                triplet = pre_obj_text + "; " + obj_text + "; " + subj_text
                triplets_counter[triplet] += 1
                if triplet in triplets:
                    approved = True
                else:
                    approved = relation_approvement(sent, relation)
                if approved:
                    file_relations.append(relation)
                    triplets.add(triplet)
        if file_relations:
            print(filename)
            write_relations_to_file(
                file_relations, filename, max_relation_code
            )


def open_lines(filename: str) -> Set[str]:
    with open(filename) as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    output = set(lines)
    return output


def find_ner_match(ner_text, sent_text, tried=False):
    sent_text = " ".join(sent_text.split())
    match = re.search(ner_text, sent_text, re.UNICODE)
    if not match:
        if tried:
            print(ner_text, "; ", sent_text)
            print(ner_text in sent_text)
            return None
        ner_text = ner_text.replace(", ", ",")
        # ner_text = re.escape(ner_text)
        # sent_text = re.escape(sent_text)
        span = find_ner_match(ner_text, sent_text, tried=True)
    else:
        span = match.span()
    return span


join_conll_bio_ners = convert_ner_to_bio


def to_batches(long_text, ner_path, model_max_length=None):
    tokenizer = AutoTokenizer.from_pretrained(ner_path)
    model_max_length = (
        tokenizer.model_max_length
        if model_max_length is None
        else model_max_length
    )
    # tokenizer.model_max_length = 512

    # tokenize without truncation
    inputs_no_trunc = tokenizer(
        long_text,
        max_length=None,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )

    # get batches of tokens corresponding to the exact model_max_length
    chunk_start = 0
    chunk_end = model_max_length  # == 1024 for Bart
    inputs_batch_lst = []
    while chunk_start <= len(inputs_no_trunc["input_ids"][0]):
        inputs_batch = inputs_no_trunc["input_ids"][0][
                       chunk_start:chunk_end
                       ]  # get batch of n tokens
        inputs_batch = torch.unsqueeze(inputs_batch, 0)
        inputs_batch_lst.append(inputs_batch)
        chunk_start += model_max_length  # == 1024 for Bart
        chunk_end += model_max_length  # == 1024 for Bart

    return [tokenizer.decode(i[0]) for i in inputs_batch_lst]


def to_batches_by_razdel(long_text, ner_path, model_max_length=None, max_num_sentences=1):
    """
        assume that razdel.sentenizer will split long_text to sentences
        short enough to fit in the model
    """
    tokenizer = AutoTokenizer.from_pretrained(ner_path)
    model_max_length = (
        tokenizer.model_max_length
        if model_max_length is None
        else model_max_length
    )
    # tokenizer.model_max_length = 512
    sentences = [x for x in razdel.sentenize(long_text)]
    tokenized_sents_lens = [len(tokenizer.tokenize(x.text)) for x in sentences]
    total_num_sents = len(sentences)

    batches = []
    bucket = []
    num_words = 0
    overflow = False
    for sent_id, (sent, sent_len) in enumerate(zip(sentences, tokenized_sents_lens)):
        num_words += sent_len
        bucket.append(sent)
        if len(bucket) == max_num_sentences or sent_id == total_num_sents - 1:
            start = bucket[0].start
            stop = bucket[-1].stop
            batches.append((start, long_text[start:stop]))
            bucket = []
            num_words = 0
        elif num_words > model_max_length:
            start = bucket[0].start
            stop = bucket[-2].stop
            if sent_id == total_num_sents - 1:
                overflow = True
            batches.append((start, long_text[start:stop]))
            bucket = bucket[-1:]
            num_words = sent_len

    if overflow:
        start = bucket[-1].start
        stop = bucket[-1].stop
        batches.append((start, long_text[start:stop]))

    return batches
