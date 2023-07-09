import datetime
import json
import random
from typing import Dict, List, Optional
import fire
import os

from utils import get_brat_annotations, open_lines


def main(path: Optional[str] = None, dataset_name: Optional[str] = ""):
    """[Convert files from brat to tacred]
    
    Keyword Arguments:
        path {Optional[str]} -- [is a string in glob format] (default: {None})
         e.g. "brat/data/relations/Экономика/*/*.ann"
        dataset_name {Optional[str]} -- [description] (default: {""})
    """
    if not dataset_name:
        dataset_name = "./"
        prefix = datetime.datetime.now().strftime("%b_%d").lower()
        dataset_name = f"{dataset_name}{prefix}/"
    os.makedirs(dataset_name + "/ner_data/", exist_ok=True)
    gold_files = open_lines("golden_files.txt")
    if path:
        if not path.endswith(".ann"):
            raise Exception("path should end with '.ann'")
        text_data = get_brat_annotations(path)
    else:
        text_data = get_brat_annotations()
    with open(f"{dataset_name}/all_data.json", "w") as f:
        json.dump(text_data, f)
    all_data: List[Dict] = []

    for k, v in text_data.items():
        for sent_i, sent in enumerate(v):
            sent["gold"] = False
            sent["filename"] = k
            if k in gold_files:
                sent["gold"] = True
        all_data += list(v)

    filter_data = [d for d in all_data if d["relation"] or d["gold"]]
    with open(f"{dataset_name}/brat_tacred.json", "w") as f:
        json.dump(filter_data, f)

    gold_data = [d for d in all_data if d["gold"]]
    with open(f"{dataset_name}/brat_tacred_gold.json", "w") as f:
        json.dump(gold_data, f)
    random.shuffle(filter_data)

    train = filter_data[: int(len(filter_data) * 0.8)]
    valid = filter_data[
        int(len(filter_data) * 0.8) : int(len(filter_data) * 0.9)
    ]
    test = filter_data[int(len(filter_data) * 0.9) :]

    keys = ["train", "valid", "test"]
    #keys = [f"{prefix}_{k}" for k in keys]

    datasets = dict(zip(keys, [train, valid, test]))

    for name, data in datasets.items():
        data_strings = []
        for d in data:
            zipped = list(zip(*[d["token"], d["stanford_ner"]]))
            data_string = "\n".join(["\t".join(z) for z in zipped]) + "\n"
            data_strings.append(data_string)
        with open(f"{dataset_name}/ner_data/{name}.txt", "w") as f:
            f.write("\n".join(data_strings))


if __name__ == "__main__":
    fire.Fire(main)
