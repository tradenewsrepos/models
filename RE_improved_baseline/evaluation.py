import numpy as np
import unittest
from sklearn.metrics import f1_score, classification_report
from random import shuffle


def get_f1(key, prediction):
    correct_by_relation = ((key == prediction) & (prediction != 0)).astype(np.int32).sum()
    guessed_by_relation = (prediction != 0).astype(np.int32).sum()
    gold_by_relation = (key != 0).astype(np.int32).sum()

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro


class TestEvaluation(unittest.TestCase):
    def test_evaluation(self):
        _pred = [2, 2, 2, 1, 1, 3, 3, 6, 3, 1, 2, 0]
        _key = [0, 0, 1, 2, 1, 3, 1, 6, 0, 0, 0, 0]
        key = np.array(_key)
        for _ in range(50):
            shuffle(_pred)
            pred = np.array(_pred)
            expected = classification_report(
                key, pred, labels=sorted(set([x for x in key if x])), output_dict=True
            )['micro avg']['f1-score']
            got = get_f1(key, pred)[-1]
            self.assertEqual(got, expected,
                             f'expected: {expected}, got: {got}')
