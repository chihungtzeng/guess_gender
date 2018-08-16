# -*- coding: utf-8 -*-
import argparse
import os
import logging
import time
import numpy as np
from sklearn import linear_model
from calc_feature import (
    calc_feature_by_name, CJK_UNICODE_RANGES, _char_to_feature_index)
from guess_gender.base_model import BaseModel
from guess_gender.predict_by_random_forest import read_name_gender_from_csv


def _get_lrmodel_file():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(cur_dir, "model", "lrmodel.pkl")


class LRModel(BaseModel):
    """
    Wrapper of Logistic Regression model.
    """
    def __init__(self):
        super().__init__()
        self.ineffective_features = None

    def calc_ineffective_features(self, ndarray):
        """
        Return the column index in which the sum of column values is > 0.

        Args:
        ndarray -- A 2d array.
        """
        column_sums = list(np.sum(ndarray, axis=0))
        self.ineffective_features = [
            idx for idx, value in enumerate(column_sums) if value == 0]

    def rm_ineffective_features(self, ndarray):
        """
        Extract effective columns from |ndarray|.
        """
        return np.delete(ndarray, self.ineffective_features, axis=1)

    def transform_raw_x(self, raw_x):
        """
        Args:
        raw_x -- A list of names.
        """
        name_features = [calc_feature_by_name(_) for _ in raw_x]
        train_x = np.array(name_features)
        if self.ineffective_features is None:
            self.calc_ineffective_features(train_x)
        return self.rm_ineffective_features(train_x)

    def transform_raw_y(self, raw_y):
        """
        Args:
        raw_y -- A list of genders, where a gender is 0(female) or 1 (male).
        """
        return np.array(raw_y, np.int32)

    def _init_model(self):
        return linear_model.LogisticRegression()

    def feature_importances(self):
        importances = self.model.coef_
        idx = 0
        char_importance = []
        for start, end in CJK_UNICODE_RANGES:
            for code in range(start, end+1):
                char = chr(code)
                cidx = _char_to_feature_index(char)
                if cidx not in self.ineffective_features:
                    importance = importances[0][idx]
                    char_importance.append((char, importance))
                    idx += 1
        char_importance.sort(key=lambda x: x[1])
        for char, importance in char_importance:
            print(u"{} {}".format(char, importance))


def __validate(lrmodel):
    names, genders = read_name_gender_from_csv("testdata.csv")
    result = lrmodel.predict(names)
    correct = 0
    for index, name in enumerate(names):
        expect = genders[index]
        actual = result[index]
        if expect != actual:
            logging.info("%s: Expect: %d, actual: %d", name, expect, actual)
        else:
            correct += 1
    accuracy = correct / len(names)
    logging.info("Accuracy: %d/%d = %f", correct, len(names), accuracy)


def main():
    """Prog entry."""
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--show-proba", action="store_true")
    parser.add_argument("--names-to-predict", "-n", nargs="+", default=[])
    args = parser.parse_args()

    fpath = _get_lrmodel_file()
    lrmodel = LRModel()

    if args.from_scratch or not os.path.isfile(fpath):
        names, genders = read_name_gender_from_csv("gender.csv")
        lrmodel.train(names, genders)
        lrmodel.save(fpath)
    else:
        lrmodel = BaseModel.load(fpath)

    if args.show_proba:
        result = lrmodel.predict_proba(args.names_to_predict)
    else:
        result = lrmodel.predict(args.names_to_predict)
    for index, name in enumerate(args.names_to_predict):
        print(u"{}: {}".format(name, result[index]))

    # dump internal data:
    # __validate(lrmodel)
    # lrmodel.feature_importances()

    elapsed = time.time() - start_time
    logging.info(u"Total time: %dm%s", elapsed // 60, elapsed % 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
