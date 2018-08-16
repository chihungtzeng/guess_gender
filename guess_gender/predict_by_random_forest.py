# -*- coding: utf-8 -*-
import argparse
import os
import csv
import logging
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from calc_feature import (
    calc_feature_by_name, CJK_UNICODE_RANGES, _char_to_feature_index)
from guess_gender.base_model import BaseModel


def read_name_gender_from_csv(csv_file):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dfile = os.path.join(cur_dir, "data", csv_file)
    names = []
    genders = []
    with open(dfile, newline="") as _fp:
        rows = csv.reader(_fp)
        for name, gender in rows:
            genders.append(int(gender))
            names.append(name)
    return names, genders


def _get_rfmodel_file():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(cur_dir, "model", "rfmodel.pkl")


def __validate(rfmodel):
    names, genders = read_name_gender_from_csv("testdata.csv")
    result = rfmodel.predict(names)
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


class RFModel(BaseModel):
    """
    Wrapper of Random Forest model.
    """
    def __init__(self):
        super().__init__()
        self.ineffective_features = None

    def __calc_ineffective_features(self, ndarray):
        """
        Return the column index in which the sum of column values is > 0.

        Args:
        ndarray -- A 2d array.
        """
        column_sums = list(np.sum(ndarray, axis=0))
        self.ineffective_features = [
            idx for idx, value in enumerate(column_sums) if value == 0]

    def __rm_ineffective_features(self, ndarray):
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
            self.__calc_ineffective_features(train_x)
        return self.__rm_ineffective_features(train_x)

    def transform_raw_y(self, raw_y):
        """
        Args:
        raw_y -- A list of genders, where a gender is 0(female) or 1(male).
        """
        return np.array(raw_y, np.int32)

    def _init_model(self):
        return RandomForestClassifier(n_estimators=64, n_jobs=4)

    def feature_importances(self):
        importances = self.model.feature_importances_
        idx = 0
        char_importance = []
        for start, end in CJK_UNICODE_RANGES:
            for code in range(start, end+1):
                char = chr(code)
                cidx = _char_to_feature_index(char)
                if cidx not in self.ineffective_features:
                    importance = importances[idx]
                    char_importance.append((char, importance))
                    idx += 1
        char_importance.sort(key=lambda x: x[1])
        for char, importance in char_importance:
            print(u"{} {}".format(char, importance))


def main():
    """Prog entry."""
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--show-proba", action="store_true")
    parser.add_argument("--names-to-predict", "-n", nargs="+", default=[])
    args = parser.parse_args()

    fpath = _get_rfmodel_file()
    rfmodel = RFModel()

    if args.from_scratch or not os.path.isfile(fpath):
        names, genders = read_name_gender_from_csv("gender.csv")
        rfmodel.train(names, genders)
        rfmodel.save(fpath)
    else:
        rfmodel = BaseModel.load(fpath)

    if args.show_proba:
        result = rfmodel.predict_proba(args.names_to_predict)
    else:
        result = rfmodel.predict(args.names_to_predict)
    for index, name in enumerate(args.names_to_predict):
        print(u"{}: {}".format(name, result[index]))

    # dump internal data:
    # __validate(rfmodel)
    # rfmodel.feature_importances()

    elapsed = time.time() - start_time
    logging.info(u"Total time: %dm%s", elapsed // 60, elapsed % 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
