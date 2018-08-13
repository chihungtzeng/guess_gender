# -*- coding: utf-8 -*-
import argparse
import os
import csv
import io
import logging
import time
import pickle
import numpy as np
from sklearn import linear_model
from calc_feature import (
    calc_feature_by_name, CJK_UNICODE_RANGES, _char_to_feature_index)


def _read_data(csv_file):
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


def _get_lrmodel_file():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(cur_dir, "model", "lrmodel.pkl")


class LRModel(object):
    """
    Wrapper of Logistic Regression model.
    """
    def __init__(self):
        self.lrmodel = None
        self.ineffective_features = None

    def predict(self, names):
        if not names:
            names = [u"承憲", u"均平", u"建安", u"美雲", u"乃馨", u"建民",
                     u"莎拉波娃", u"青", u"去病"]
        test_x = np.array([calc_feature_by_name(_) for _ in names])
        test_x = self.rm_ineffective_features(test_x)
        predict_result = self.lrmodel.predict(test_x)
        return predict_result

    def predict_proba(self, names):
        if not names:
            names = [u"承憲", u"均平", u"建安", u"美雲", u"乃馨", u"建民",
                     u"莎拉波娃", u"青", u"去病"]
        test_x = np.array([calc_feature_by_name(_) for _ in names])
        test_x = self.rm_ineffective_features(test_x)
        predict_result = self.lrmodel.predict_proba(test_x)
        return predict_result

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

    def train(self):
        """
        Train model.
        """
        names, genders = _read_data("gender.csv")

        name_features = [calc_feature_by_name(_) for _ in names]
        self.lrmodel = linear_model.LogisticRegression()

        logging.info("Feature selection")
        train_x = np.array(name_features)
        self.calc_ineffective_features(train_x)
        train_x = self.rm_ineffective_features(train_x)
        logging.info(train_x.shape)

        train_y = np.array(genders, np.int32)

        logging.debug(train_y.shape)

        logging.info("Train random forest")
        self.lrmodel.fit(train_x, train_y)
        self.__save()

    def feature_importances(self):
        importances = self.lrmodel.coef_
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

    def __save(self):
        fpath = _get_lrmodel_file()
        logging.warning("Save model to %s", fpath)
        _dir = os.path.dirname(fpath)
        if not os.path.isdir(_dir):
            os.makedirs(_dir)

        with io.open(fpath, "wb") as _fp:
            pickle.dump(self, _fp, pickle.HIGHEST_PROTOCOL)


def __load_lrmodel():
    fpath = _get_lrmodel_file()
    if os.path.isfile(fpath):
        logging.warning("Load from %s", fpath)
        with io.open(fpath, "rb") as _fp:
            lrmodel = pickle.load(_fp)
    else:
        lrmodel = LRModel()
        lrmodel.train()
    return lrmodel


def __validate(lrmodel):
    names, genders = _read_data("testdata.csv")
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

    if args.from_scratch:
        lrmodel = LRModel()
        lrmodel.train()
    else:
        lrmodel = __load_lrmodel()

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
