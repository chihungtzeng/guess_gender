# -*- coding: utf-8 -*-
import argparse
import os
import csv
import io
import logging
import time
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from calc_feature import calc_feature_by_name


def _read_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dfile = os.path.join(cur_dir, "data", "gender.txt")
    name_features = []
    genders = []
    with open(dfile, newline="") as _fp:
        rows = csv.reader(_fp)
        for name, gender in rows:
            genders.append(int(gender))
            name_features.append(calc_feature_by_name(name))
    return name_features, genders


def _get_rfmodel_file():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(cur_dir, "model", "rfmodel.pkl")


class RFModel(object):
    """
    Wrapper of Random Forest model.
    """
    def __init__(self):
        self.rfmodel = None
        self.ineffective_features = None

    def predict(self, names, show_proba):
        if not names:
            names = [u"承憲", u"均平", u"建安", u"美雲", u"乃馨", u"建民",
                     u"莎拉波娃", u"青", u"去病"]
        test_x = np.array([calc_feature_by_name(_) for _ in names])
        test_x = self.rm_ineffective_features(test_x)
        proba = self.rfmodel.predict_proba(test_x)
        for index, name in enumerate(names):
            gender = u"男" if proba[index][1] > 0.5 else u"女"
            if show_proba:
                print(u"{} {} {}".format(name, gender, proba[index][1]))
            else:
                print(u"{} {}".format(name, gender))

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
        name_features, genders = _read_data()
        self.rfmodel = RandomForestClassifier(n_estimators=64, n_jobs=4)

        logging.info("Feature selection")
        train_x = np.array([_f for _f in name_features])
        self.calc_ineffective_features(train_x)
        train_x = self.rm_ineffective_features(train_x)
        logging.info(train_x.shape)

        train_y = np.array(genders, np.int32)

        logging.debug(train_y.shape)

        logging.info("Train random forest")
        self.rfmodel.fit(train_x, train_y)
        self.__save()

    def __save(self):
        fpath = _get_rfmodel_file()
        logging.warning("Save model to %s", fpath)
        _dir = os.path.dirname(fpath)
        if not os.path.isdir(_dir):
            os.makedirs(_dir)

        with io.open(fpath, "wb") as _fp:
            pickle.dump(self, _fp, pickle.HIGHEST_PROTOCOL)


def __load_rfmodel():
    fpath = _get_rfmodel_file()
    if os.path.isfile(fpath):
        logging.warning("Load from %s", fpath)
        with io.open(fpath, "rb") as _fp:
            rfmodel = pickle.load(_fp)
    return rfmodel


def main():
    """Prog entry."""
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--show-proba", action="store_true")
    parser.add_argument("--names-to-predict", "-n", nargs="+", default=None)
    args = parser.parse_args()

    if args.from_scratch:
        rfmodel = RFModel()
        rfmodel.train()
    else:
        rfmodel = __load_rfmodel()
    rfmodel.predict(args.names_to_predict, args.show_proba)
    elapsed = time.time() - start_time
    logging.info(u"Total time: %dm%s", elapsed // 60, elapsed % 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
