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


def __read_data():
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


def __predict(rfmodel, names, show_proba):
    if not names:
        names = [u"承憲", u"均平", u"建安", u"美雲", u"乃馨", u"建民",
                 u"莎拉波娃", u"青", u"去病"]
    for name in names:
        _x = calc_feature_by_name(name)
        _x = _x.reshape(1, -1)
        proba = rfmodel.predict_proba(_x)[0]

        gender = u"男" if proba[1] > 0.5 else u"女"
        if show_proba:
            print(u"{} {} {}".format(name, gender, proba))
        else:
            print(u"{} {}".format(name, gender))


def __get_rfmodel_file():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(cur_dir, "model", "rfmodel.pkl")


def __load_rfmodel():
    fpath = __get_rfmodel_file()
    if not os.path.isfile(fpath):
        return None
    logging.warning("Load from %s", fpath)
    with io.open(fpath, "rb") as _fp:
        rfmodel = pickle.load(_fp)
    return rfmodel


def __save_rfmodel(rfmodel):
    fpath = __get_rfmodel_file()
    logging.warning("Dump to %s", fpath)
    _dir = os.path.dirname(fpath)
    if not os.path.isdir(_dir):
        os.makedirs(_dir)

    with io.open(fpath, "wb") as _fp:
        pickle.dump(rfmodel, _fp, pickle.HIGHEST_PROTOCOL)


def __get_rfmodel(from_scratch):
    if not from_scratch:
        rfmodel = __load_rfmodel()
        if rfmodel:
            return rfmodel
    # Train model
    name_features, genders = __read_data()
    rfmodel = RandomForestClassifier(n_estimators=64, n_jobs=4)
    logging.info("Train random forest")
    train_x = np.array([_f for _f in name_features])
    train_y = np.array(genders, np.int32)
    logging.debug(train_x.shape)
    logging.debug(train_y.shape)
    rfmodel.fit(train_x, train_y)
    __save_rfmodel(rfmodel)
    return rfmodel


def main():
    """Prog entry."""
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--show-proba", action="store_true")
    parser.add_argument("--names-to-predict", "-n", nargs="+", default=None)
    args = parser.parse_args()

    rfmodel = __get_rfmodel(args.from_scratch)
    __predict(rfmodel, args.names_to_predict, args.show_proba)
    elapsed = time.time() - start_time
    logging.info(u"Total time: %dm%s", elapsed // 60, elapsed % 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
