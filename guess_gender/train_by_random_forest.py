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

def __read_data():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dfile = os.path.join(cur_dir, "data", "gender.txt")
    names = []
    genders = []
    with open(dfile, newline="") as _fp:
        rows = csv.reader(_fp)
        for name, gender in rows:
            igender = int(gender)
            for char in name:
                names.append(ord(char))
                genders.append(igender)
    return names, genders


def __predict(rfmodel):
    names = [u"承憲", u"均平", u"建安", u"美雲", u"乃馨"]
    for name in names:
        x = np.array([ord(_) for _ in name], np.int32)
        x = x.reshape(-1, 1)
        prob = rfmodel.predict_proba(x)
        print(prob)


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


def __get_rfmodel(force):
    if not force:
        rfmodel = __load_rfmodel()
        if rfmodel:
            return rfmodel
    # Train model
    names, genders = __read_data()
    rfmodel = RandomForestClassifier(n_estimators=500, n_jobs=4)
    logging.info("Train random forest")
    x = np.array(names, np.int32)
    y = np.array(genders, np.int32)
    x = x.reshape(-1, 1)
    print(x)
    print(x.shape)
    print(y.shape)
    rfmodel.fit(x, y)
    __save_rfmodel(rfmodel)
    return rfmodel


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rfmodel = __get_rfmodel(args.force)
    __predict(rfmodel)
    elapsed = time.time() - start_time
    logging.info(u"Total time: %dm%s", elapsed // 60, elapsed % 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
