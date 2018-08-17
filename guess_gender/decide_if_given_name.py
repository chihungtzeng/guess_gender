# -*- coding: utf-8 -*-
import argparse
import os
import io
import logging
import time
import math
import pickle


def _get_model_file():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(cur_dir, "model", "char_freq.pkl")


def _read_name_char_freq():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    given_names_file = os.path.join(cur_dir, "data", "all_given_names.txt")
    with io.open(given_names_file, encoding="utf-8") as _fp:
        names = _fp.read().splitlines()
    char_freq = {}
    for name in names:
        for char in name:
            char_freq[char] = 1 + char_freq.get(char, 0)
    for char in char_freq:
        char_freq[char] = 1.0 / (1 + math.exp(-char_freq[char]))
    return char_freq


class GivenNameModel(object):
    def __init__(self, fpath, from_scratch):
        if os.path.isfile(fpath) and not from_scratch:
            logging.warning("Load model: %s", fpath)
            with io.open(fpath, "rb") as _fp:
                self.char_freq = pickle.load(_fp)
        else:
            self.char_freq = _read_name_char_freq()
            self.save(fpath)

    def save(self, fpath):
        logging.warning("Save model: %s", fpath)
        _dir = os.path.dirname(fpath)
        if not os.path.isdir(_dir):
            os.makedirs(_dir)
        with io.open(fpath, "wb") as _fp:
            pickle.dump(self.char_freq, _fp, pickle.HIGHEST_PROTOCOL)

    def __get_name_weight(self, name):
        weight = 0
        for char in name:
            weight += self.char_freq.get(char, -0.73)
        return max(weight, 0)

    def predict_proba(self, names):
        weights = [self.__get_name_weight(name) for name in names]
        print(weights)
        return [math.tanh(_) for _ in weights]

    def predict(self, names):
        return [_ > 0.5 for _ in self.predict_proba(names)]


def main():
    """Prog entry."""
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--show-proba", action="store_true")
    parser.add_argument("--names-to-predict", "-n", nargs="+", default=[])
    args = parser.parse_args()

    model = GivenNameModel(_get_model_file(), args.from_scratch)

    if args.show_proba:
        result = model.predict_proba(args.names_to_predict)
    else:
        result = model.predict(args.names_to_predict)
    for index, name in enumerate(args.names_to_predict):
        print(u"{}: {:f}".format(name, result[index]))

    elapsed = time.time() - start_time
    logging.info(u"Total time: %dm%s", elapsed // 60, elapsed % 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
