# -*- coding: utf-8 -*-
import argparse
import os
import csv
import logging
from .calc_feature import CJK_UNICODE_RANGES


def _read_data(csv_file):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dfile = os.path.join(cur_dir, "data", csv_file)
    logging.info("Read %s", dfile)
    name_genders = []
    with open(dfile, newline="") as _fp:
        rows = csv.reader(_fp)
        for name, gender in rows:
            name_genders.append((name, gender))
    return name_genders


def _within_cjk_unicode_range(char):
    code = ord(char)
    for start, end in CJK_UNICODE_RANGES:
        if code >= start and code <= end:
            return True
    return False


def _does_contain_invalid_char(name):
    for char in name:
        if not _within_cjk_unicode_range(char):
            return True
    return False


def _rm_spurious_names(csv_file):
    ngs = _read_data(csv_file)
    for name, gender in ngs:
        if not _does_contain_invalid_char(name):
            print(u"{},{}".format(name, gender))
        else:
            logging.warning("Invalid name: %s", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    _rm_spurious_names(args.csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
