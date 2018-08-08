#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert *.txt from big5 to utf8.
"""
import argparse
import io
import re
from check_cjk_range import is_standard_chinese_name
__PAT = u"[\\d]{6,}\\s*(?P<name>[^\\d]+)"
__RGX = re.compile(__PAT)


def __extract_names(src_file):
    with io.open(src_file, encoding="utf-8") as _fp:
        lines = _fp.read().splitlines()
    for line in lines:
        for name in __RGX.findall(line):
            name = name.replace(u" ", u"").replace("(", "").replace(")", "")
            if is_standard_chinese_name(name):
                print(name)


def main():
    """
    Prog entry.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-jcee", "-s", required=True)
    args = parser.parse_args()
    __extract_names(args.src_jcee)

if __name__ == "__main__":
    main()
