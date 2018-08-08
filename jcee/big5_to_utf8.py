#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert *.txt from big5 to utf8.

Note:
Use python2.7 to avoid encoding problems.
"""
import io
import os
import logging
import subprocess
import sys


def __get_src_file_names():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ret = []
    for fname in os.listdir(cur_dir):
        if fname.endswith(".txt") and "utf8" not in fname:
            fullpath = os.path.join(cur_dir, fname)
            ret.append(fullpath)
    return ret


def __get_dest_file_name(src_fname):
    return src_fname.replace(".txt", "_utf8.txt")


def __iconv_big5_to_utf8(fullpath):
    logging.warning("Handle %s", fullpath)
    with open(fullpath) as _fp:
        contents = _fp.read()
    new_lines = []
    for line in contents.splitlines():
        uline = []
        for item in line.split():
            try:
                uitem = item.decode("big5")
                # full space -> half space.
                uitem = uitem.replace(u"　", u"  ")
            except UnicodeDecodeError:
                uitem = u""
            if uitem:
                uline.append(uitem)
        new_lines.append(u" ".join(uline))
    fcontents = u"\n".join(new_lines)
    dest_file = __get_dest_file_name(fullpath)
    logging.warning("Write %s", dest_file)
    with io.open(dest_file, "w", encoding="utf-8") as _fp:
        _fp.write(fcontents)
        _fp.write(u"\n")


def main():
    """
    Prog entry.
    """
    for fpath in __get_src_file_names():
        __iconv_big5_to_utf8(fpath)

if __name__ == "__main__":
    main()
