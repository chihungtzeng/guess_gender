# -*- coding: utf-8 -*-
"""
Check if all chars in names fall within cjk unicode ranges.
"""
import io

CJK_UNICODE_RANGES = [
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF), (0x2F800, 0x2FA1F)]


def is_within_cjk_range(ucode):
    for start, end in CJK_UNICODE_RANGES:
        if ucode >= start and ucode <= end:
            return True
    return False


def main():
    with io.open("all_names.txt", encoding="utf-8") as _fp:
        contents = _fp.read()
    for line in contents.splitlines():
        for char in line:
            ucode = ord(char)
            if not is_within_cjk_range(ucode):
                print(char)



if __name__ == "__main__":
    main()
