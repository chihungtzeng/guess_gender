# -*- coding: utf-8 -*-
"""
Check if all chars in names fall within cjk unicode ranges.
"""
import io

CJK_UNICODE_RANGES = [
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2F800, 0x2FA1F)]


def is_within_cjk_range(ucode):
    for start, end in CJK_UNICODE_RANGES:
        if ucode >= start and ucode <= end:
            return True
    return False


def is_standard_chinese_name(name):
    for char in name:
        ucode = ord(char)
        if not is_within_cjk_range(ucode):
            return False
    return True


def __check_standard_chinese_names(lines):
    print("Non-standard Chinese names:")
    for line in lines:
        if not is_standard_chinese_name(line):
            print(line)


def __list_chars_not_in_chinese_names(lines):
    appeared = {}
    for line in lines:
        for char in line:
            ucode = ord(char)
            appeared[ucode] = 1

    print("Chars not used as Chinese names:")
    num_not_appeared = 0
    for start, end in CJK_UNICODE_RANGES:
        for ucode in range(start, end + 1):
            if ucode not in appeared:
                print(u"0x{:x}: {}".format(ucode, chr(ucode)))
                num_not_appeared += 1
    print("{} chars used as Chinese names.".format(len(appeared)))
    print("{} chars not used as Chinese names.".format(num_not_appeared))


def main():
    with io.open("all_names.txt", encoding="utf-8") as _fp:
        lines = _fp.read().splitlines()

    __check_standard_chinese_names(lines)
    __list_chars_not_in_chinese_names(lines)



if __name__ == "__main__":
    main()
