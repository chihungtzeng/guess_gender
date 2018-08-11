# -*- coding: utf-8 -*-

CJK_UNICODE_RANGES = [
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0x2F800, 0x2FA1F)]


def char_to_feature_index(char):
    """
    CJK unicode range is sparse. It is inefficient if we use the code of
    |char| as the index of a feature. Hence, we map |char| to a more dense
    feature array.

    Args:
    char -- A unicode char.
    """
    offset = 0
    ucode = ord(char)
    for start, end in CJK_UNICODE_RANGES:
        if ucode >= start and ucode <= end:
            return ucode - start + offset
        else:
            offset += end - start + 1
    return -1


def calc_feature_by_name(name):
    """
    Args:
    name -- A unicode string.
    """
    return None
