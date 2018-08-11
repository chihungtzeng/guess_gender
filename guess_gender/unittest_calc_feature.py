# -*- coding: utf-8 -*-
import unittest
from calc_feature import char_to_feature_index, CJK_UNICODE_RANGES


class CalcFeatureTest(unittest.TestCase):
    def test_char_to_feature_index(self):
        self.assertEqual(0, char_to_feature_index(chr(0x3400)))
        self.assertEqual(1, char_to_feature_index(chr(0x3401)))
        self.assertNotEqual(char_to_feature_index(chr(0x4dbf)),
                            char_to_feature_index(chr(0x4e00)))
        self.assertEqual(6592, char_to_feature_index(chr(0x4e00)))
        self.assertEqual(81519, char_to_feature_index(chr(0x2fa1f)))

        prev_idx = -1
        for start, end in CJK_UNICODE_RANGES:
            for ucode in range(start, end + 1):
                idx = char_to_feature_index(chr(ucode))
                if idx == prev_idx:
                    print("{:x}".format(ucode))
                self.assertTrue(idx != prev_idx)
                prev_idx = idx


if __name__ == "__main__":
    unittest.main()
