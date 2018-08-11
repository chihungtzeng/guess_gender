# -*- coding: utf-8 -*-
import unittest
import numpy as np
from .calc_feature import (
    _char_to_feature_index,
    _num_features,
    calc_feature_by_name)


class CalcFeatureTest(unittest.TestCase):
    def test__char_to_feature_index(self):
        self.assertEqual(0, _char_to_feature_index(chr(0x3400)))
        self.assertEqual(1, _char_to_feature_index(chr(0x3401)))
        self.assertNotEqual(_char_to_feature_index(chr(0x4dbf)),
                            _char_to_feature_index(chr(0x4e00)))
        self.assertEqual(6592, _char_to_feature_index(chr(0x4e00)))
        self.assertEqual(81519, _char_to_feature_index(chr(0x2fa1f)))
        self.assertEqual(_num_features() - 1,
                         _char_to_feature_index(chr(0x2fa1f)))

    def test_calc_feature_by_name(self):
        feature = calc_feature_by_name(u"建函")
        self.assertTrue(isinstance(feature, np.ndarray))
        idx0 = _char_to_feature_index(u"建")
        idx1 = _char_to_feature_index(u"函")
        for idx in range(0, _num_features()):
            if idx == idx0 or idx == idx1:
                self.assertEqual(1, feature[idx])
            else:
                self.assertEqual(0, feature[idx])


if __name__ == "__main__":
    unittest.main()
