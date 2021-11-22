import unittest
from unittest import TestCase
import appvib
import numpy as np


class TestClSig(TestCase):

    def setUp(self):
        self.np_test = np.array([0.1, 1.0, 10.0])
        self.ylim_tb_test = np.array([-1.1, 1.1])

    def test_b_complex(self):

        # Is the real-valued class setting the flags correctly?
        class_test_real = appvib.ClSigReal(self.np_test)
        self.assertFalse(class_test_real.b_complex)

    def test_np_sig(self):

        # Parent class
        class_test = appvib.ClSig(self.np_test)
        self.assertAlmostEqual(self.np_test[0], class_test.np_sig[0], 12)

        # Real-valued child
        class_test_real = appvib.ClSigReal(self.np_test)
        self.assertAlmostEqual(self.np_test[0], class_test_real.np_sig[0], 12)

    def test_i_ns(self):

        # Parent class
        class_test = appvib.ClSig(self.np_test)
        self.assertEqual(class_test.i_ns, 3)

        # Real-valued child
        class_test_real = appvib.ClSigReal(self.np_test)
        self.assertEqual(class_test_real.i_ns, 3)

    def test_ylim_tb(self):

        # Real-valued child
        class_test_real = appvib.ClSigReal(self.np_test)
        class_test_real.ylim_tb = self.ylim_tb_test
        self.assertAlmostEqual(self.ylim_tb_test[0], class_test_real.ylim_tb[0], 12)


if __name__ == '__main__':
    unittest.main()
