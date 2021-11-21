import unittest
from unittest import TestCase
import appvib
import numpy as np


class TestClSig(TestCase):

    def test_b_complex(self):

        # Is the real-valued class setting the flags correctly?
        np_test = np.array([0.1, 1.0, 10.0])
        class_test_real = appvib.ClSigReal(np_test)
        self.assertFalse(class_test_real.b_complex)

    def test_np_sig(self):

        # Parent class
        np_test = np.array([0.1, 1.0, 10.0])
        class_test = appvib.ClSig(np_test)
        self.assertAlmostEqual(np_test[0], class_test.np_sig[0], 12)

        # Real-valued child
        class_test_real = appvib.ClSigReal(np_test)
        self.assertAlmostEqual(np_test[0], class_test_real.np_sig[0], 12)

    def test_i_ns(self):
        self.fail()

    def test_ylim_tb(self):
        self.fail()

    def test_ylim_tb(self):
        self.fail()

    def test_set_ylim_tb(self):
        self.fail()
