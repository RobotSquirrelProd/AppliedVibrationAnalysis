import unittest
from unittest import TestCase
import appvib
import numpy as np


class TestClSig(TestCase):

    def setUp(self):

        # Define the initial values for the test
        self.np_test = np.array([0.1, 1.0, 10.0])
        self.np_test_real = np.array([1.0, 2.0, 3.0])
        self.np_test_comp = np.array([0.1-0.2j, 1.0-2.0j, 10.0-20j])
        self.np_test_comp_long = np.array([0.1-0.2j, 1.0-2.0j, 10.0-20j, 100.0-200j, 1000.0-2000j])
        self.ylim_tb_test = np.array([-1.1, 1.1])
        self.ylim_tb_test_alt = np.array([-3.3, 3.3])
        self.timebase_scale_test = 1e-1

    def test_b_complex(self):

        # Is the real-valued class setting the flags correctly?
        class_test_real = appvib.ClSigReal(self.np_test)
        self.assertFalse(class_test_real.b_complex)

        # Is the complex-valued class setting the flags correctly?
        class_test_comp = appvib.ClSigComp(self.np_test_comp)
        self.assertTrue(class_test_comp.b_complex)
        self.assertFalse(class_test_real.b_complex)

        # Is the signal feature class setting flags correctly?
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.timebase_scale_test)
        self.assertTrue(class_test_comp.b_complex)
        self.assertFalse(class_test_real.b_complex)
        self.assertFalse(class_test_sig_features.b_complex)

    def test_np_sig(self):

        # Real-valued child
        class_test_real = appvib.ClSigReal(self.np_test_real)
        self.assertAlmostEqual(self.np_test_real[0], class_test_real.np_sig[0], 12)

        # Complex-valued child and verify inheritance is working
        class_test_comp = appvib.ClSigComp(self.np_test_comp)
        self.assertAlmostEqual(self.np_test_comp[0], class_test_comp.np_sig[0], 12)
        self.assertAlmostEqual(self.np_test_real[0], class_test_real.np_sig[0], 12)

        # Signal feature class
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.timebase_scale_test)
        self.assertAlmostEqual(self.np_test_comp[0], class_test_comp.np_sig[0], 12)
        self.assertAlmostEqual(self.np_test_real[0], class_test_real.np_sig[0], 12)
        self.assertAlmostEqual(self.np_test[0], class_test_sig_features.np_d_ch1[0], 12)

    def test_i_ns(self):

        # Real-valued child number of samples test
        class_test_real = appvib.ClSigReal(self.np_test)
        self.assertEqual(class_test_real.i_ns, 3)

        # Complex-valued child sample count correct?
        class_test_comp = appvib.ClSigComp(self.np_test_comp_long)
        self.assertEqual(class_test_real.i_ns, 3)
        self.assertEqual(class_test_comp.i_ns, 5)

        # Signal feature class check on sample count
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.timebase_scale_test)
        self.assertEqual(class_test_sig_features.i_ns, 3)
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_comp_long, self.timebase_scale_test)
        self.assertEqual(class_test_sig_features.i_ns, 5)

        # TO DO: need to test validation that all signals are the same length

    def test_ylim_tb(self):

        # Real-valued child y-limits test
        class_test_real = appvib.ClSigReal(self.np_test)
        class_test_real.ylim_tb = self.ylim_tb_test
        self.assertAlmostEqual(self.ylim_tb_test[0], class_test_real.ylim_tb[0], 12)

        # Complex-valued child y-limits test
        class_test_comp = appvib.ClSigComp(self.np_test_comp)
        class_test_comp.ylim_tb = self.ylim_tb_test
        self.assertAlmostEqual(self.ylim_tb_test[0], class_test_comp.ylim_tb[0], 12)
        class_test_comp.ylim_tb = self.ylim_tb_test_alt
        self.assertAlmostEqual(self.ylim_tb_test_alt[1], class_test_comp.ylim_tb[1], 12)


if __name__ == '__main__':
    unittest.main()
