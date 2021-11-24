import unittest
from unittest import TestCase
import appvib
import numpy as np


class TestClSig(TestCase):

    def setUp(self):

        # Define the initial values for the test
        self.np_test = np.array([0.1, 1.0, 10.0])
        self.np_test_ch2 = np.array([2.1, 3.0, 12.0])
        self.np_test_ch3 = np.array([3.1, 4.0, 13.0])
        self.np_test_real = np.array([1.0, 2.0, 3.0])
        self.np_test_comp = np.array([0.1-0.2j, 1.0-2.0j, 10.0-20j])
        self.np_test_comp_long = np.array([0.1-0.2j, 1.0-2.0j, 10.0-20j, 100.0-200j, 1000.0-2000j])
        self.ylim_tb_test = [-1.1, 1.1]
        self.ylim_tb_test_alt = [-3.3, 3.3]
        self.d_fs = 1.024e3
        self.d_fs_ch2 = 2.048e3
        self.d_fs_ch3 = 4.096e3

    def test_b_complex(self):

        # Is the real-valued class setting the flags correctly?
        class_test_real = appvib.ClSigReal(self.np_test, self.d_fs)
        self.assertFalse(class_test_real.b_complex)

        # Is the complex-valued class setting the flags correctly?
        class_test_comp = appvib.ClSigComp(self.np_test_comp, self.d_fs)
        self.assertTrue(class_test_comp.b_complex)
        self.assertFalse(class_test_real.b_complex)

        # Is the signal feature class setting flags correctly?
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        self.assertTrue(class_test_comp.b_complex)
        self.assertFalse(class_test_real.b_complex)
        self.assertFalse(class_test_sig_features.b_complex)

    def test_np_sig(self):

        # Real-valued child
        class_test_real = appvib.ClSigReal(self.np_test_real, self.d_fs)
        self.assertAlmostEqual(self.np_test_real[0], class_test_real.np_sig[0], 12)

        # Attempt to send a complex-valued signal to the real-valued class
        with self.assertRaises(Exception):
            class_test_real = appvib.ClSigReal(self.np_test_comp, self.d_fs)

        # Complex-valued child and verify inheritance is working
        class_test_comp = appvib.ClSigComp(self.np_test_comp, self.d_fs)
        self.assertAlmostEqual(self.np_test_comp[0], class_test_comp.np_sig[0], 12)
        self.assertAlmostEqual(self.np_test_real[0], class_test_real.np_sig[0], 12)

        # Signal feature class, first signal
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        self.assertAlmostEqual(self.np_test_comp[0], class_test_comp.np_sig[0], 12)
        self.assertAlmostEqual(self.np_test[0], class_test_sig_features.np_d_sig[0], 12)

        # Signal feature class, second signal
        idx_new = class_test_sig_features.idx_add_sig(self.np_test_ch2, self.d_fs)
        self.assertEqual(idx_new, 1, msg='Failed to return correct index')
        self.np_return = class_test_sig_features.get_np_d_sig(idx=1)
        self.assertAlmostEqual(self.np_test_ch2[0], self.np_return[0], 12)
        self.assertAlmostEqual(self.np_test[0], class_test_sig_features.np_d_sig[0], 12)

        # Signal feature class, third signal
        idx_new = class_test_sig_features.idx_add_sig(self.np_test_ch3, d_fs_in=self.d_fs_ch3)
        self.assertEqual(idx_new, 2, msg='Failed to return correct index')
        self.np_return = class_test_sig_features.get_np_d_sig(idx=2)
        self.assertAlmostEqual(self.np_test_ch3[1], self.np_return[1], 12)
        self.assertAlmostEqual(self.np_test[0], class_test_sig_features.np_d_sig[0], 12)

    def test_i_ns(self):

        # Real-valued child number of samples test
        class_test_real = appvib.ClSigReal(self.np_test, self.d_fs)
        self.assertEqual(class_test_real.i_ns, 3)

        # Complex-valued child sample count correct?
        class_test_comp = appvib.ClSigComp(self.np_test_comp_long, self.d_fs)
        self.assertEqual(class_test_real.i_ns, 3)
        self.assertEqual(class_test_comp.i_ns, 5)

        # Signal feature class check on sample count for the first signal
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        self.assertEqual(class_test_sig_features.i_ns, 3)
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_comp_long, self.d_fs)
        self.assertEqual(class_test_sig_features.i_ns, 5)

        # Signal feature class check on sample count for the second signal
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_comp_long, self.d_fs)
        with self.assertRaises(Exception):
            class_test_sig_features.idx_add_sig(self.np_test_ch2, d_fs_in=self.d_fs_ch2)
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        self.assertEqual(class_test_sig_features.i_ns, 3)

    def test_ylim_tb(self):

        # Real-valued child y-limits test
        class_test_real = appvib.ClSigReal(self.np_test, self.d_fs)
        class_test_real.set_ylim_tb(self.ylim_tb_test)
        self.assertAlmostEqual(self.ylim_tb_test[0], class_test_real.ylim_tb[0], 12)

        # Complex-valued child y-limits test
        class_test_comp = appvib.ClSigComp(self.np_test_comp, self.d_fs)
        class_test_comp.ylim_tb = self.ylim_tb_test
        self.assertAlmostEqual(self.ylim_tb_test[0], class_test_comp.ylim_tb[0], 12)
        class_test_comp.ylim_tb = self.ylim_tb_test_alt
        self.assertAlmostEqual(self.ylim_tb_test_alt[1], class_test_comp.ylim_tb[1], 12)

        # Signal feature class check on y-limits test
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        class_test_sig_features.ylim_tb = self.ylim_tb_test
        self.assertAlmostEqual(self.ylim_tb_test[0], class_test_sig_features.ylim_tb[0], 12)

    def test_d_fs(self):

        # Signal feature class check signal sampling frequency on instantiation
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        self.assertAlmostEqual(self.d_fs, class_test_sig_features.d_fs(), 12)

        # Add a second signal with a different sampling rate
        idx_new = class_test_sig_features.idx_add_sig(self.np_test_ch2, self.d_fs)
        self.assertEqual(idx_new, 1, msg='Failed to return correct index')
        class_test_sig_features.d_fs_update(self.d_fs_ch2, idx=1)

    def test_plt_sigs(self):

        # Signal feature class check of plotting on instantiation
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        class_test_sig_features.plt_sigs()

        # Signal feature class, second signal
        idx_new = class_test_sig_features.idx_add_sig(self.np_test_ch2, self.d_fs)
        self.assertEqual(idx_new, 1, msg='Failed to return correct index')
        class_test_sig_features.plt_sigs()

    def test_plt_spec(self):

        # Signal feature class check of plotting on instantiation
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        class_test_sig_features.plt_spec()

        # Add peak label
        class_test_sig_features.b_spec_peak = True
        class_test_sig_features.plt_spec()


if __name__ == '__main__':
    unittest.main()
