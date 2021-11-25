import unittest
from unittest import TestCase
import appvib
import math
import numpy as np


class TestClSig(TestCase):

    def setUp(self):
        # Define the initial values for the test
        self.np_test = np.array([0.1, 1.0, 10.0])
        self.np_test_ch2 = np.array([2.1, 3.0, 12.0])
        self.np_test_ch3 = np.array([3.1, 4.0, 13.0])

        # Data set for the real-valued class. This is a sawtooth waveform. For
        # triggering on rising values it should trigger at 7.5 seconds (between
        # sample 8 and 9). For falling it should trigger at 4.5 seconds (between
        # sample 5 and 6).
        self.np_test_real = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0])
        self.d_fs_real = 1.0
        self.d_threshold_real = 2.5
        self.d_hysteresis_real = 0.1
        self.i_direction_real_rising = 0
        self.i_direction_real_falling = 1

        # Data set for the signal feature class. Intent is to push more data through the class
        self.d_fs_test_trigger = 2047
        i_ns = (self.d_fs_test_trigger * 3)
        self.d_freq_law = 10.
        d_time_ext = np.linspace(0, (i_ns - 1), i_ns) / float(self.d_fs_test_trigger)
        self.np_test_trigger = np.sin(2 * math.pi * self.d_freq_law * d_time_ext)
        self.d_threshold_test_trigger = 0.0
        self.d_hysteresis_test_trigger = 0.1
        self.i_direction_test_trigger_rising = 0
        self.i_direction_test_trigger_falling = 1

        self.np_test_comp = np.array([0.1 - 0.2j, 1.0 - 2.0j, 10.0 - 20j])
        self.np_test_comp_long = np.array([0.1 - 0.2j, 1.0 - 2.0j, 10.0 - 20j, 100.0 - 200j, 1000.0 - 2000j])
        self.ylim_tb_test = [-1.1, 1.1]
        self.ylim_tb_test_alt = [-3.3, 3.3]
        self.d_fs = 1.024e3
        self.d_fs_ch2 = 2.048e3
        self.d_fs_ch3 = 4.096e3
        self.str_eu_default = "volts"
        self.str_eu_acc = "g's"

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

    def test_str_eu(self):
        # Signal feature base class unit check
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        self.assertEqual(self.str_eu_default, class_test_sig_features.str_eu())
        class_test_sig_features.str_eu = self.str_eu_acc
        self.assertEqual(self.str_eu_acc, class_test_sig_features.str_eu)

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

    def test_np_d_est_triggers(self):
        # Real-valued check, rising signal
        class_test_real = appvib.ClSigReal(self.np_test_real, self.d_fs_real)
        class_test_real.np_d_est_triggers(np_sig_in=self.np_test_real, i_direction=self.i_direction_real_rising,
                                          d_threshold=self.d_threshold_real, d_hysteresis=self.d_hysteresis_real,
                                          b_verbose=True)
        self.assertAlmostEqual(class_test_real.np_d_eventtimes[0], 7.5, 12)

        print('--------------------')

        # Real-valued check, falling signal
        class_test_real = appvib.ClSigReal(self.np_test_real, self.d_fs_real)
        class_test_real.np_d_est_triggers(np_sig_in=self.np_test_real, i_direction=self.i_direction_real_falling,
                                          d_threshold=self.d_threshold_real, d_hysteresis=self.d_hysteresis_real,
                                          b_verbose=True)
        self.assertAlmostEqual(class_test_real.np_d_eventtimes[0], 4.5, 12)

        # Signal feature class test, rising signal with threshold of zero
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_trigger, self.d_fs_test_trigger)
        print('Signal frequency, hertz: ' + '%0.6f' % self.d_freq_law)
        class_test_sig_features.plt_sigs()
        class_test_sig_features.np_d_est_triggers(np_sig_in=self.np_test_trigger,
                                                  i_direction=self.i_direction_test_trigger_rising,
                                                  d_threshold=self.d_threshold_test_trigger,
                                                  d_hysteresis=self.d_hysteresis_test_trigger,
                                                  b_verbose=False)
        d_est_freq = 1./(np.mean(np.diff(class_test_sig_features.np_d_eventtimes())))
        self.assertAlmostEqual(d_est_freq, self.d_freq_law, 7)

        # check the plot
        class_test_sig_features.plt_eventtimes()

        # Signal feature class test, falling signal with threshold of zero
        class_test_sig_features.np_d_est_triggers(np_sig_in=self.np_test_trigger,
                                                  i_direction=self.i_direction_test_trigger_falling,
                                                  d_threshold=self.d_threshold_test_trigger,
                                                  d_hysteresis=self.d_hysteresis_test_trigger,
                                                  b_verbose=False)
        d_est_freq = 1./(np.mean(np.diff(class_test_sig_features.np_d_eventtimes())))
        self.assertAlmostEqual(d_est_freq, self.d_freq_law, 7)


if __name__ == '__main__':
    unittest.main()
