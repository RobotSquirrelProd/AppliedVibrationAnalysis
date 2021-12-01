import unittest
from unittest import TestCase
import appvib
import math
import numpy as np
import csv
import pandas as pd


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
        self.d_test_trigger_amp = 1.0
        d_time_ext = np.linspace(0, (i_ns - 1), i_ns) / float(self.d_fs_test_trigger)
        self.np_test_trigger = self.d_test_trigger_amp * np.sin(2 * math.pi * self.d_freq_law * d_time_ext)
        self.d_test_trigger_amp_ch2 = 2.1
        self.np_test_trigger_ch2 = self.d_test_trigger_amp_ch2 * np.sin(2 * math.pi * self.d_freq_law * d_time_ext)
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

        # Test data set 000 : This one caused the nx_est to fail
        self.str_filename_000 = 'test_appvib_data_000.csv'
        self.df_test_000 = pd.read_csv(self.str_filename_000, header=None, skiprows=1,
                                       names=['Ch1', 'Ch2', 'FS'])
        self.np_d_test_data_000_Ch1 = self.df_test_000.Ch1
        self.np_d_test_data_000_Ch2 = self.df_test_000.Ch2
        self.d_fs_data_000 = self.df_test_000.FS[0]
        self.i_direction_test_000_trigger_slope = 0
        self.d_threshold_test_000 = 0.125

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
        self.assertAlmostEqual(self.np_test_real[0], class_test_real.np_d_sig[0], 12)

        # Attempt to send a complex-valued signal to the real-valued class
        with self.assertRaises(Exception):
            class_test_real = appvib.ClSigReal(self.np_test_comp, self.d_fs)

        # Complex-valued child and verify inheritance is working
        class_test_comp = appvib.ClSigComp(self.np_test_comp, self.d_fs)
        self.assertAlmostEqual(self.np_test_comp[0], class_test_comp.np_d_sig[0], 12)
        self.assertAlmostEqual(self.np_test_real[0], class_test_real.np_d_sig[0], 12)

        # Signal feature class, first signal
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        self.assertAlmostEqual(self.np_test_comp[0], class_test_comp.np_d_sig[0], 12)
        self.assertAlmostEqual(self.np_test[0], class_test_sig_features.np_d_sig[0], 12)

        # Signal feature class, second signal
        idx_new = class_test_sig_features.idx_add_sig(self.np_test_ch2, self.d_fs)
        self.assertEqual(idx_new, 1, msg='Failed to return correct index')
        self.np_return = class_test_sig_features.get_np_d_sig(idx=1)
        self.assertAlmostEqual(self.np_test_ch2[0], self.np_return[0], 12)
        self.assertAlmostEqual(self.np_test[0], class_test_sig_features.np_d_sig[0], 12)

        # Signal feature class, third signal
        idx_new = class_test_sig_features.idx_add_sig(self.np_test_ch3, d_fs=self.d_fs_ch3)
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
            class_test_sig_features.idx_add_sig(self.np_test_ch2, d_fs=self.d_fs_ch2)
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
        class_test_sig_features.ylim_tb(ylim_tb_in=self.ylim_tb_test, idx=0)
        d_ylim_tb_check = class_test_sig_features.ylim_tb()
        self.assertAlmostEqual(self.ylim_tb_test[0], d_ylim_tb_check[0], 12)

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

        # Signal feature class, second signal auto y-limits
        idx_new = class_test_sig_features.idx_add_sig(self.np_test_ch2, self.d_fs)
        self.assertEqual(idx_new, 1, msg='Failed to return correct index')
        class_test_sig_features.plt_sigs()

        # Signal feature class, second signal manual y-limits
        class_test_sig_features.ylim_tb(ylim_tb_in=[0.0, 16.0], idx=0)
        class_test_sig_features.plt_sigs()

    def test_plt_spec(self):
        # Signal feature class check of plotting on instantiation
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        class_test_sig_features.plt_spec()

        # Add peak label
        class_test_sig_features.b_spec_peak = True
        class_test_sig_features.plt_spec()

    def test_np_d_est_triggers(self):

        # Real-valued check, rising signal, explicitly defining the arguments
        class_test_real = appvib.ClSigReal(self.np_test_real, self.d_fs_real)
        class_test_real.np_d_est_triggers(np_sig_in=self.np_test_real, i_direction=self.i_direction_real_rising,
                                          d_threshold=self.d_threshold_real, d_hysteresis=self.d_hysteresis_real,
                                          b_verbose=True)
        self.assertAlmostEqual(class_test_real.np_d_eventtimes[0], 7.5, 12)

        print('--------------------')

        # Real-valued check, rising signal, inferred arguments
        class_test_real.np_d_est_triggers(b_verbose=True)
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
        class_test_sig_features.np_d_est_triggers(np_d_sig=self.np_test_trigger,
                                                  i_direction=self.i_direction_test_trigger_rising,
                                                  d_threshold=self.d_threshold_test_trigger,
                                                  d_hysteresis=self.d_hysteresis_test_trigger,
                                                  b_verbose=False)
        d_est_freq = 1. / (np.mean(np.diff(class_test_sig_features.np_d_eventtimes())))
        self.assertAlmostEqual(d_est_freq, self.d_freq_law, 7)

        # check the plot
        class_test_sig_features.plt_eventtimes()

        # Signal feature class test, falling signal with threshold of zero
        class_test_sig_features.np_d_est_triggers(np_d_sig=self.np_test_trigger,
                                                  i_direction=self.i_direction_test_trigger_falling,
                                                  d_threshold=self.d_threshold_test_trigger,
                                                  d_hysteresis=self.d_hysteresis_test_trigger,
                                                  b_verbose=False)
        d_est_freq = 1. / (np.mean(np.diff(class_test_sig_features.np_d_eventtimes())))
        self.assertAlmostEqual(d_est_freq, self.d_freq_law, 7)

    def test_nX_est(self):

        # Test real signal, rising signal
        class_test_real = appvib.ClSigReal(self.np_test_trigger, self.d_fs_test_trigger)
        d_eventtimes_real = class_test_real.np_d_est_triggers(np_sig_in=class_test_real.np_d_sig,
                                                              i_direction=self.i_direction_test_trigger_rising,
                                                              d_threshold=self.d_threshold_test_trigger,
                                                              d_hysteresis=self.d_hysteresis_test_trigger,
                                                              b_verbose=False)
        self.assertAlmostEqual(d_eventtimes_real[0], 1. / self.d_freq_law, 7)
        self.assertAlmostEqual(d_eventtimes_real[-1] - d_eventtimes_real[-2], 1. / self.d_freq_law, 7)
        np_d_nx = class_test_real.calc_nx(np_sig_in=class_test_real.np_d_sig, np_d_eventtimes=d_eventtimes_real,
                                          b_verbose=False)
        self.assertAlmostEqual(np.abs(np_d_nx[0]), self.d_test_trigger_amp, 2)
        self.assertLess(np.rad2deg(np.angle(np_d_nx[0])) - 90.0, 1.0)
        self.assertAlmostEqual(np.abs(np_d_nx[-1]), self.d_test_trigger_amp, 2)
        self.assertLess(np.rad2deg(np.angle(np_d_nx[-1])) - 90.0, 1.0)

        # Signal feature class test using same input as the real signal class above
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_trigger, self.d_fs_test_trigger)
        d_eventtimes_sig = class_test_sig_features.np_d_est_triggers(np_d_sig=class_test_sig_features.np_d_sig,
                                                                     i_direction=self.i_direction_test_trigger_rising,
                                                                     d_threshold=self.d_threshold_test_trigger,
                                                                     d_hysteresis=self.d_hysteresis_test_trigger,
                                                                     b_verbose=False, idx=0)
        self.assertAlmostEqual(d_eventtimes_sig[0], 1. / self.d_freq_law, 7)
        self.assertAlmostEqual(d_eventtimes_sig[-1] - d_eventtimes_sig[-2], 1. / self.d_freq_law, 7)
        np_d_nx_sig = class_test_sig_features.calc_nx(np_d_sig=class_test_sig_features.np_d_sig,
                                                      np_d_eventtimes=d_eventtimes_real,
                                                      b_verbose=False, idx=0)
        self.assertAlmostEqual(np.abs(np_d_nx_sig[0]), self.d_test_trigger_amp, 2)
        self.assertLess(np.rad2deg(np.angle(np_d_nx_sig[0])) - 90.0, 1.0)
        self.assertAlmostEqual(np.abs(np_d_nx_sig[-1]), self.d_test_trigger_amp, 2)
        self.assertLess(np.rad2deg(np.angle(np_d_nx_sig[-1])) - 90.0, 1.0)

        # This call structure surfaced a reference bug
        class_test_sig_features = appvib.ClSigFeatures(self.np_d_test_data_000_Ch1,
                                                       self.d_fs_data_000)
        class_test_sig_features.idx_add_sig(np_d_sig=self.np_d_test_data_000_Ch2,
                                            d_fs=self.d_fs_data_000)
        class_test_sig_features.np_d_est_triggers(np_d_sig=class_test_sig_features.np_d_sig,
                                                  i_direction=self.i_direction_test_000_trigger_slope,
                                                  d_threshold=self.d_threshold_test_000)
        class_test_sig_features.calc_nx(np_d_sig=class_test_sig_features.np_d_sig,
                                        np_d_eventtimes=class_test_sig_features.np_d_eventtimes(),
                                        b_verbose=False, idx=0)
        class_test_sig_features.calc_nx(np_d_sig=class_test_sig_features.get_np_d_sig(idx=1),
                                        np_d_eventtimes=class_test_sig_features.np_d_eventtimes(),
                                        b_verbose=False, idx=0)

    def test_plt_apht(self):

        # Test real signal, rising signal
        class_test_real = appvib.ClSigReal(self.np_test_trigger, self.d_fs_test_trigger)
        d_eventtimes_real = class_test_real.np_d_est_triggers(np_sig_in=class_test_real.np_d_sig,
                                                              i_direction=self.i_direction_test_trigger_rising,
                                                              d_threshold=self.d_threshold_test_trigger,
                                                              d_hysteresis=self.d_hysteresis_test_trigger,
                                                              b_verbose=False)
        np_d_nx = class_test_real.calc_nx(np_sig_in=class_test_real.np_d_sig, np_d_eventtimes=d_eventtimes_real,
                                          b_verbose=False)
        self.assertAlmostEqual(np.abs(np_d_nx[0]), self.d_test_trigger_amp, 2)
        class_test_real.plt_apht()
        class_test_real.str_plot_apht_desc = 'Test data'
        class_test_real.ylim_apht_mag = [-0.1, 1.1]
        class_test_real.plt_apht()

        # Signal feature class test for apht plots
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_trigger, self.d_fs_test_trigger)
        d_eventtimes_sig = class_test_sig_features.np_d_est_triggers(np_d_sig=class_test_real.np_d_sig,
                                                                     i_direction=self.i_direction_test_trigger_rising,
                                                                     d_threshold=self.d_threshold_test_trigger,
                                                                     d_hysteresis=self.d_hysteresis_test_trigger,
                                                                     b_verbose=False, idx=0)
        self.assertAlmostEqual(d_eventtimes_sig[0], 1. / self.d_freq_law, 7)
        np_d_nx_sig = class_test_sig_features.calc_nx(np_d_sig=class_test_real.np_d_sig,
                                                      np_d_eventtimes=d_eventtimes_real,
                                                      b_verbose=False, idx=0)
        self.assertAlmostEqual(np.abs(np_d_nx_sig[0]), self.d_test_trigger_amp, 2)
        class_test_sig_features.plt_apht(str_plot_apht_desc='Signal feature class data ')

    def test_plt_polar(self):

        # Test real signal, falling signal
        class_test_real = appvib.ClSigReal(self.np_test_trigger, self.d_fs_test_trigger)
        d_eventtimes_real = class_test_real.np_d_est_triggers(np_sig_in=class_test_real.np_d_sig,
                                                              i_direction=self.i_direction_test_trigger_falling,
                                                              d_threshold=self.d_threshold_test_trigger,
                                                              d_hysteresis=self.d_hysteresis_test_trigger,
                                                              b_verbose=False)
        np_d_nx = class_test_real.calc_nx(np_sig_in=class_test_real.np_d_sig, np_d_eventtimes=d_eventtimes_real,
                                          b_verbose=False)
        self.assertAlmostEqual(np.abs(np_d_nx[0]), self.d_test_trigger_amp, 1)
        class_test_real.plt_polar()
        class_test_real.str_plot_polar_desc = 'Polar test data'
        class_test_real.ylim_apht_mag = [-0.1, 2.1]
        class_test_real.plt_polar()

        # Signal feature class test for apht plots. Also on the falling part of the signal
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_trigger, self.d_fs_test_trigger)
        d_eventtimes_sig = class_test_sig_features.np_d_est_triggers(np_d_sig=class_test_real.np_d_sig,
                                                                     i_direction=self.i_direction_test_trigger_falling,
                                                                     d_threshold=self.d_threshold_test_trigger,
                                                                     d_hysteresis=self.d_hysteresis_test_trigger,
                                                                     b_verbose=False, idx=0)
        self.assertAlmostEqual(d_eventtimes_sig[1] - d_eventtimes_sig[0], 1. / self.d_freq_law, 7)
        np_d_nx_sig = class_test_sig_features.calc_nx(np_d_sig=class_test_real.np_d_sig,
                                                      np_d_eventtimes=d_eventtimes_real,
                                                      b_verbose=False, idx=0)
        self.assertAlmostEqual(np.abs(np_d_nx_sig[0]), self.d_test_trigger_amp, 1)
        class_test_sig_features.plt_polar(str_plot_polar_desc='Signal feature class data ')

    def test_save_data(self):

        # Signal feature class test for a single data set
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_trigger, self.d_fs_test_trigger)
        class_test_sig_features.b_save_data(str_data_prefix='SignalFeatureTest')

        str_filename = class_test_sig_features.str_file
        print(str_filename)
        file_handle = open(str_filename)
        csv_reader = csv.reader(file_handle)
        csv_header = next(csv_reader)
        print(csv_header)
        file_handle.close()

        df_test = pd.read_csv(str_filename, header=None, skiprows=2, names=csv_header[0:5])
        for idx in range(class_test_sig_features.i_ns - 1):
            self.assertAlmostEqual(df_test.CH1[idx], self.np_test_trigger[idx], 8)
        # Be sure delta time and sampling frequency are coherent
        self.assertAlmostEqual(df_test['Delta Time'][0], 1.0 / df_test['Sampling Frequency'][0], 9)

        # Add a signal
        class_test_sig_features.idx_add_sig(self.np_test_trigger_ch2,
                                            d_fs=class_test_sig_features.d_fs(idx=0))
        class_test_sig_features.b_save_data(str_data_prefix='SignalFeatureTestCh2')

        # Read the CSV headers
        str_filename = class_test_sig_features.str_file
        print(str_filename)
        file_handle = open(str_filename)
        csv_reader = csv.reader(file_handle)
        csv_header = next(csv_reader)
        print(csv_header)
        file_handle.close()

        df_test_ch2 = pd.read_csv(str_filename, header=None, skiprows=2, names=csv_header[0:5])
        for idx in range(class_test_sig_features.i_ns - 1):
            self.assertAlmostEqual(df_test_ch2.CH2[idx], self.np_test_trigger_ch2[idx], 8)
        # Be sure delta time and sampling frequency are coherent
        self.assertAlmostEqual(df_test_ch2['Delta Time'][0], 1.0 / df_test_ch2['Sampling Frequency'][0], 9)


if __name__ == '__main__':
    unittest.main()
