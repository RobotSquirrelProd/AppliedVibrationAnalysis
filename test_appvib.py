import unittest
from unittest import TestCase
import appvib
import math
import numpy as np
import time
import pandas as pd
from datetime import datetime, timedelta, timezone


class TestClSig(TestCase):

    def setUp(self):
        # Define the initial values for the test
        self.np_test = np.array([0.1, 1.0, 10.0])
        self.np_test_ch2 = np.array([2.1, 3.0, 12.0])
        self.np_test_ch3 = np.array([3.1, 4.0, 13.0])

        # Data set for the real-valued class. This is a sawtooth waveform. For
        # triggering on rising values it should trigger at 7.5 seconds (between
        # sample 8 and 9). For falling edges it should trigger at 4.5 seconds (between
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
        self.np_test_trigger = self.d_test_trigger_amp * np.cos(2 * math.pi * self.d_freq_law * d_time_ext)
        self.d_test_trigger_amp_ch2 = 2.1
        self.np_test_trigger_ch2 = self.d_test_trigger_amp_ch2 * np.cos(2 * math.pi * self.d_freq_law * d_time_ext)
        self.d_threshold_test_trigger = 0.0
        self.d_hysteresis_test_trigger = 0.1
        self.i_direction_test_trigger_rising = 0
        self.i_direction_test_trigger_falling = 1

        # Data set for the signal feature class, nx phase test
        self.d_fs_test_trigger_ph = 2048
        i_ns = (self.d_fs_test_trigger_ph * 1)
        self.d_freq_law_ph = 5.
        self.d_test_trigger_amp_ph = 1.5
        d_time_ext = np.linspace(0, (i_ns - 1), i_ns) / float(self.d_fs_test_trigger_ph)
        self.d_phase_ph_ch1 = np.deg2rad(80)
        self.np_test_trigger_ph = self.d_test_trigger_amp_ph * np.cos(2 * math.pi * self.d_freq_law_ph * d_time_ext -
                                                                      self.d_phase_ph_ch1)
        self.d_test_trigger_amp_ph_ch2 = 3.1
        self.d_phase_ph_ch2 = np.deg2rad(33)
        self.np_test_trigger_ph_ch2 = self.d_test_trigger_amp_ph_ch2 * np.cos(2 * math.pi * self.d_freq_law_ph *
                                                                              d_time_ext - self.d_phase_ph_ch2)
        self.d_threshold_test_trigger_ph = 0.0
        self.d_hysteresis_test_trigger_ph = 0.1
        self.i_direction_test_trigger_rising_ph = 0
        self.i_direction_test_trigger_falling_ph = 1

        # Data set for the signal feature class event plots
        self.d_fs_test_plt_eventtimes = 2048
        i_ns = (self.d_fs_test_plt_eventtimes * 1)
        self.d_freq_law_test_plt_eventtimes = 5.
        self.d_amp_test_plt_eventtimes = 2.0
        d_time_ext_plt_eventtimes = np.linspace(0, (i_ns - 1), i_ns) / float(self.d_fs_test_plt_eventtimes)
        self.np_test_plt_eventtimes = self.d_amp_test_plt_eventtimes * np.sin(2 * math.pi *
                                                                              self.d_freq_law_test_plt_eventtimes *
                                                                              d_time_ext_plt_eventtimes)
        self.d_amp_ch2_test_plt_eventtimes = 0.5
        self.np_test_plt_eventtimes_ch2 = self.d_amp_ch2_test_plt_eventtimes * np.sin(2 * math.pi *
                                                                                      self.d_freq_law_test_plt_eventtimes *
                                                                                      d_time_ext_plt_eventtimes)
        self.d_threshold_test_plt_eventtimes = 1.0
        self.d_hysteresis_test_plt_eventtimes = 0.1
        self.i_direction_test_plt_eventtimes_rising = 0
        self.i_direction_test_plt_eventtimes_falling = 1

        self.np_test_comp = np.array([0.1 - 0.2j, 1.0 - 2.0j, 10.0 - 20j])
        self.np_test_comp_long = np.array([0.1 - 0.2j, 1.0 - 2.0j, 10.0 - 20j, 100.0 - 200j, 1000.0 - 2000j])
        self.ylim_tb_test = [-1.1, 1.1]
        self.ylim_tb_test_alt = [-3.3, 3.3]
        self.d_fs = 1.024e3
        self.d_fs_ch2 = 2.048e3
        self.d_fs_ch3 = 4.096e3
        self.str_eu_default = "volts"
        self.str_eu_acc = "g's"
        self.str_eu_vel = "ips"
        self.str_point_name = 'CH1 Test'
        self.str_point_name_ch2 = 'CH2 Test'
        self.str_machine_name = '7"  x 10" Mini Lathe | item number 93212 | serial no. 01504'

        # Test data set 000 : This one caused the nx_est to fail
        self.str_filename_000 = 'test_appvib_data_000.csv'
        self.df_test_000 = pd.read_csv(self.str_filename_000, header=None, skiprows=1,
                                       names=['Ch1', 'Ch2', 'FS'])
        self.np_d_test_data_000_Ch1 = self.df_test_000.Ch1
        self.np_d_test_data_000_Ch2 = self.df_test_000.Ch2
        self.d_fs_data_000 = self.df_test_000.FS[0]
        self.i_direction_test_000_trigger_slope = 0
        self.d_threshold_test_000 = 0.125

        # Test data set 001 : This one caused the nx_est to fail, no vectors
        self.str_filename_001 = 'test_appvib_data_001.csv'
        self.df_test_001 = pd.read_csv(self.str_filename_001, header=None, skiprows=1,
                                       names=['Ch1', 'Ch2', 'FS'])
        self.np_d_test_data_001_Ch1 = self.df_test_001.Ch1
        self.np_d_test_data_001_Ch2 = self.df_test_001.Ch2
        self.d_fs_data_001 = self.df_test_001.FS[0]
        self.i_direction_test_001_trigger_slope = 0
        self.d_threshold_test_001 = 0.125

        # Test values for finding index to the closest timestamp
        self.np_d_time_close_time = ([1.07754901e-01, 2.25514589e-01, 3.42310042e-01, 4.60151792e-01,
                                      5.77884125e-01, 6.94631708e-01, 8.12446104e-01, 6.04685230e+01,
                                      6.05901046e+01, 6.07109186e+01, 6.08323248e+01, 6.09531169e+01,
                                      1.26010386e+02, 1.26094825e+02, 1.26179829e+02, 1.26264849e+02,
                                      1.26349703e+02, 1.26434165e+02, 1.26519065e+02])
        self.dt_timestamp_close_time = datetime(2021, 12, 9, 5, 36, 10, 782000,
                                                tzinfo=timezone(timedelta(days=-1, seconds=57600)))
        self.dt_timestamp_close_time_mark = datetime(2021, 12, 9, 5, 37, 11, 203000,
                                                     tzinfo=timezone(timedelta(days=-1, seconds=57600)))

    def test_est_signal_features(self):

        # Test helper functions
        idx_test = appvib.ClassPlotSupport.get_idx_by_dt(self.np_d_time_close_time, self.dt_timestamp_close_time,
                                                         self.dt_timestamp_close_time_mark)
        print(idx_test)
        self.assertEqual(idx_test, 7)

        # Begin with amplitude estimation
        np_d_test = appvib.ClSignalFeaturesEst.np_d_est_amplitude(self.np_test_trigger_ph)
        self.assertAlmostEqual(float(np.mean(np_d_test)), self.d_test_trigger_amp_ph, 15)

        # Now for the rms estimation
        np_d_test_rms = appvib.ClSignalFeaturesEst.np_d_est_rms(self.np_test_trigger_ph)
        class_test_est = appvib.ClSigFeatures(self.np_test_trigger_ph, d_fs=self.d_fs_test_trigger_ph)
        class_test_est.ylim_tb([-1.7, 1.7], idx=0)
        class_test_est.plt_sigs()
        self.assertAlmostEqual(float(np.mean(np_d_test_rms)), self.d_test_trigger_amp_ph / np.sqrt(2.0), 4)

        # Mean estimation
        d_mean = 1.1
        np_d_mean_sig = self.np_test_trigger_ph + d_mean
        np_d_test_est_mean = appvib.ClSignalFeaturesEst.np_d_est_mean(np_d_mean_sig)
        class_test_est_mean = appvib.ClSigFeatures(np_d_mean_sig, d_fs=self.d_fs_test_trigger_ph)
        class_test_est_mean.d_threshold_update(d_mean, idx=0)
        class_test_est_mean.ylim_tb([-2.7, 2.7], idx=0)
        class_test_est_mean.dt_timestamp_mark_update(class_test_est_mean.dt_timestamp(idx=0) +
                                                     timedelta(seconds=0.5), idx=0)
        class_test_est_mean.plt_sigs()
        self.assertAlmostEqual(float(np.mean(np_d_test_est_mean)), d_mean, 2)

        # Custom sparklines
        i_ns_test = len(np_d_test_est_mean)
        np_d_sig_spark1 = np_d_test_est_mean + np.linspace(0, (i_ns_test - 1), i_ns_test) + \
                          np.random.normal(0, 100, i_ns_test)
        d_mean_max = max(np_d_sig_spark1)
        lst_fmt = appvib.ClassPlotSupport.get_plot_round(d_mean_max)
        str_point_spark1 = appvib.ClassPlotSupport.get_plot_sparkline_desc(lst_fmt[1],
                                                                           d_mean_max,
                                                                           'GOATS',
                                                                           'max')
        np_sparklines = np.array([appvib.ClSigCompUneven(np_d_sig_spark1, class_test_est_mean.d_time_plot(idx=0),
                                                         str_eu='GOATS', str_point_name=str_point_spark1,
                                                         str_machine_name=class_test_est_mean.str_machine_name(idx=0),
                                                         dt_timestamp=class_test_est_mean.dt_timestamp(idx=0))])
        np_sparklines[0].ylim_tb = [-300.0, 3000.0]
        class_test_est_mean.np_sparklines_update(np_sparklines, idx=0)
        class_test_est_mean.str_plot_desc = 'Test of custom sparkline'
        class_test_est_mean.plt_sigs()

    def test_b_complex(self):
        # Is the real-valued class setting the flags correctly?
        class_test_real = appvib.ClSigReal(self.np_test, self.d_fs)
        self.assertFalse(class_test_real.b_complex)

        # Are point names stored correctly in the real-valued object?
        class_test_real.str_point_name = self.str_point_name
        self.assertEqual(class_test_real.str_point_name, self.str_point_name)
        class_test_real = appvib.ClSigReal(self.np_test, self.d_fs, str_point_name=self.str_point_name)
        self.assertEqual(class_test_real.str_point_name, self.str_point_name)

        # Are machine names stored correctly in the real-valued object?
        class_test_real.str_machine_name = self.str_machine_name
        self.assertEqual(class_test_real.str_machine_name, self.str_machine_name)
        class_test_real = appvib.ClSigReal(self.np_test, self.d_fs, str_point_name=self.str_point_name,
                                           str_machine_name=self.str_machine_name)
        self.assertEqual(class_test_real.str_machine_name, self.str_machine_name)

        # Is the complex-valued class setting the flags correctly?
        class_test_comp = appvib.ClSigComp(self.np_test_comp, self.d_fs)
        self.assertTrue(class_test_comp.b_complex)
        self.assertFalse(class_test_real.b_complex)

        # Are point names stored correctly in the complex-valued object?
        class_test_comp.str_point_name = self.str_point_name
        self.assertEqual(class_test_comp.str_point_name, self.str_point_name)
        class_test_comp = appvib.ClSigComp(self.np_test, self.d_fs, str_point_name=self.str_point_name)
        self.assertEqual(class_test_comp.str_point_name, self.str_point_name)

        # Are machine names stored correctly in the complex-valued object?
        class_test_comp.str_machine_name = self.str_machine_name
        self.assertEqual(class_test_comp.str_machine_name, self.str_machine_name)
        class_test_comp = appvib.ClSigComp(self.np_test, self.d_fs, str_point_name=self.str_point_name,
                                           str_machine_name=self.str_machine_name)
        self.assertEqual(class_test_comp.str_machine_name, self.str_machine_name)

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

        # Are point names stored correctly in the signal feature object?
        class_test_sig_features.str_point_name_set(str_point_name=self.str_point_name, idx=0)
        self.assertEqual(class_test_sig_features.str_point_name(), self.str_point_name)
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs, str_point_name=self.str_point_name,
                                                       str_machine_name="Test")
        self.assertEqual(class_test_sig_features.str_point_name(), self.str_point_name)

        # Are machine names stored correctly in the signal feature object?
        class_test_sig_features.str_machine_name_set(str_machine_name=self.str_machine_name, idx=0)
        self.assertEqual(class_test_sig_features.str_machine_name(0), self.str_machine_name)
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs, str_point_name=self.str_point_name,
                                                       str_machine_name=self.str_machine_name)
        self.assertEqual(class_test_sig_features.str_machine_name(), self.str_machine_name)

        # Signal feature class, second signal
        idx_new = class_test_sig_features.idx_add_sig(self.np_test_ch2, self.d_fs, str_point_name='CH2')
        self.assertEqual(idx_new, 1, msg='Failed to return correct index')
        self.np_return = class_test_sig_features.get_np_d_sig(idx=1)
        self.assertAlmostEqual(self.np_test_ch2[0], self.np_return[0], 12)
        self.assertAlmostEqual(self.np_test[0], class_test_sig_features.np_d_sig[0], 12)

        # Signal feature class, third signal
        idx_new = class_test_sig_features.idx_add_sig(self.np_test_ch3, d_fs=self.d_fs_ch3, str_point_name='CH3')
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
            class_test_sig_features.idx_add_sig(self.np_test_ch2, d_fs=self.d_fs_ch2, str_point_name='CH2')
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
        idx_new = class_test_sig_features.idx_add_sig(self.np_test_ch2, self.d_fs, str_point_name='CH2')
        self.assertEqual(idx_new, 1, msg='Failed to return correct index')
        class_test_sig_features.d_fs_update(self.d_fs_ch2, idx=1)

    def test_str_eu(self):
        # Signal feature base class unit check
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        self.assertEqual(self.str_eu_default, class_test_sig_features.str_eu())
        class_test_sig_features.str_eu_set(str_eu=self.str_eu_acc)
        self.assertEqual(self.str_eu_acc, class_test_sig_features.str_eu())

        # Add second signal and set point name
        idx_ch2 = class_test_sig_features.idx_add_sig(np_d_sig=self.np_test_ch2, d_fs=self.d_fs_ch2,
                                                      str_point_name=self.str_point_name_ch2)
        class_test_sig_features.str_eu_set(str_eu=self.str_eu_vel, idx=idx_ch2)
        self.assertEqual(self.str_eu_vel, class_test_sig_features.str_eu(idx=idx_ch2))

    def test_str_point_name(self):
        # Signal feature base class signal point name check
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        class_test_sig_features.str_point_name_set(str_point_name=self.str_point_name)
        self.assertEqual(self.str_point_name, class_test_sig_features.str_point_name())

        # Add second signal and set point name
        idx_ch2 = class_test_sig_features.idx_add_sig(np_d_sig=self.np_test_ch2, d_fs=self.d_fs_ch2,
                                                      str_point_name=self.str_point_name_ch2)
        class_test_sig_features.str_point_name_set(str_point_name=self.str_point_name_ch2, idx=idx_ch2)
        self.assertEqual(self.str_point_name_ch2, class_test_sig_features.str_point_name(idx=idx_ch2))

    def test_plt_sigs(self):
        # Signal feature class check of plotting on instantiation
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        class_test_sig_features.str_plot_desc = 'test_plt_sigs | CLSigFeatures | Defaults'
        class_test_sig_features.str_machine_name_set('Harness')
        class_test_sig_features.plt_sigs()

        # Signal feature class, second signal auto y-limits
        idx_new = class_test_sig_features.idx_add_sig(self.np_test_ch2, self.d_fs, str_point_name='CH2')
        class_test_sig_features.str_plot_desc = 'test_plt_sigs | CLSigFeatures | 2nd Point'
        self.assertEqual(idx_new, 1, msg='Failed to return correct index')
        class_test_sig_features.plt_sigs()

        # Signal feature class, second signal manual y-limits, new data
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_trigger_ph, self.d_fs_test_trigger_ph)
        class_test_sig_features.ylim_tb(ylim_tb_in=[-16.0, 16.0], idx=0)
        idx_new = class_test_sig_features.idx_add_sig(self.np_test_trigger_ph_ch2, self.d_fs_test_trigger_ph,
                                                      str_point_name='CH2')
        class_test_sig_features.ylim_tb(ylim_tb_in=[-16.0, 16.0], idx=1)
        class_test_sig_features.str_plot_desc = 'test_plt_sigs | CLSigFeatures | New data, y-limits'
        class_test_sig_features.plt_sigs()

        class_test_sig_features.str_plot_desc = 'test_plt_sigs | CLSigFeatures | SG Filtered'
        class_test_sig_features.plt_sigs(b_plot_sg=True)

        class_test_sig_features.str_plot_desc = 'test_plt_sigs | CLSigFeatures | FIR Filtered'
        class_test_sig_features.plt_sigs(b_plot_filt=True)

        class_test_sig_features.str_plot_desc = 'test_plt_sigs | CLSigFeatures | All Filtered'
        class_test_sig_features.plt_sigs(b_plot_sg=True, b_plot_filt=True)

    def test_plt_spec(self):
        # Signal feature class check of plotting on instantiation
        class_test_sig_features = appvib.ClSigFeatures(self.np_test, self.d_fs)
        class_test_sig_features.str_plot_desc = 'test_plt_spec | CLSigFeatures | Defaults'
        class_test_sig_features.plt_spec()

        # Add peak label
        class_test_sig_features.b_spec_peak = True
        class_test_sig_features.str_plot_desc = 'test_plt_spec | CLSigFeatures | Defaults w/ Peak Label'
        class_test_sig_features.plt_spec()

        # Signal feature class, second signal manual y-limits, new data
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_trigger, self.d_fs_test_trigger)
        class_test_sig_features.ylim_tb(ylim_tb_in=[-16.0, 16.0], idx=0)
        idx_new = class_test_sig_features.idx_add_sig(self.np_test_trigger_ch2, self.d_fs_test_trigger,
                                                      str_point_name='CH2')
        class_test_sig_features.ylim_tb(ylim_tb_in=[-16.0, 16.0], idx=1)
        class_test_sig_features.b_spec_peak = True
        class_test_sig_features.str_plot_desc = 'test_plt_spec | CLSigFeatures | New data, y-limits'
        class_test_sig_features.plt_spec()

    def test_np_d_est_triggers(self):

        # Real-valued check, rising signal, explicitly defining the arguments
        class_test_real = appvib.ClSigReal(self.np_test_real, self.d_fs_real)
        class_test_real.np_d_est_triggers(np_d_sig=self.np_test_real, i_direction=self.i_direction_real_rising,
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
        class_test_real.np_d_est_triggers(np_d_sig=self.np_test_real, i_direction=self.i_direction_real_falling,
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

    def test_plt_eventtimes(self):

        # Signal feature class test, rising signal with threshold of 0.5
        class_test_plt_eventtimes = appvib.ClSigFeatures(self.np_test_plt_eventtimes, self.d_fs_test_plt_eventtimes)
        print('Signal frequency, hertz: ' + '%0.6f' % self.d_freq_law_test_plt_eventtimes)
        class_test_plt_eventtimes.np_d_est_triggers(np_d_sig=self.np_test_plt_eventtimes,
                                                    i_direction=self.i_direction_test_plt_eventtimes_rising,
                                                    d_threshold=self.d_threshold_test_plt_eventtimes,
                                                    d_hysteresis=self.d_hysteresis_test_plt_eventtimes,
                                                    b_verbose=False)
        class_test_plt_eventtimes.str_plot_desc = 'plt_eventtimes test (single)'
        class_test_plt_eventtimes.str_point_name_set(str_point_name='CX1', idx=0)
        d_est_freq = 1. / (np.mean(np.diff(class_test_plt_eventtimes.np_d_eventtimes())))
        self.assertAlmostEqual(d_est_freq, self.d_freq_law_test_plt_eventtimes, 5)

        # check the plot for a single channel
        class_test_plt_eventtimes.plt_eventtimes()

        # Add a second channel and plot those events
        class_test_plt_eventtimes.idx_add_sig(np_d_sig=self.np_test_plt_eventtimes_ch2,
                                              d_fs=self.d_fs_test_plt_eventtimes, str_point_name='CX2')
        class_test_plt_eventtimes.str_plot_desc = 'plt_eventtimes test (second channel)'
        class_test_plt_eventtimes.ylim_tb(ylim_tb_in=[-16.0, 16.0], idx=0)
        class_test_plt_eventtimes.plt_eventtimes(idx_eventtimes=0, idx=1)

    def test_plt_rpm(self):
        # Signal feature class test, rising signal with threshold of 0.5
        class_test_plt_rpm = appvib.ClSigFeatures(self.np_test_plt_eventtimes, self.d_fs_test_plt_eventtimes)
        class_test_plt_rpm.np_d_est_triggers(np_d_sig=self.np_test_plt_eventtimes,
                                             i_direction=self.i_direction_test_plt_eventtimes_rising,
                                             d_threshold=self.d_threshold_test_plt_eventtimes,
                                             d_hysteresis=self.d_hysteresis_test_plt_eventtimes,
                                             b_verbose=False)
        class_test_plt_rpm.str_plot_desc = 'test_plt_rpm | Single Channel'
        class_test_plt_rpm.str_point_name_set(str_point_name='CX1', idx=0)
        d_est_freq = 1. / (np.mean(np.diff(class_test_plt_rpm.np_d_eventtimes())))
        self.assertAlmostEqual(d_est_freq, self.d_freq_law_test_plt_eventtimes, 5)

        # check the plot for a single channel
        lst_plot = class_test_plt_rpm.plt_rpm()
        np_d_rpm = lst_plot[1]
        self.assertAlmostEqual(float(np.mean(np_d_rpm)), self.d_freq_law_test_plt_eventtimes * 60.0, 3)

    def test_nX_est(self):

        # Test real signal, rising signal
        class_test_real = appvib.ClSigReal(self.np_test_trigger, self.d_fs_test_trigger)
        d_eventtimes_real = class_test_real.np_d_est_triggers(np_d_sig=class_test_real.np_d_sig,
                                                              i_direction=self.i_direction_test_trigger_rising,
                                                              d_threshold=self.d_threshold_test_trigger,
                                                              d_hysteresis=self.d_hysteresis_test_trigger,
                                                              b_verbose=False)
        np_d_nx = class_test_real.calc_nx(np_d_sig=class_test_real.np_d_sig, np_d_eventtimes=d_eventtimes_real,
                                          b_verbose=False)
        self.assertAlmostEqual(np.abs(np_d_nx[0]), self.d_test_trigger_amp, 2)
        class_test_real.str_plot_desc = 'test_plt_apht | ClSigReal | Initial call'
        class_test_real.plt_apht(b_verbose=True)
        class_test_real.str_plot_desc = 'test_plt_apht | ClSigReal | Test call'
        class_test_real.ylim_apht_mag = [-0.1, 1.1]
        class_test_real.plt_apht()

        # Test base class
        class_test_uneven = appvib.ClSigCompUneven(np_d_nx, class_test_real.np_d_eventtimes, str_eu='cat whiskers',
                                                   str_point_name='CATFISH')
        class_test_uneven.plt_apht()

        # Signal feature class test for apht plots
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_trigger, self.d_fs_test_trigger)
        d_eventtimes_sig = class_test_sig_features.np_d_est_triggers(np_d_sig=class_test_sig_features.np_d_sig,
                                                                     i_direction=self.i_direction_test_trigger_rising,
                                                                     d_threshold=self.d_threshold_test_trigger,
                                                                     d_hysteresis=self.d_hysteresis_test_trigger,
                                                                     b_verbose=False, idx=0)
        self.assertAlmostEqual(d_eventtimes_sig[1] - d_eventtimes_sig[0], 1. / self.d_freq_law, 7)
        np_d_nx_sig = class_test_sig_features.calc_nx(np_d_sig=class_test_sig_features.np_d_sig,
                                                      np_d_eventtimes=d_eventtimes_sig,
                                                      b_verbose=False, idx=0)
        self.assertAlmostEqual(np.abs(np_d_nx_sig[0]), self.d_test_trigger_amp, 2)
        class_test_sig_features.plt_apht(str_plot_apht_desc='test_nX_est ClSigFeatures')

    def test_plt_nx(self):

        # Signal feature class test for nx plots, start with implicit calls
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_trigger, self.d_fs_test_trigger)
        class_test_sig_features.plt_nx()
        d_eventtimes_sig = class_test_sig_features.np_d_est_triggers(np_d_sig=class_test_sig_features.np_d_sig,
                                                                     i_direction=self.i_direction_test_trigger_rising,
                                                                     d_threshold=self.d_threshold_test_trigger,
                                                                     d_hysteresis=self.d_hysteresis_test_trigger,
                                                                     b_verbose=False)
        self.assertAlmostEqual(d_eventtimes_sig[1] - d_eventtimes_sig[0], 1. / self.d_freq_law, 7)

        # More plots and more implicit calls
        class_test_sig_features.str_plot_desc = 'test_plt_nx ClSigFeatures eventtimes'
        class_test_sig_features.plt_eventtimes()
        d_eventtimes_sig = class_test_sig_features.np_d_est_triggers(np_d_sig=class_test_sig_features.np_d_sig,
                                                                     i_direction=self.i_direction_test_trigger_rising,
                                                                     d_threshold=self.d_threshold_test_trigger,
                                                                     d_hysteresis=self.d_hysteresis_test_trigger,
                                                                     b_verbose=False)
        # This sequence caught mis-management of plot titles.
        class_test_sig_features.plt_nx()
        class_test_sig_features.plt_nx()
        class_test_sig_features.plt_nx(str_plot_desc='test_plt_nx ClSigFeatures Implicit call complete')

        np_d_nx_sig = class_test_sig_features.calc_nx(np_d_sig=class_test_sig_features.np_d_sig,
                                                      np_d_eventtimes=d_eventtimes_sig,
                                                      b_verbose=False, idx=0)
        self.assertAlmostEqual(np.abs(np_d_nx_sig[0]), self.d_test_trigger_amp, 2)
        class_test_sig_features.plt_nx(str_plot_desc='test_plt_nx ClSigFeatures Explicit call complete')

    # Validate phase
    def test_plt_nx_phase(self):

        # Test phase for a single channel
        class_phase = appvib.ClSigFeatures(self.np_test_trigger_ph, d_fs=self.d_fs_test_trigger_ph)
        class_phase.np_d_est_triggers(class_phase.np_d_sig, i_direction=self.i_direction_test_trigger_rising,
                                      d_threshold=self.np_test_trigger_ph[0])
        np_d_eventtimes = class_phase.np_d_eventtimes()
        np_d_nx = class_phase.calc_nx(class_phase.np_d_sig, np_d_eventtimes)
        class_phase.str_plot_desc = 'test_plt_nx_phase | Single channel'
        class_phase.plt_eventtimes()
        class_phase.plt_nx()
        self.assertAlmostEqual(np.angle(np_d_nx[0]), self.d_phase_ph_ch1, 1)
        class_phase.idx_add_sig(np_d_sig=self.np_test_trigger_ph_ch2, d_fs=self.d_fs_test_trigger_ph,
                                str_point_name='CH2')
        np_d_eventtimes = class_phase.np_d_eventtimes()
        np_d_nx = class_phase.calc_nx(class_phase.np_d_sig, np_d_eventtimes, idx=0)
        self.assertAlmostEqual(np.rad2deg(np.angle(np_d_nx[0])), np.rad2deg(self.d_phase_ph_ch1), 0)
        class_phase.plt_nx()

    # Tests targeted to behavior discovered in specific data sets
    def test_plt_nx_001(self):

        # This data set caused no vectors to be found, re-worked
        # to handle this gracefully
        class_001 = appvib.ClSigFeatures(self.np_d_test_data_001_Ch1, self.d_fs_data_001)
        class_001.idx_add_sig(self.np_d_test_data_001_Ch2, self.d_fs_data_001, str_point_name='CH2')
        np_d_eventtimes = class_001.np_d_est_triggers(np_d_sig=class_001.np_d_sig,
                                                      i_direction=self.i_direction_test_001_trigger_slope,
                                                      d_threshold=self.d_threshold_test_001)
        print(np_d_eventtimes)
        class_001.plt_nx(str_plot_desc='Unique eventtimes and nx vectors')

    def test_plt_apht(self):

        # Test real signal, rising signal
        class_test_real = appvib.ClSigReal(self.np_test_trigger, self.d_fs_test_trigger)
        d_eventtimes_real = class_test_real.np_d_est_triggers(np_d_sig=class_test_real.np_d_sig,
                                                              i_direction=self.i_direction_test_trigger_rising,
                                                              d_threshold=self.d_threshold_test_trigger,
                                                              d_hysteresis=self.d_hysteresis_test_trigger,
                                                              b_verbose=False)
        np_d_nx = class_test_real.calc_nx(np_d_sig=class_test_real.np_d_sig, np_d_eventtimes=d_eventtimes_real,
                                          b_verbose=False)
        self.assertAlmostEqual(np.abs(np_d_nx[0]), self.d_test_trigger_amp, 2)
        class_test_real.str_plot_desc = 'test_plt_apht | ClSigReal | Initial call'
        class_test_real.plt_apht()
        class_test_real.str_plot_desc = 'test_plt_apht | ClSigReal | Test call'
        class_test_real.ylim_apht_mag = [-0.1, 1.1]
        class_test_real.plt_apht()

        # Test base class
        class_test_uneven = appvib.ClSigCompUneven(np_d_nx, class_test_real.np_d_eventtimes, str_eu='cat whiskers',
                                                   str_point_name='TUNA')
        class_test_uneven.plt_apht()

        # Signal feature class test for apht plots
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_trigger, self.d_fs_test_trigger)
        d_eventtimes_sig = class_test_sig_features.np_d_est_triggers(np_d_sig=class_test_real.np_d_sig,
                                                                     i_direction=self.i_direction_test_trigger_rising,
                                                                     d_threshold=self.d_threshold_test_trigger,
                                                                     d_hysteresis=self.d_hysteresis_test_trigger,
                                                                     b_verbose=False, idx=0)
        self.assertAlmostEqual(d_eventtimes_sig[1] - d_eventtimes_sig[0], 1. / self.d_freq_law, 7)
        np_d_nx_sig = class_test_sig_features.calc_nx(np_d_sig=class_test_real.np_d_sig,
                                                      np_d_eventtimes=d_eventtimes_real,
                                                      b_verbose=False, idx=0)
        self.assertAlmostEqual(np.abs(np_d_nx_sig[0]), self.d_test_trigger_amp, 2)
        class_test_sig_features.plt_apht(str_plot_apht_desc='Signal feature class data ')

    def test_plt_polar(self):

        # Test real signal, falling signal
        class_test_real = appvib.ClSigReal(self.np_test_trigger, self.d_fs_test_trigger)
        d_eventtimes_real = class_test_real.np_d_est_triggers(np_d_sig=class_test_real.np_d_sig,
                                                              i_direction=self.i_direction_test_trigger_falling,
                                                              d_threshold=self.d_threshold_test_trigger,
                                                              d_hysteresis=self.d_hysteresis_test_trigger,
                                                              b_verbose=False)
        np_d_nx = class_test_real.calc_nx(np_d_sig=class_test_real.np_d_sig, np_d_eventtimes=d_eventtimes_real,
                                          b_verbose=False)
        self.assertAlmostEqual(np.abs(np_d_nx[0]), self.d_test_trigger_amp, 1)
        class_test_real.plt_polar()
        class_test_real.str_plot_desc = 'Polar test data'
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
        class_test_sig_features.plt_polar(str_plot_desc='Signal feature class data ')

    def test_save_read_data(self):

        # Signal feature class test for a single data set
        dt_local = datetime.now()
        class_test_sig_features = appvib.ClSigFeatures(self.np_test_trigger, self.d_fs_test_trigger,
                                                       dt_timestamp=dt_local)
        class_test_sig_features.plt_sigs()
        class_test_sig_features.b_save_data(str_data_prefix='SignalFeatureTest')
        print("Testing file: " + class_test_sig_features.str_file)
        lst_file = class_test_sig_features.b_read_data_as_df(str_filename=class_test_sig_features.str_file)
        # Extract the data frame
        df_test = lst_file[0]
        dt_test = lst_file[1]
        d_fs_test = lst_file[2]
        d_delta_t_test = lst_file[3]

        for idx in range(class_test_sig_features.i_ns - 1):
            self.assertAlmostEqual(df_test.CH1[idx], self.np_test_trigger[idx], 8)

        # Be sure timestamps, delta time and sampling frequency are coherent
        self.assertEqual(dt_local.day, dt_test[0].day)
        self.assertEqual(dt_local.hour, dt_test[0].hour)
        self.assertEqual(dt_local.minute, dt_test[0].minute)
        self.assertEqual(dt_local.second, dt_test[0].second)
        self.assertAlmostEqual(d_fs_test[0], class_test_sig_features.d_fs(idx=0), 9)
        self.assertAlmostEqual(1.0 / d_fs_test[0], d_delta_t_test[0], 9)

        # Add a signal, save it, bring it back in
        time.sleep(1)
        dt_local_ch2 = datetime.now()
        class_test_sig_features.idx_add_sig(self.np_test_trigger_ch2,
                                            d_fs=class_test_sig_features.d_fs(idx=0), str_point_name='CH2',
                                            dt_timestamp=dt_local_ch2)
        class_test_sig_features.b_save_data(str_data_prefix='SignalFeatureTestCh2')
        lst_file = class_test_sig_features.b_read_data_as_df(str_filename=class_test_sig_features.str_file)
        # Extract the data frame
        df_test_ch2 = lst_file[0]
        dt_test_ch2 = lst_file[1]
        d_fs_test_ch2 = lst_file[2]
        d_delta_t_test_ch2 = lst_file[3]

        for idx in range(class_test_sig_features.i_ns - 1):
            self.assertAlmostEqual(df_test_ch2.CH2[idx], self.np_test_trigger_ch2[idx], 8)

        # Be sure timestamps, delta time and sampling frequency are coherent
        for idx, _ in enumerate(d_fs_test_ch2):
            self.assertEqual(dt_local_ch2.day, dt_test_ch2[1].day)
            self.assertEqual(dt_local_ch2.hour, dt_test_ch2[1].hour)
            self.assertEqual(dt_local_ch2.minute, dt_test_ch2[1].minute)
            self.assertEqual(dt_local_ch2.second, dt_test_ch2[1].second)
            self.assertAlmostEqual(d_fs_test_ch2[idx], class_test_sig_features.d_fs(idx=idx), 9)
            self.assertAlmostEqual(1.0 / d_fs_test_ch2[idx], d_delta_t_test_ch2[idx], 9)

        # Surfaced this defect in a stand-alone plotting workbook
        class_file_test = appvib.ClSigFeatures([1., 2., 3.], 1.)
        lst_file = class_file_test.b_read_data_as_df(str_filename=class_test_sig_features.str_file)
        # Extract the data frame et. al.
        df_test_file = lst_file[0]

        for idx in range(class_file_test.i_ns - 1):
            self.assertAlmostEqual(df_test_file.CH2[idx], self.np_test_trigger_ch2[idx], 8)


if __name__ == '__main__':
    unittest.main()
