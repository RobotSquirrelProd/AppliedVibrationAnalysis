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
        self.d_test_trigger_amp_ph = 1.65
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

        # This one caused the plotting to crash due to indexing over-run
        # in the rms function
        self.str_filename_002 = 'Free Free, no damping'
        self.d_fs_data_002 = 1.0
        self.np_d_test_data_002 = np.array([2.00000000, 2.0078732 , 2.01574152, 2.02360008, 2.03144402,
            2.03926845, 2.04706854, 2.05483945, 2.06257636, 2.07027448,
            2.07792903, 2.08553526, 2.09308846, 2.10058396, 2.10801709,
            2.11538326, 2.12267789, 2.12989647, 2.13703451, 2.1440876 ,
            2.15105135, 2.15792146, 2.16469367, 2.17136377, 2.17792762,
            2.18438117, 2.19072041, 2.19694141, 2.20304031, 2.20901333,
            2.21485677, 2.220567  , 2.22614049, 2.23157378, 2.2368635 ,
            2.24200638, 2.24699921, 2.25183892, 2.25652249, 2.26104702,
            2.26540972, 2.26960787, 2.27363887, 2.27750022, 2.28118953,
            2.28470451, 2.28804299, 2.29120289, 2.29418224, 2.29697922,
            2.29959208, 2.3020192 , 2.30425908, 2.30631032, 2.30817166,
            2.30984195, 2.31132014, 2.31260533, 2.31369671, 2.31459361,
            2.31529546, 2.31580185, 2.31611245, 2.31622706, 2.31614563,
            2.3158682 , 2.31539493, 2.31472614, 2.31386222, 2.31280371,
            2.31155128, 2.3101057 , 2.30846786, 2.30663878, 2.30461959,
            2.30241155, 2.30001602, 2.29743449, 2.29466856, 2.29171995,
            2.28859048, 2.28528209, 2.28179683, 2.27813687, 2.27430448,
            2.27030202, 2.26613199, 2.26179696, 2.25729962, 2.25264277,
            2.24782928, 2.24286215, 2.23774445, 2.23247936, 2.22707014,
            2.22152014, 2.21583281, 2.21001166, 2.20406032, 2.19798246,
            2.19178187, 2.18546237, 2.17902789, 2.17248242, 2.16583002,
            2.1590748 , 2.15222097, 2.14527276, 2.13823449, 2.13111051,
            2.12390526, 2.11662318, 2.1092688 , 2.10184668, 2.09436142,
            2.08681766, 2.07922007, 2.07157336, 2.06388229, 2.05615161,
            2.04838612, 2.04059062, 2.03276997, 2.024929  , 2.01707257,
            2.00920556, 2.00133284, 1.99345929])

        # This one also caused the plotting to crash 
        self.str_filename_003 = 'Free Free, no damping'
        self.d_fs_data_003 = 1.0
        self.np_d_test_data_003 = np.array([ 0.        ,  0.00195693,  0.00391379,  0.0058705 ,  0.00782699,
        0.00978317,  0.01173899,  0.01369435,  0.01564918,  0.01760342,
        0.01955698,  0.0215098 ,  0.02346179,  0.02541288,  0.027363  ,
        0.02931207,  0.03126002,  0.03320677,  0.03515225,  0.03709638,
        0.03903909,  0.04098031,  0.04291996,  0.04485796,  0.04679425,
        0.04872875,  0.05066137,  0.05259206,  0.05452074,  0.05644732,
        0.05837175,  0.06029394,  0.06221382,  0.06413131,  0.06604636,
        0.06795887,  0.06986878,  0.07177601,  0.0736805 ,  0.07558216,
        0.07748093,  0.07937673,  0.08126949,  0.08315914,  0.08504561,
        0.08692881,  0.08880869,  0.09068517,  0.09255818,  0.09442764,
        0.09629348,  0.09815564,  0.10001403,  0.1018686 ,  0.10371927,
        0.10556596,  0.10740861,  0.10924715,  0.1110815 ,  0.1129116 ,
        0.11473738,  0.11655876,  0.11837568,  0.12018807,  0.12199585,
        0.12379896,  0.12559733,  0.12739089,  0.12917957,  0.1309633 ,
        0.13274202,  0.13451565,  0.13628413,  0.1380474 ,  0.13980538,
        0.141558  ,  0.1433052 ,  0.14504691,  0.14678307,  0.14851361,
        0.15023846,  0.15195756,  0.15367084,  0.15537823,  0.15707967,
        0.1587751 ,  0.16046444,  0.16214764,  0.16382463,  0.16549535,
        0.16715973,  0.16881771,  0.17046922,  0.17211421,  0.1737526 ,
        0.17538434,  0.17700936,  0.1786276 ,  0.18023901,  0.18184351,
        0.18344104,  0.18503155,  0.18661498,  0.18819126,  0.18976033,
        0.19132213,  0.19287661,  0.1944237 ,  0.19596335,  0.19749549,
        0.19902006,  0.20053702,  0.20204629,  0.20354783,  0.20504158,
        0.20652747,  0.20800545,  0.20947546,  0.21093746,  0.21239137,
        0.21383715,  0.21527475,  0.21670409,  0.21812514,  0.21953784,
        0.22094213,  0.22233795,  0.22372527,  0.22510401,  0.22647413,
        0.22783558,  0.22918831,  0.23053226,  0.23186738,  0.23319362,
        0.23451093,  0.23581926,  0.23711855,  0.23840877,  0.23968986,
        0.24096176,  0.24222444,  0.24347785,  0.24472193,  0.24595663,
        0.24718192,  0.24839774,  0.24960405,  0.2508008 ,  0.25198795,
        0.25316544,  0.25433324,  0.2554913 ,  0.25663958,  0.25777802,
        0.2589066 ,  0.26002526,  0.26113396,  0.26223266,  0.26332132,
        0.2643999 ,  0.26546835,  0.26652663,  0.2675747 ,  0.26861253,
        0.26964007,  0.27065729,  0.27166414,  0.27266059,  0.27364659,
        0.27462212,  0.27558712,  0.27654158,  0.27748544,  0.27841868,
        0.27934125,  0.28025313,  0.28115427,  0.28204465,  0.28292423,
        0.28379297,  0.28465084,  0.28549781,  0.28633385,  0.28715892,
        0.287973  ,  0.28877604,  0.28956803,  0.29034893,  0.29111871,
        0.29187734,  0.29262479,  0.29336104,  0.29408605,  0.2947998 ,
        0.29550226,  0.2961934 ,  0.2968732 ,  0.29754163,  0.29819867,
        0.29884428,  0.29947845,  0.30010116,  0.30071237,  0.30131206,
        0.30190021,  0.30247681,  0.30304182,  0.30359522,  0.304137  ,
        0.30466713,  0.30518559,  0.30569236,  0.30618743,  0.30667077,
        0.30714237,  0.3076022 ,  0.30805026,  0.30848652,  0.30891096,
        0.30932357,  0.30972434,  0.31011325,  0.31049028,  0.31085542,
        0.31120865,  0.31154997,  0.31187935,  0.3121968 ,  0.31250228,
        0.3127958 ,  0.31307734,  0.31334689,  0.31360444,  0.31384998,
        0.3140835 ,  0.31430499,  0.31451445,  0.31471186,  0.31489722,
        0.31507051,  0.31523175,  0.31538091,  0.31551799,  0.31564299,
        0.3157559 ,  0.31585672,  0.31594544,  0.31602207,  0.31608659,
        0.31613901,  0.31617932,  0.31620752,  0.31622361,  0.31622759,
        0.31621946,  0.31619922,  0.31616687,  0.31612241,  0.31606585,
        0.31599718,  0.31591642,  0.31582355,  0.31571859,  0.31560153,
        0.3154724 ,  0.31533117,  0.31517788,  0.31501251,  0.31483508,
        0.31464559,  0.31444406,  0.31423048,  0.31400486,  0.31376722,
        0.31351757,  0.31325591,  0.31298225,  0.31269661,  0.31239899,
        0.31208941,  0.31176787,  0.3114344 ,  0.311089  ,  0.31073168,
        0.31036247,  0.30998137,  0.3095884 ,  0.30918357,  0.3087669 ,
        0.30833841,  0.30789811,  0.30744602,  0.30698215,  0.30650653,
        0.30601917,  0.30552009,  0.30500931,  0.30448685,  0.30395273,
        0.30340696,  0.30284958,  0.3022806 ,  0.30170005,  0.30110794,
        0.3005043 ,  0.29988915,  0.29926252,  0.29862442,  0.29797489,
        0.29731395,  0.29664162,  0.29595793,  0.29526291,  0.29455658,
        0.29383897,  0.29311011,  0.29237002,  0.29161874,  0.29085628,
        0.29008269,  0.28929799,  0.28850221,  0.28769538,  0.28687754,
        0.28604871,  0.28520892,  0.28435821,  0.28349661,  0.28262416,
        0.28174088,  0.28084681,  0.27994198,  0.27902644,  0.27810021,
        0.27716333,  0.27621583,  0.27525776,  0.27428914,  0.27331002,
        0.27232044,  0.27132042,  0.27031002,  0.26928926,  0.26825819,
        0.26721685,  0.26616527,  0.2651035 ,  0.26403158,  0.26294955,
        0.26185744,  0.26075531,  0.25964319,  0.25852113,  0.25738917,
        0.25624735,  0.25509572,  0.25393432,  0.25276319,  0.25158239,
        0.25039195,  0.24919192,  0.24798234,  0.24676328,  0.24553476,
        0.24429683,  0.24304955,  0.24179297,  0.24052712,  0.23925207,
        0.23796785,  0.23667451,  0.23537212,  0.2340607 ,  0.23274033,
        0.23141104,  0.23007289,  0.22872593,  0.22737021,  0.22600579,
        0.2246327 ,  0.22325102,  0.22186078,  0.22046205,  0.21905488,
        0.21763931,  0.21621542,  0.21478324,  0.21334284,  0.21189426,
        0.21043757,  0.20897283,  0.20750007,  0.20601938,  0.20453079,
        0.20303437,  0.20153018,  0.20001826,  0.19849869,  0.19697152,
        0.1954368 ,  0.1938946 ,  0.19234497,  0.19078797,  0.18922367,
        0.18765213,  0.18607339,  0.18448753,  0.18289461,  0.18129468,
        0.17968781,  0.17807406,  0.17645349,  0.17482616,  0.17319213,
        0.17155147,  0.16990424,  0.16825051,  0.16659033,  0.16492378,
        0.1632509 ,  0.16157178,  0.15988646,  0.15819503,  0.15649753,
        0.15479405,  0.15308463,  0.15136935,  0.14964828,  0.14792147,
        0.146189  ,  0.14445093,  0.14270733,  0.14095826,  0.1392038 ,
        0.137444  ,  0.13567894,  0.13390869,  0.1321333 ,  0.13035286,
        0.12856743,  0.12677707,  0.12498185,  0.12318185,  0.12137713,
        0.11956777,  0.11775382,  0.11593537,  0.11411247,  0.11228521,
        0.11045364,  0.10861785,  0.10677789,  0.10493385,  0.10308578,
        0.10123377,  0.09937789,  0.09751819,  0.09565477,  0.09378768,
        0.09191699,  0.09004279,  0.08816514,  0.08628411,  0.08439978,
        0.08251222,  0.08062149,  0.07872768,  0.07683086,  0.07493109,
        0.07302845,  0.07112301,  0.06921486,  0.06730405,  0.06539066,
        0.06347477,  0.06155645,  0.05963577,  0.0577128 ,  0.05578763,
        0.05386032,  0.05193095,  0.04999959,  0.04806631,  0.04613119,
        0.04419431,  0.04225574,  0.04031554,  0.0383738 ,  0.0364306 ,
        0.03448599,  0.03254007,  0.0305929 ,  0.02864456,  0.02669512,
        0.02474466,  0.02279325,  0.02084097,  0.01888789,  0.01693409,
        0.01497964,  0.01302461,  0.01106909,  0.00911314,  0.00715685,
        0.00520028,  0.00324351,  0.00128661, -0.00067033, -0.00262725,
       -0.00458407, -0.00654071])

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
        self.assertEqual(idx_test, 7)

        # Begin with amplitude estimation
        np_d_test = appvib.ClSignalFeaturesEst.np_d_est_amplitude(self.np_test_trigger_ph)
        self.assertAlmostEqual(float(np.mean(np_d_test)), self.d_test_trigger_amp_ph, 14)

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
        class_test_est_mean.ylim_tb([-2.8, 2.8], idx=0)
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

        # This use case caused the code to crash in the rolling rms function
        class_test_sig_features = appvib.ClSigFeatures(self.np_d_test_data_002, self.d_fs_data_002 )
        class_test_sig_features.str_plot_desc = 'test_plt_sigs | CLSigFeatures | Harmonic motion'
        class_test_sig_features.str_machine_name_set('Harness')
        class_test_sig_features.plt_sigs()

        # This use case caused the code to crash
        class_test_sig_features = appvib.ClSigFeatures(self.np_d_test_data_003, self.d_fs_data_003 )
        class_test_sig_features.str_plot_desc = 'test_plt_sigs | CLSigFeatures | Harmonic motion2'
        class_test_sig_features.str_machine_name_set('Harness')
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
        class_test_plt_eventtimes.plt_eventtimes(idx_eventtimes=0, idx_ch=1)

        # Check the plot for a single channel, but with threshold set too high
        class_test_plt_eventtimes_err = appvib.ClSigFeatures(self.np_test_plt_eventtimes, self.d_fs_test_plt_eventtimes)
        print('Signal frequency, hertz: ' + '%0.6f' % self.d_freq_law_test_plt_eventtimes)
        class_test_plt_eventtimes_err.np_d_est_triggers(np_d_sig=self.np_test_plt_eventtimes,
                                                        i_direction=self.i_direction_test_plt_eventtimes_rising,
                                                        d_threshold=self.d_threshold_test_plt_eventtimes,
                                                        d_hysteresis=self.d_hysteresis_test_plt_eventtimes+20.0,
                                                        b_verbose=False)
        class_test_plt_eventtimes_err.str_plot_desc = 'plt_eventtimes test (single) on Error'
        class_test_plt_eventtimes_err.str_point_name_set(str_point_name='CY1', idx=0)
        with self.assertRaises(Exception):
            class_test_plt_eventtimes_err.plt_eventtimes()

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
