from ds1054z import DS1054Z
import unittest
import fcn_scope_ds1054z as scp


class TestScopeDS1054Z(unittest.TestCase):

    def setUp(self):
        # Define the initial values for the test
        self.timebase_scale_test = 1.0
        self.i_ns_test = 120
        self.lst_ch_active = [True, True, False, False]
        self.lst_ch_scale = [5.e-1, 2., 1., 1.]

        # This unittest requires a scope be connected
        self.scope = DS1054Z('192.168.1.206')

        # Assumes the signal on channel 1 has a fundamental frequency of 200-500
        # cycles per minutes (CPM).
        self.timebase_scale_test_slow = 1e-1

    def test_d_get_delta_time(self):
        d_t_del = scp.d_get_delta_time(timebase_scale=self.timebase_scale_test, i_ns=self.i_ns_test)
        # 12 divisions on the scope time so multiply by 12 to get total time
        # duration of sample
        d_check = (12. * self.timebase_scale_test) / float(self.i_ns_test)
        self.assertAlmostEqual(d_t_del, d_check, 12)

    def test_b_setup_scope(self):
        lst_as_left = scp.b_setup_scope(self.scope, lst_ch_scale=self.lst_ch_scale,
                                        lst_ch_active=self.lst_ch_active,
                                        timebase_scale=self.timebase_scale_test_slow,
                                        d_trigger_level=1e-01, b_single=False)
        self.assertGreater(lst_as_left[0], 0., 'Failed to retrieve scale from scope')
        self.assertAlmostEqual(lst_as_left[0], self.lst_ch_scale[0], 'Failed to set vertical scale')


if __name__ == '__main__':
    unittest.main()
