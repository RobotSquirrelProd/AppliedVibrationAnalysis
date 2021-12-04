import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import csv
import pandas as pd
import scipy.signal as sig
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d

import abc as abc


class ClSig(abc.ABC):
    """Abstract class to manage signals. Abstract base class"""

    @property
    @abc.abstractmethod
    def np_d_sig(self):
        """Numpy array containing the signal"""
        pass

    @property
    @abc.abstractmethod
    def b_complex(self):
        """Boolean, set to true to treat signal as complex"""
        pass

    @property
    @abc.abstractmethod
    def i_ns(self):
        """Number of samples in the scope data"""
        pass

    @property
    @abc.abstractmethod
    def ylim_tb(self):
        """Real-valued Timebase vertical limits"""
        pass

    @property
    @abc.abstractmethod
    def str_eu(self):
        """Engineering units for the signal"""
        pass

    @property
    @abc.abstractmethod
    def str_point_name(self):
        """Point name signal"""
        pass

    @abc.abstractmethod
    def set_ylim_tb(self, ylim_tb):
        pass


class ClSigReal(ClSig):
    """
    Class for storing, plotting, and manipulating real-valued signals

    ...

    Attributes
    ----------

    Methods
    ----------

    """

    def __init__(self, np_sig, d_fs, str_eu='volts', str_point_name='CH1'):
        """
        Parameters
        ----------
        np_sig : numpy array
            Vector with real-valued signal of interest
        d_fs : double
            Sampling frequency, hertz
        str_eu : string
            Engineering units. Defaults to 'volts'
        str_point_name : string
            Signal point name
        """
        # Parent class
        super(ClSigReal, self).__init__()

        # Signal meta data
        self.__b_complex = False
        self.np_d_sig = np_sig
        self.__d_fs = d_fs
        self.__b_is_stale_fs = True
        self.str_eu = str_eu
        self.str_point_name = str_point_name

        # Derived features for the signal
        self.__i_ns = self.__get_num_samples()
        self.__d_time = np.array([0, 1])
        self.__d_time_max = 0.0
        self.__d_time_min = 0.0

        # Timebase plot attributes. Some/many are derived from the signal
        # itself so they need to be in this object, even though other
        # object could be generating the plot.
        self.__d_time_plot = self.__d_time
        self.__ylim_tb = [0]
        self.set_ylim_tb(self.__ylim_tb)
        self.__i_x_divisions_tb = 12
        self.__i_y_divisions_tb = 9
        self.__str_eu_x = 'seconds'

        # Setup the s-g array and filtering parameters
        self.__np_d_sig_filt_sg = np_sig
        self.__i_win_len = 31
        self.__i_poly_order = 1
        self.__str_filt_sg_desc = 'No Savitsky-Golay filtering'
        self.__str_filt_sg_desc_short = 'No S-G Filter'
        self.__b_is_stale_filt_sg = True

        # Setup the butterworth FIR filtered signal vector and parameters
        self.__np_d_sig_filt_butter = np_sig
        self.__i_poles = 1
        self.__d_wn = 0.
        self.__str_filt_butter_desc = 'No Butterworth filtering'
        self.__str_filt_butter_desc_short = 'No Butter'
        self.__b_is_stale_filt_butter = True

        # Attributes related to the half-spectrum calculation
        self.__i_ns_rfft = 0

        # Attributes for the zero-crossing instantaneous frequency estimation
        self.__d_threshold = 0.
        self.__d_hysteresis = 0.1
        self.__i_direction = 0
        self.__np_d_eventtimes = np.zeros_like(np_sig)
        self.__b_is_stale_eventtimes = True

        # Attributes for the nX vector estimation and plotting
        self.__class_sig_comp = ClSigCompUneven([0 + 1j, 0 - 1j], 1.)
        self.__b_is_stale_nx = True

        # Final step: since this is instantiation, flag new signal in class
        self.__set_new_sig(True)

    @property
    def np_d_sig(self):
        """Numpy array containing the signal"""
        return self.__np_d_sig

    @np_d_sig.setter
    def np_d_sig(self, np_d_sig_in):
        """
        Update the signal vector. This update forces a recalculation of all derived parameters.

        Parameters
        ----------
        np_d_sig_in : numpy array
            Vector with the signal of interest. Must be real-valued.

        """
        # With a new signal, all the filtering will have to be done
        if np.iscomplexobj(np_d_sig_in):
            raise Exception("Must be a real-valued signal vector")

        # Store the vector into the object, reset filtering state, and update related features
        self.__np_d_sig = np_d_sig_in
        self.__set_new_sig(True)
        self.__i_ns = self.__get_num_samples()

    @property
    def d_fs(self):
        """Sampling frequency in hertz"""
        return self.__d_fs

    @d_fs.setter
    def d_fs(self, d_fs_in):
        """
        Update the sampling frequency. This will force a recalculation of filtered signal since
        normalized frequency is calculated from the sampling frequency.

        Parameters
        ----------
        d_fs_in : double
            New sampling frequency, hertz

        """
        self.__d_fs = d_fs_in
        self.__set_new_sig(True)

    def __set_new_sig(self, b_state):
        """
        Internal function that sets flags based on state of signal

        Parameters
        ----------
        b_state : boolean
            Set to True to re-calculate all functions dependant on either signal values or
            sampling frequency.

        """
        if b_state:
            self.__b_is_stale_fs = True
            self.__b_is_stale_filt_sg = True
            self.__b_is_stale_filt_butter = True
            self.__b_is_stale_eventtimes = True

    @property
    def str_eu(self):
        return self.__str_eu

    @str_eu.setter
    def str_eu(self, str_eu_in):
        self.__str_eu = str_eu_in

    @property
    def str_point_name(self):
        return self.__str_point_name

    @str_point_name.setter
    def str_point_name(self, str_point_name):
        self.__str_point_name = str_point_name

    @property
    def b_complex(self):
        return self.__b_complex

    # Calculate the number of samples in the signal
    def __get_num_samples(self):
        """Calculate number of samples in the signal"""
        return len(self.__np_d_sig)

    @property
    def i_ns(self):
        self.__i_ns = self.__get_num_samples()
        return self.__i_ns

    @property
    def ylim_tb(self):
        """Real-valued Timebase vertical limits
        Return
        ------

        list (double) :  plot y-limits

        """
        return self.__ylim_tb

    @ylim_tb.setter
    def ylim_tb(self, ylim_tb_in):
        # Update object attribute
        self.set_ylim_tb(ylim_tb_in)
        # This impacts other plot attributes, update those
        self.__get_d_time()

    def set_ylim_tb(self, ylim_tb):
        """
        Set the real-valued y limits

        Parameters
        ----------
        ylim_tb : list
            List of [min max] values for y-axis

        """
        # Only use limits if they are valid
        if len(ylim_tb) == 2:
            self.__ylim_tb = ylim_tb
        else:
            self.__ylim_tb = np.array(
                [np.min(self.__np_d_sig), np.max(self.__np_d_sig)])

    @property
    def i_y_divisions_tb(self):
        return self.__i_y_divisions_tb

    @i_y_divisions_tb.setter
    def i_y_divisions_tb(self, i_y_divisions_tb_in):
        self.__i_y_divisions_tb = i_y_divisions_tb_in

    @property
    def str_eu_x(self):
        return self.__str_eu_x

    @str_eu_x.setter
    def str_eu_x(self, str_eu_x_in):
        self.__str_eu_x = str_eu_x_in

    @property
    def d_time_plot(self):
        """

        This method returns the time series values scaled for plotting

        """
        if self.__b_is_stale_fs:
            self.__get_d_time()
        return self.__d_time_plot

    @property
    def xlim_tb(self):
        """
        This method returns the x-axis limits, accounting for scaling
        for good plot presentation

        """
        if self.__b_is_stale_fs:
            self.__get_d_time()
        return [self.__d_time_min, self.__d_time_max]

    @property
    def ylim_apht_mag(self):
        return self.__class_sig_comp.ylim_mag

    @ylim_apht_mag.setter
    def ylim_apht_mag(self, ylim_apht_mag):
        self.__class_sig_comp.ylim_mag = ylim_apht_mag

    @property
    def str_plot_apht_desc(self):
        return self.__class_sig_comp.str_plot_apht_desc

    @str_plot_apht_desc.setter
    def str_plot_apht_desc(self, str_plot_apht_desc_in):
        self.__class_sig_comp.str_plot_apht_desc = str_plot_apht_desc_in

    def __get_d_time(self):
        """
        Calculate signal features that depend on the sampling frequency including:
        - Time series (d_time)
        - Plotting features

        Returns
        -------

        numpy array, double : Updated time series

        """
        # Re-calculate the time series
        self.__d_time = np.linspace(0, (self.i_ns - 1), self.i_ns) * self.d_t_del()

        # These are plot attributes that need to be updated when the time time series
        # changes
        self.__d_time_max = 0.0
        self.__d_time_min = 0.0

        # Remove leading zeros if time scale is small
        self.__d_time_plot = self.__d_time
        self.__d_time_max = np.max(self.__d_time_plot)
        if self.__d_time_max < 1e-1:
            self.__d_time_plot = self.__d_time_plot * 1e3
            self.__str_eu_x = 'milliseconds'

        # Update signal maximum and minimum
        self.__d_time_max = np.max(self.__d_time_plot)
        self.__d_time_min = np.min(self.__d_time_plot)

        # With everything updated set the stale data flag to false
        self.__b_is_stale_fs = False

        return self.__d_time

    def d_t_del(self):
        """
        Delta time between each sample.
        """
        return 1.0 / self.d_fs

    @property
    def d_time(self):
        """Numpy array with time values, in seconds"""
        if self.__b_is_stale_fs:
            self.__d_time = self.__get_d_time()
        return self.__d_time

    @property
    def d_time_max(self):
        """Maximum value in the time series"""
        return self.__d_time_max

    @property
    def d_time_min(self):
        """Minimum value in the time series"""
        return self.__d_time_min

    @property
    def i_x_divisions_tb(self):
        return self.__i_x_divisions_tb

    @i_x_divisions_tb.setter
    def i_x_divisions_tb(self, i_x_divisions_tb_in):
        self.__i_x_divisions_tb = i_x_divisions_tb_in

    @property
    def str_filt_sg_desc(self):
        """Long Savitsky-Golay description"""
        return self.__str_filt_sg_desc

    @property
    def str_filt_sg_desc_short(self):
        """Short Savitsky-Golay description"""
        return self.__str_filt_sg_desc_short

    @property
    def np_d_sig_filt_sg(self):
        """ Return the signal, filtered with Savitsky-Golay"""

        # Does the filter need to be applied (signal updated) or can
        # we return the prior instance?
        if self.__b_is_stale_filt_sg:

            # If there are enough samples, filter
            if self.i_ns > self.__i_win_len:
                self.__np_d_sig_filt_sg = sig.savgol_filter(self.np_d_sig,
                                                            self.__i_win_len,
                                                            self.__i_poly_order)
                self.__str_filt_sg_desc = ('Savitsky-Golay | Window Length: ' +
                                           '%3.f' % self.__i_win_len +
                                           ' | Polynomial Order: ' +
                                           '%2.f' % self.__i_poly_order)
                self.__str_filt_sg_desc_short = 'SGolay'

            else:
                # Since we cannot perform the filtering, copy the original
                # signal into the vector and modify the descriptions
                self.__np_d_sig_filt_sg = self.np_d_sig
                self.__str_filt_sg_desc = 'No Savitsky-Golay filtering'
                self.__str_filt_sg_desc_short = 'No S-G Filter'

            # Flag that the filtering is done
            self.__b_is_stale_filt_sg = False

        return self.__np_d_sig_filt_sg

    @property
    def str_filt_butter_desc(self):
        """Long Butterworth FIR filter description"""
        return self.__str_filt_butter_desc

    @property
    def str_filt_butter_desc_short(self):
        """Short Butterworth FIR filter description"""
        return self.__str_filt_butter_desc_short

    @property
    def i_poles(self):
        return self.__i_poles

    @property
    def np_d_sig_filt_butter(self):
        """
        Return the signal, filtered with butterworth FIR filter

        """

        # Does the filter need to applied?
        if self.__b_is_stale_filt_sg:

            # This is a guess of the filter corner, useful for general vibration
            # analysis of physical displacements.
            # TO DO: This needs to be own method, should not be setting this here
            if self.d_fs < 300:
                self.__d_wn = self.d_fs / 8.
            else:
                self.__d_wn = 100.

            # Store the filter parameters in second-order sections to avoid
            # numerical errors
            sos = sig.butter(self.__i_poles, self.__d_wn, btype='low',
                             fs=self.d_fs, output='sos')

            # Perform the filtering
            self.__np_d_sig_filt_butter = sig.sosfilt(sos, self.np_d_sig)

            # Generate the plain text descriptions for the plots
            self.__str_filt_butter_desc = ('Butterworth | Poles: ' +
                                           '%2.f' % self.__i_poles +
                                           ' | Lowpass corner (Hz): ' +
                                           '%0.2f' % self.__d_wn)
            self.__str_filt_butter_desc_short = 'Butter'
            self.__b_is_stale_filt_butter = False

        # Return the filtered signal
        return self.__np_d_sig_filt_butter

    @property
    def i_ns_rfft(self):
        return self.__i_ns_rfft

    # Method for calculating the spectrum for a real signal
    def d_fft_real(self):
        """Calculate the half spectrum since this is a real-valued signal"""
        d_y = rfft(self.np_d_sig)
        self.__i_ns_rfft = len(d_y)

        # Scale the fft. I'm using the actual number
        # of points to scale.
        d_y = d_y / float(self.__i_ns_rfft)

        # Calculate the frequency scale
        d_ws = rfftfreq(self.i_ns, 1. / self.d_fs)

        # Return the values
        return [d_ws, d_y]

    # The value of this attribute can be read, but it should
    # not be set, outside of the estimate crossings methods, since
    # any change requires a re-calculation of the eventtimes
    @property
    def d_threshold(self):
        return self.__d_threshold

    # The value of this attribute can be read, but it should
    # not be set, outside of the estimate crossings methods, since
    # any change requires a re-calculation of the eventtimes
    @property
    def d_hysteresis(self):
        return self.__d_hysteresis

    @property
    def i_direction(self):
        return self.__i_direction

    # This is effectively set with the estimate crossings methods
    @property
    def np_d_eventtimes(self):
        self.np_d_est_triggers(np_sig_in=None, i_direction=None, d_threshold=None,
                               d_hysteresis=None, b_verbose=False)
        return self.__np_d_eventtimes

    # Interpolation of points for instantaneous frequency estimation
    def calc_interpolate_crossing(self, np_sig_in, idx, b_verbose=False):

        """
        This method estimates time of crossing using linear interpolation

        Parameters
        ----------
        idx  : integer
            This is the sample index immediately after the trigger changed to active. The
            function assumes the prior sample was taken before the trigger state changed.
        np_sig_in : numpy array
            Signal to be evaluated for crossings
        b_verbose : boolean
            Print the intermediate steps (default: False). Useful for stepping through the
            method to troubleshoot or understand it better.

        Returns
        -------
        double : estimated trigger activation time

        """

        # Interpolate to estimate the actual crossing from
        # the 2 nearest points
        xp = np.array([np_sig_in[idx], np_sig_in[idx + 1]])
        fp = np.array([self.d_time[idx], self.d_time[idx + 1]])
        f_interp = interp1d(xp, fp, assume_sorted=False)
        d_time_estimated = f_interp(self.d_threshold)

        # More intermediate results
        if b_verbose:
            print('xp: ' + np.array2string(xp) + ' | fp: ' +
                  np.array2string(fp) + ' | d_thresh: ' +
                  '%0.4f' % self.d_threshold + ' | eventtimes: ' +
                  '%0.4f' % d_time_estimated)

        # Return the estimated crossing time
        return d_time_estimated

    # Estimate triggers for speed
    def np_d_est_triggers(self, np_sig_in=None, i_direction=None, d_threshold=None,
                          d_hysteresis=None, b_verbose=False):
        """
        This method estimates speed by identifying trigger points in time,
        a given threshold and hysteresis. When the signal level crosses
        the threshold, the trigger holds off. The trigger holds off
        until the signal crosses the hysteresis level. Hysteresis is
        defined relative to the threshold voltage.

        The trigger times are a first-order approximation of the instantaneous
        frequency. Most commonly used to estimate the rotating speed from
        magnetic pick-ups or eddy-current probes.

        Parameters
        ----------
        np_sig_in : numpy array, None
            Signal to be evaluated for crossings. It can be any signal, but the class is designed
            for the input to be one of the signals already defined in the class so that an example
            looks like: np_sig_in=class_test_real.np_d_sig. This defaults to 'None' and assigns the
            class attribute 'np_sig' to 'np_sig_in'
        i_direction : integer, None
            0 to search for threshold on rising signal, 1 to search on a falling signal. Set to
            'None' to use prior value stored in the class
        d_threshold : double, None
            Threshold value (default: 0.0 volts for zero crossings)
        d_hysteresis : double, None
            Hysteresis value (default: 0.1 volts)
        b_verbose : boolean
            Print the intermediate steps (default: False). Useful for stepping through the
            method to troubleshoot or understand it better.

        Returns
        -------
        numpy array : list of trigger event times

        """

        # Parse the inputs, flagging stale data if any of these have been changed. Changes
        # in any of these attributes forces new eventtimes and nX calculations
        if np_sig_in is None:

            # Copy the class vector into this method
            np_sig_in = self.np_d_sig
        else:
            # User is possibly adding a new signal, force recalculation
            self.__b_is_stale_eventtimes = True
            self.__b_is_stale_nx = True

        if i_direction is not None:
            # User is possibly adding a new direction, force recalculation
            self.__i_direction = i_direction
            self.__b_is_stale_eventtimes = True
            self.__b_is_stale_nx = True
            if b_verbose:
                print('i_direction: ' + '%1.0f' % self.__i_direction)

        if d_threshold is not None:
            # User is possibly adding a new threshold, force recalculation
            self.__d_threshold = d_threshold
            self.__b_is_stale_eventtimes = True
            self.__b_is_stale_nx = True
            if b_verbose:
                print('d_threshold: ' + '%0.4f' % self.__d_threshold)

        if d_hysteresis is not None:
            # User is possibly adding a new hysteresis, force recalculation
            self.__d_hysteresis = d_hysteresis
            self.__b_is_stale_eventtimes = True
            self.__b_is_stale_nx = True
            if b_verbose:
                print('d_threshold: ' + '%0.4f' % self.__d_threshold)

        # Run the calculation if stale data is present
        if self.__b_is_stale_eventtimes:

            # Initialize trigger state to hold off: the trigger will be active
            # once the signal crosses the hysteresis
            b_trigger_hold = True

            # Initiate state machine: one state for rising signal,
            # 'up', (i_direction = 0) and another for falling signal,
            # 'down', (i_direction = 1)
            idx_event = 0
            self.__np_d_eventtimes = np.zeros_like(np_sig_in)
            if self.__i_direction == 0:

                # Define the absolute hysteretic value, rising
                d_hysteresis_abs = self.d_threshold - d_hysteresis
                if b_verbose:
                    print('d_hysteresis_abs: ' + '%0.4f' % d_hysteresis_abs)

                # Loop through the signal
                for idx, x in enumerate(np_sig_in[0:-1]):

                    # Intermediate results
                    if b_verbose:
                        print('idx: ' + '%2.f' % idx + ' | x: ' + '%0.5f' % x +
                              ' | s-g: ' + '%0.4f' % np_sig_in[idx])

                    # Only the sign matters so subtract this point from next to
                    # get sign of slope
                    d_slope = np_sig_in[idx + 1] - np_sig_in[idx]

                    # The trigger leaves 'hold-off' state if the slope is
                    # negative and we fall below the threshold
                    if x <= d_hysteresis_abs and d_slope < 0 and b_trigger_hold:
                        # Next time the signal rises above the threshold, trigger
                        # will be set to hold-off state
                        b_trigger_hold = False
                        if b_verbose:
                            print('Trigger hold off (false), rising')

                    # If we are on the rising portion of the signal and there is
                    # no hold off state on the trigger then trigger, and change
                    # state
                    if (x < self.d_threshold <= np_sig_in[idx + 1]) and d_slope > 0 and not b_trigger_hold:
                        # Change state to hold off
                        b_trigger_hold = True
                        if b_verbose:
                            print('Triggered, rising')

                        # Estimate time of crossing with interpolation
                        self.__np_d_eventtimes[idx_event] = self.calc_interpolate_crossing(np_sig_in, idx)

                        # Increment the eventtimes index
                        idx_event += 1

            else:

                # Define the absolute hysteretic value, falling
                d_hysteresis_abs = self.d_threshold + self.d_hysteresis

                # Loop through the signal
                for idx, x in enumerate(np_sig_in[0:-1]):

                    # Intermediate results
                    if b_verbose:
                        print('idx: ' + '%2.f' % idx + ' | x: ' + '%0.5f' % x +
                              ' | s-g: ' + '%0.4f' % np_sig_in[idx])

                    # Only the sign matters so subtract this point from next to
                    # get sign of slope
                    d_slope = np_sig_in[idx + 1] - np_sig_in[idx]

                    # The trigger leaves 'hold-off' state if the slope is
                    # positive and we rise above the threshold
                    if x >= d_hysteresis_abs and d_slope > 0 and b_trigger_hold:
                        # Next time the signal rises above the threshold, trigger
                        # will be set to hold-off state
                        b_trigger_hold = False
                        if b_verbose:
                            print('Trigger hold off (false), falling')

                    # If we are on the falling portion of the signal and
                    # there is no hold off state on the trigger then trigger
                    # and change state
                    if (x > self.d_threshold >= np_sig_in[idx + 1]) and d_slope < 0 and not b_trigger_hold:
                        # Change state to hold off
                        b_trigger_hold = True
                        if b_verbose:
                            print('Triggered, falling')

                        # Estimate time of crossing with interpolation
                        self.__np_d_eventtimes[idx_event] = self.calc_interpolate_crossing(np_sig_in, idx)

                        # Increment the eventtimes index
                        idx_event += 1

            # Remove zero-valued element
            self.__np_d_eventtimes = np.delete(
                self.__np_d_eventtimes, np.where(self.__np_d_eventtimes == 0))

            # Freshly updated eventtimes
            self.__b_is_stale_eventtimes = False

            # Since the eventtimes were calculated the nX vectors have to marked
            # as stale
            self.__b_is_stale_nx = True

        # Return the list of eventtimes.
        return self.__np_d_eventtimes

    # Estimate nX vectors, given trigger events and a signal
    def calc_nx(self, np_sig_in=None, np_d_eventtimes=None, b_verbose=False):
        """
        This method estimates the 1X vectors, given trigger event times. The
        phase reported in this estimation is intended to be used for balancing
        so phase lag is positive (spectral phase lag is negative). Since this is 
        implemented in the real signal class, the method assumes the signal in
        is also real. 

        Parameters
        ----------
        np_sig_in : numpy array
            Signal to be evaluated for crossings. Should reference a signal already loaded
            into the object (i.e. np_sig_in = {ClSigReal}.np_d_sig). Setting 'np_sig_in' to None
            forces the function to use .np sig.
        np_d_eventtimes : numpy array
            Vector of trigger event times. Setting to None forces use of eventtimes defined
            in the class
        b_verbose : boolean
            Print the intermediate steps (default: False). Useful for stepping through the
            method to troubleshoot or understand it better.

        Returns
        -------
        numpy array : complex signal vector with the nX vectors

        """

        # Parse the inputs, flagging stale data if any of these have been changed. Changes
        # in any of these attributes forces new eventtimes and nX calculations
        if np_sig_in is None:

            # Copy the class vector into this method
            np_sig_in = self.np_d_sig

        else:
            # User is possibly adding a new signal, force recalculation
            self.__b_is_stale_nx = True

        if np_d_eventtimes is not None:
            # User is possibly adding a new set of eventtimes, force recalculation
            self.__np_d_eventtimes = np_d_eventtimes
            self.__b_is_stale_nx = True

        # Does this calculation need to be refreshed?
        if self.__b_is_stale_nx:

            # Begin by identifying the closest index to the eventtimes
            idx_events = np.round(self.__np_d_eventtimes * self.d_fs, decimals=0)
            d_nx = np.zeros_like(self.__np_d_eventtimes, dtype=complex)
            for idx, idx_active in enumerate(idx_events[0:-1]):

                # Define starting and ending index
                idx_start = int(idx_active)
                idx_end = int(idx_events[idx + 1]) - 1

                # Calculate the single-sided FFT, multiplying the result by -1 change
                # from spectral phase to balance phase.
                d_np_y = rfft(np_sig_in[idx_start:idx_end]) * (-1.0 + 0.0j)
                i_ns_rfft = len(d_np_y)

                # Scale the fft using the actual number
                # of points to scale.
                d_np_y = d_np_y / float(i_ns_rfft)

                # Grab the first element since it is the best estimate
                # of the sinusoid with the same frequency as the
                # spacing of eventtimes
                d_nx[idx] = d_np_y[1]

                # Print summary
                if b_verbose:
                    print('idx_start: ' + '%5.0f' % idx_start + ' | idx_end: ' +
                          '%5.0f' % idx_end + ' | nX mag: ' + '%2.6f' % abs(d_nx[idx]) +
                          ' | %2.6f' % np.rad2deg(np.angle(d_nx[idx])) + ' deg.')

            # Pad the end
            d_nx[-1] = d_nx[-2]

            # Update calculation status
            self.__b_is_stale_nx = False

            # Update the complex class
            self.__class_sig_comp = ClSigCompUneven(d_nx, self.__np_d_eventtimes)

        # Return the values
        return self.__class_sig_comp.np_d_sig

    # Call the method to plot the apht plot
    def plt_apht(self, str_plot_apht_desc_in=None):
        return self.__class_sig_comp.plt_apht(str_plot_apht_desc=str_plot_apht_desc_in)

    # Call polar plotting method.
    def plt_polar(self, str_plot_polar_desc=None):
        """Plot out amplitude in phase in polar format

        Parameters
        ----------
        str_plot_polar_desc : string
            Additional title text for the plot. If 'None' then method uses class attribute.

        Return values:
        handle to the plot
        """

        return self.__class_sig_comp.plt_polar(str_plot_polar_desc)


class ClSigComp(ClSig):
    """Class for storing, plotting, and manipulating complex-valued
       signals"""

    def __init__(self, np_sig, d_fs, str_eu='volts', str_point_name='CH1'):

        """
        Parameters
        ----------
        np_sig : numpy array, double
            Vector with real-valued signal of interest
        d_fs : double
            Sampling frequency, hertz
        str_eu : string
            Engineering units. Defaults to 'volts'
        str_point_name : string
            Signal point name
        """

        super(ClSigComp, self).__init__()
        self.__b_complex = True
        self.__np_d_sig = np_sig

        # Signal metadata
        self.__str_eu = str_eu
        self.__d_fs = d_fs
        self.__i_ns = self.__get_num_samples()
        self.str_point_name = str_point_name

        # Plot attributes
        self.__ylim_tb = [0]
        self.set_ylim_tb(self.__ylim_tb)

    @property
    def np_d_sig(self):
        """Numpy array containing the signal"""
        self.__i_ns = self.__get_num_samples()
        return self.__np_d_sig

    @property
    def d_fs(self):
        """Sampling frequency in hertz"""
        return self.__d_fs

    @property
    def b_complex(self):
        return self.__b_complex

    # Calculate the number of samples in the signal
    def __get_num_samples(self):
        """Calculate number of samples in the signal"""
        return len(self.__np_d_sig)

    @property
    def str_eu(self):
        return self.__str_eu

    @str_eu.setter
    def str_eu(self, str_eu_in):
        self.__str_eu = str_eu_in

    @property
    def str_point_name(self):
        return self.__str_point_name

    @str_point_name.setter
    def str_point_name(self, str_point_name):
        self.__str_point_name = str_point_name

    @property
    def i_ns(self):
        self.__i_ns = self.__get_num_samples()
        return self.__i_ns

    @property
    def ylim_tb(self):
        """Real-valued Timebase vertical limits"""
        return self.__ylim_tb

    @ylim_tb.setter
    def ylim_tb(self, ylim_tb):
        """Vertical limits for timebase (tb) plots"""
        self.set_ylim_tb(ylim_tb)

    def set_ylim_tb(self, ylim_tb):
        """Setter for the real-valued y limits"""
        # Only use limits if they are valid
        if len(ylim_tb) == 2:
            self.__ylim_tb = ylim_tb
        else:
            self.__ylim_tb = np.array(
                [np.max(self.__np_d_sig), np.min(self.__np_d_sig)])


class ClSigCompUneven(ClSig):
    """

    Class for storing, plotting, and manipulating complex-valued signals sampled
    at uneven time intervals. Common source of data is nX vectors derived from
    a machine with transient speed (i.e. start-up or shutdown).

    ...

    Attributes
    ----------

    Methods
    ----------

    """

    def __init__(self, np_sig, np_d_time, str_eu='volts', str_point_name='CH1'):

        """
        Parameters
        ----------
        np_sig : numpy array, double
            Vector with real-valued signal of interest
        np_d_time : numpy array, double
            Time stamp for each sample, assumed to be seconds
        str_eu : string
            Engineering units. Defaults to 'volts'
        str_point_name : string
            Signal point name
        """

        super(ClSigCompUneven, self).__init__()

        # Bring in the signal and timestamps
        self.__np_d_sig = np_sig
        self.__np_d_time = np_d_time

        # Signal metadata
        self.__b_complex = True
        self.__i_ns = self.__get_num_samples()
        self.__str_eu = str_eu
        self.str_point_name = str_point_name

        # Plotting attributes
        self.__ylim_mag = [0]
        self.set_ylim_mag(self.__ylim_mag)
        self.__ylim_tb = [0]
        self.set_ylim_tb(self.__ylim_tb)
        self.__str_plot_apht_desc = '-'
        self.__str_plot_polar_desc = '-'

    @property
    def np_d_sig(self):
        """Numpy array containing the signal"""
        self.__i_ns = self.__get_num_samples()
        return self.__np_d_sig

    @property
    def np_d_time(self):
        """Sample timestamps"""
        return self.__np_d_time

    @property
    def b_complex(self):
        return self.__b_complex

    # Calculate the number of samples in the signal
    def __get_num_samples(self):
        """Calculate number of samples in the signal"""
        return len(self.__np_d_sig)

    @property
    def i_ns(self):
        self.__i_ns = self.__get_num_samples()
        return self.__i_ns

    @property
    def ylim_mag(self):
        """apht magnitude vertical limits"""
        return self.__ylim_mag

    @ylim_mag.setter
    def ylim_mag(self, ylim_mag):
        """Vertical limits for apht magnitude (tb) plots"""
        self.set_ylim_mag(ylim_mag)

    @property
    def ylim_tb(self):
        """Magnitude timebase vertical limits"""
        return self.__ylim_tb

    @ylim_tb.setter
    def ylim_tb(self, ylim_tb):
        """Magnitude timebase vertical limits"""
        self.set_ylim_tb(ylim_tb)

    def set_ylim_mag(self, ylim_mag):
        """Setter for the magnitude y limits"""
        # Only use limits if they are valid
        if len(ylim_mag) == 2:
            self.__ylim_mag = ylim_mag
        else:
            self.__ylim_mag = np.array(
                [1.05 * np.max(np.abs(self.__np_d_sig)), 0.95 * np.min(np.abs(self.__np_d_sig))])

    def set_ylim_tb(self, ylim_tb):
        """Y limits for magnitude timebase plot"""
        # Only use limits if they are valid
        if len(ylim_tb) == 2:
            self.__ylim_tb = ylim_tb
        else:
            self.__ylim_tb = np.array(
                [np.max(np.abs(self.__np_d_sig)), np.abs(np.min(self.__np_d_sig))])

    @property
    def str_eu(self):
        return self.__str_eu

    @str_eu.setter
    def str_eu(self, str_eu_in):
        self.__str_eu = str_eu_in

    @property
    def str_point_name(self):
        return self.__str_point_name

    @str_point_name.setter
    def str_point_name(self, str_point_name):
        self.__str_point_name = str_point_name

    @property
    def str_plot_apht_desc(self):
        return self.__str_plot_apht_desc

    @str_plot_apht_desc.setter
    def str_plot_apht_desc(self, str_plot_apht_desc_in):
        self.__str_plot_apht_desc = str_plot_apht_desc_in

    # Plotting method, apht plots.
    def plt_apht(self, str_plot_apht_desc=None):
        """Plot out amplitude in phase in apht format

        Return values:
        handle to the plot
        """

        # Parse inputs
        if str_plot_apht_desc is not None:
            # Update class attribute
            self.__str_plot_apht_desc = str_plot_apht_desc

        # Figure with subplots
        fig, axs = plt.subplots(2)
        fig.suptitle('apht plot')

        # Plot the phase
        axs[0].plot(self.__np_d_time, np.rad2deg(np.angle(self.__np_d_sig)))
        axs[0].grid()
        axs[0].set_xlabel("Time, seconds")
        axs[0].set_ylabel("Phase, degrees")
        axs[0].set_ylim([-360.0, 360.0])
        axs[0].set_title(self.__str_plot_apht_desc + " phase")

        # Plot the magnitude
        axs[1].plot(self.__np_d_time, np.abs(self.__np_d_sig))
        axs[1].grid()
        axs[1].set_xlabel("Time, seconds")
        axs[1].set_ylabel("Magnitude, " + self.str_eu)
        axs[1].set_ylim(self.ylim_mag)
        axs[1].set_title(self.__str_plot_apht_desc + " magnitude")

        # Set the layout
        plt.tight_layout()

        # Save off the handle to the plot
        plot_handle = plt.gcf()

        # Show the plot, creating a new figure. This command resets the graphics context
        # so the plot handle has to be saved first.
        plt.show()

        return plot_handle

    @property
    def str_plot_polar_desc(self):
        return self.__str_plot_polar_desc

    @str_plot_polar_desc.setter
    def str_plot_polar_desc(self, str_plot_polar_desc_in):
        self.__str_plot_polar_desc = str_plot_polar_desc_in

    # Plotting method, polar plots.
    def plt_polar(self, str_plot_polar_desc=None):
        """Plot out amplitude in phase in polar format

        Parameters
        ----------
        str_plot_polar_desc : string
            Additional title text for the plot. If 'None' then method uses class attribute.

        Return values:
        handle to the plot
        """

        # Parse inputs
        if str_plot_polar_desc is not None:
            # Update class attribute
            self.__str_plot_polar_desc = str_plot_polar_desc

        # Figure with subplots
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # Polar plot
        ax.plot(np.angle(self.__np_d_sig), np.abs(self.__np_d_sig))
        ax.set_rmax(np.max(self.ylim_mag))
        d_tick_radial = np.round(np.max(self.ylim_mag) / 4.0, decimals=1)
        ax.set_rticks([d_tick_radial, d_tick_radial * 2.0, d_tick_radial * 3.0, d_tick_radial * 4.0])
        ax.set_rlabel_position(-22.5)
        ax.grid(True)
        ax.set_title(self.__str_plot_polar_desc + " polar plot")

        # Set the layout
        plt.tight_layout()

        # Save off the handle to the plot
        plot_handle = plt.gcf()

        # Show the plot, creating a new figure. This command resets the graphics context
        # so the plot handle has to be saved first.
        plt.show()

        return plot_handle


class ClSigFeatures:
    """Class to manage signal features on scope data and other signals

    Example usage:
        cl_test = ClSigFeatures(np.array([1.,2., 3.]),1.1)

    Should produce:

        print('np_d_sig: '+ np.array2string(cl_test.np_d_sig))
        print('timebase_scale: ' + '%0.3f' % cl_test.timebase_scale)
        print('i_ns: ' + '%3.f' % cl_test.i_ns)
        print('d_t_del: ' + '%0.3f' % cl_test.d_t_del)
        print('d_time' + np.array2string(cl_test.d_time))

        np_d_sig: [1. 2. 3.]
        i_ns:   3

        Attributes
        ----------

        Methods
        -------
    """

    def __init__(self, np_d_sig, d_fs, str_point_name='CH1'):
        """
        Parameters
        ----------
        np_d_sig : numpy array
            Vector with the signal of interest. Can be real- or complex-valued.
        d_fs : double
            Describes the sampling frequency in samples/second (hertz).
        str_point_name : string
            Signal point name
        """
        # Instantiation of class so begin list and add first signal
        self.__lst_cl_sgs = []
        self.__lst_b_active = []

        # Instantiate and save common signal features to this class
        self.idx_add_sig(np_d_sig, d_fs=d_fs, str_point_name=str_point_name)
        self.__np_d_rpm = np.zeros_like(self.__lst_cl_sgs[0].np_d_sig)

        # Attributes related to file save/read behavior
        self.__str_file = ''
        self.__i_header_rows = 0

        self.__d_thresh = np.NaN
        self.__d_events_per_rev = np.NaN
        self.str_plot_desc = 'Test Data'
        self.b_spec_peak = False

    @property
    def b_complex(self):
        return self.__lst_cl_sgs[0].b_complex

    @property
    def b_spec_peak(self):
        """Boolean set to true to label peak in spectrum"""
        return self.__b_spec_peak

    @property
    def np_d_sig(self):
        """Numpy array containing the first signal"""
        return self.__lst_cl_sgs[0].np_d_sig

    def get_np_d_sig(self, idx=0):
        """Numpy array containing arbitrary signal"""
        return self.__lst_cl_sgs[idx].np_d_sig

    @np_d_sig.setter
    def np_d_sig(self, lst_in):
        np_sig_in = lst_in[0]
        idx = lst_in[1]
        self.__lst_cl_sgs[idx].np_d_sig = np_sig_in
        self.__lst_b_active[idx] = True

    def idx_add_sig(self, np_d_sig, d_fs, str_point_name):
        """Add another signal to this object.
        returns index to the newly added signal.

        Parameters
        ----------
        np_d_sig : numpy array, double
            Signal to be added

        d_fs : double
            Sampling frequency, hertz

        str_point_name : string
            Signal point name
        """

        # TO DO: try/catch might be a better option here
        # Does the incoming signal have the same number of
        # samples as the ones already present?
        if len(self.__lst_cl_sgs) > 0:

            if len(np_d_sig) != self.i_ns:
                raise Exception('Cannot add signal with different number of samples')

        # cast to numpy array
        np_d_sig = np.array(np_d_sig)

        # Add the signals, looking for complex and real
        if np.iscomplexobj(np_d_sig):
            self.__lst_cl_sgs.append(ClSigComp(np_d_sig, d_fs))
        else:
            self.__lst_cl_sgs.append(ClSigReal(np_d_sig, d_fs))

        # signal index number
        idx_class = len(self.__lst_cl_sgs) - 1

        # add signal meta data
        self.__lst_cl_sgs[idx_class].str_point_name = str_point_name

        # Mark this one as active
        self.__lst_b_active.append(True)

        # Success, return index to new signal
        return idx_class

    @property
    def i_ns(self):
        """Number of samples"""
        # assumed to the be same for all signals, just return the first index
        return self.__lst_cl_sgs[0].i_ns

    def d_t_del(self, idx=0):
        """
        Delta time between each sample. This is allowed to vary across the signals

        Return
        ------
        double : Length of time between each sample

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        return 1.0 / self.__lst_cl_sgs[idx].d_del_t

    def d_fs(self, idx=0):
        """
        Return the Sampling frequency in hertz

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        return self.__lst_cl_sgs[idx].d_fs

    def d_fs_update(self, d_fs_in, idx=0):
        """
        Set the sampling frequency in hertz

        Parameters
        ---------
        d_fs_in : double
            Describes the sampling frequency in samples/second (hertz).

        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        self.__lst_cl_sgs[idx].d_fs = d_fs_in

    def str_filt_sg_desc(self, idx=0):
        """Complete Filt description of the Savitsky-Golay filter design"""
        return self.__lst_cl_sgs[idx].str_filt_sg_desc

    def str_filt_sg_desc_short(self, idx=0):
        """Short Filt description, useful for plot legend labels"""
        return self.__lst_cl_sgs[idx].str_filt_sg_desc_short

    def str_filt_butter_desc(self, idx=0):
        """
        Complete description of the Butterworth filter design

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        return self.__lst_cl_sgs[idx].str_filt_butter_desc

    def str_filt_butter_desc_short(self, idx=0):
        """
        Abbreviated description of the Butterworth filter design, useful for legend labels

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        return self.__lst_cl_sgs[idx].str_filt_butter_desc_short

    def np_d_eventtimes(self, idx=0):
        """
        Numpy array of trigger event times

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        return self.__lst_cl_sgs[idx].np_d_eventtimes

    # Estimate triggers for speed
    def np_d_est_triggers(self, np_d_sig, i_direction=0, d_threshold=0,
                          d_hysteresis=0.1, b_verbose=False, idx=0):
        """
        This method estimates speed by identifying trigger points in time,
        a given threshold and hysteresis. When the signal level crosses
        the threshold, the trigger holds off. The trigger holds off
        until the signal crosses the hysteresis level. Hysteresis is
        defined relative to the threshold voltage.

        The trigger times are a first-order approximation of the instantaneous
        frequency. Most commonly used to estimate the rotating speed from
        magnetic pick-ups or eddy-current probes.

        Parameters
        ----------
        np_d_sig : numpy array
            Signal to be evaluated for crossings
        i_direction : integer
            0 to search for threshold on rising signal, 1 to search on a falling signal.
        d_threshold : double
            Threshold value (default: 0.0 volts for zero crossings)
        d_hysteresis : double
            Hysteresis value (default: 0.1 volts)
        b_verbose : boolean
            Print the intermediate steps (default: False). Useful for stepping through the
            method to troubleshoot or understand it better.
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        Returns
        -------
        numpy array : list of trigger event times

        """
        return self.__lst_cl_sgs[idx].np_d_est_triggers(np_d_sig, i_direction, d_threshold,
                                                        d_hysteresis, b_verbose)

    # Estimate the filtered nX response
    def calc_nx(self, np_d_sig, np_d_eventtimes, b_verbose=True, idx=0):
        """
        This method calls the estimation method for each signal

        Parameters
        ----------
        np_d_sig : numpy array
            Signal to be evaluated for crossings
        np_d_eventtimes : numpy array
            Vector of trigger event times
        b_verbose : boolean
            Print the intermediate steps (default: False). Useful for stepping through the
            method to troubleshoot or understand it better.
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        Returns
        -------
        numpy array : list of trigger event times

        """
        return self.__lst_cl_sgs[idx].calc_nx(np_d_sig,
                                              np_d_eventtimes=np_d_eventtimes, b_verbose=b_verbose)

    @property
    def np_d_rpm(self):
        """Estimated RPM values"""
        return self.__np_d_rpm

    def str_eu(self, idx=0):
        """
        Engineering unit descriptor

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        return self.__lst_cl_sgs[idx].str_eu

    def str_point_name(self, idx=0):
        """
        Signal point name

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        return self.__lst_cl_sgs[idx].str_eu

    @property
    def str_plot_desc(self):
        """Plot description"""
        return self.__str_plot_desc

    @property
    def str_file(self):
        """Output (.csv) file name"""
        return self.__str_file

    @b_spec_peak.setter
    def b_spec_peak(self, b_spec_peak):
        self.__b_spec_peak = b_spec_peak

    @str_plot_desc.setter
    def str_plot_desc(self, str_plot_desc):
        self.__str_plot_desc = str_plot_desc

    # Interface for the vertical plotting limits
    def ylim_tb(self, ylim_tb_in=None, idx=0):
        """
        Parameter
        ---------
        ylim_tb_in : list of doubles, None
            vertical plot limits. If set to None, returns the limits without change
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        Returns
        -------
        list of doubles : ylim_tb applied to signal with index=idx

        """

        # Is there anything to update?
        if ylim_tb_in is not None:
            self.__lst_cl_sgs[idx].ylim_tb = ylim_tb_in

        return self.__lst_cl_sgs[idx].ylim_tb

    # Plotting method, time domain signals.
    def plt_sigs(self):
        """Plot out the data in this signal feature class in the time domain

        Return values:
        handle to the plot
        """

        # How many plots, assuming 1 is given?
        i_plots = 0
        for b_obj in self.__lst_b_active:
            if b_obj:
                i_plots += 1

        # Figure with subplots
        fig, axs = plt.subplots(i_plots)
        fig.suptitle('Oscilloscope data')

        # Initialize active channel value
        i_ch = 0

        # Branching because a single axis is not scriptable
        if i_plots > 1:
            ax1 = axs[i_ch]
        else:
            ax1 = axs

        # Channel 1
        ax1.plot(self.__lst_cl_sgs[0].d_time_plot, self.get_np_d_sig(idx=0))
        ax1.plot(self.__lst_cl_sgs[0].d_time_plot, self.__lst_cl_sgs[0].np_d_sig_filt_sg)
        ax1.plot(self.__lst_cl_sgs[0].d_time_plot, self.__lst_cl_sgs[0].np_d_sig_filt_butter)
        ax1.grid()
        ax1.set_xlabel("Time, " + self.__lst_cl_sgs[0].str_eu_x)
        ax1.set_xlim(self.__lst_cl_sgs[0].xlim_tb)
        ax1.set_xticks(np.linspace(self.__lst_cl_sgs[0].xlim_tb[0],
                                   self.__lst_cl_sgs[0].xlim_tb[1],
                                   self.__lst_cl_sgs[0].i_x_divisions_tb))
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.set_ylabel("Channel output, " + self.__lst_cl_sgs[0].str_eu)
        ax1.set_ylim(self.__lst_cl_sgs[0].ylim_tb)
        ax1.set_yticks(np.linspace(self.__lst_cl_sgs[0].ylim_tb[0],
                                   self.__lst_cl_sgs[0].ylim_tb[1],
                                   self.__lst_cl_sgs[0].i_y_divisions_tb))
        # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax1.set_title(self.__str_plot_desc + " Timebase")
        ax1.legend(['as-acquired', self.str_filt_sg_desc_short(),
                    self.str_filt_butter_desc_short()])

        # Debug - remove
        print('ylim_tb[0] : ' + '%0.6f' % self.__lst_cl_sgs[0].ylim_tb[0] +
              ' | ylim_tb[1] : ' + '%0.6f' % self.__lst_cl_sgs[0].ylim_tb[1] +
              ' | i_y_divisions_tb : ' + '%0.6f' % self.__lst_cl_sgs[0].i_y_divisions_tb)

        # Channel 2
        if len(self.__lst_b_active) > 1:
            if self.__lst_b_active[1]:
                i_ch = 1

                axs[i_ch].plot(self.__lst_cl_sgs[i_ch].d_time_plot, self.get_np_d_sig(idx=i_ch))
                axs[i_ch].plot(self.__lst_cl_sgs[i_ch].d_time_plot, self.__lst_cl_sgs[i_ch].np_d_sig_filt_sg)
                axs[i_ch].plot(self.__lst_cl_sgs[i_ch].d_time_plot, self.__lst_cl_sgs[i_ch].np_d_sig_filt_butter)
                axs[i_ch].grid()
                axs[i_ch].set_xlabel("Time, " + self.__lst_cl_sgs[i_ch].str_eu_x)
                axs[i_ch].set_xlim(self.__lst_cl_sgs[i_ch].xlim_tb)
                axs[i_ch].set_xticks(np.linspace(self.__lst_cl_sgs[i_ch].xlim_tb[0],
                                                 self.__lst_cl_sgs[i_ch].xlim_tb[1],
                                                 self.__lst_cl_sgs[i_ch].i_x_divisions_tb))
                axs[i_ch].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                axs[i_ch].set_ylabel("Channel output, " + self.__lst_cl_sgs[i_ch].str_eu)
                axs[i_ch].set_ylim(self.__lst_cl_sgs[i_ch].ylim_tb)
                axs[i_ch].set_yticks(np.linspace(self.__lst_cl_sgs[i_ch].ylim_tb[0],
                                                 self.__lst_cl_sgs[i_ch].ylim_tb[1],
                                                 self.__lst_cl_sgs[i_ch].i_y_divisions_tb))
                axs[i_ch].set_title(self.__str_plot_desc + " Timebase")
                axs[i_ch].legend(['as-acquired', self.str_filt_sg_desc_short(idx=i_ch),
                                  self.str_filt_butter_desc_short(idx=i_ch)])

        # Set the layout
        plt.tight_layout()

        # Save off the handle to the plot
        plot_handle = plt.gcf()

        # Show the plot, creating a new figure. This command resets the graphics context
        # so the plot handle has to be saved first.
        plt.show()

        return plot_handle

    # Plotting method for single-sided (real signal) spectrum
    def plt_spec(self):
        """Plot data in frequency domain. This method assumes a real signal

        Return values:
        list : [handle to the plot, frequency labels, complex-spectral values]

        """
        spec = self.__lst_cl_sgs[0].d_fft_real()
        d_mag = np.abs(spec[1])
        plt.figure()
        plt.plot(spec[0], d_mag)
        plt.grid()
        plt.xlabel("Frequency, hertz")
        plt.ylabel("Channel amplitude, " + self.__lst_cl_sgs[0].str_eu)
        plt.title(self.__str_plot_desc + " Spectrum")

        # Annotate the peak
        if self.__b_spec_peak:
            idx_max = np.argmax(d_mag)
            d_ws_peak = spec[0][idx_max]
            d_ws_span = (spec[0][-1] - spec[0][0])
            d_mag_peak = d_mag[idx_max]
            plt.plot(d_ws_peak, d_mag_peak, 'ok')
            str_label = ('%0.3f' % d_mag_peak + ' ' +
                         self.__lst_cl_sgs[0].str_eu + ' @ ' + '%0.2f' % d_ws_peak + ' Hz')
            plt.annotate(str_label, [
                d_ws_peak + (0.02 * d_ws_span), d_mag_peak * 0.95])

        # Save off the handle to the plot
        plot_handle = plt.gcf()

        # Show the plot, creating a new figure.
        plt.show()

        return [plot_handle, spec[0], spec[1]]

    # Plotting method for the eventtimes
    def plt_eventtimes(self, idx_eventtimes=0, idx=0):
        """
        Plot a signal and overlay event data in timebase format.
        Parameter
        ---------
        idx_eventtimes : integer
            Index of signal eventtimes. Defaults to 0 (first signal)
        idx : integer
            Index of signal to be plotted. Defaults to 0 (first signal)

        Returns
        -------
        list: [handle to the plot, np array of eventtimes]
        """

        # The eventtimes all should have threshold value for voltage
        np_d_event_value = np.ones_like(
            self.__lst_cl_sgs[idx_eventtimes].np_d_eventtimes) * self.__lst_cl_sgs[idx_eventtimes].d_threshold

        # Put up the the plot time
        plt.figure()
        plt.plot(self.__lst_cl_sgs[idx].d_time, self.__lst_cl_sgs[idx].np_d_sig)
        plt.plot(self.np_d_eventtimes(idx=idx_eventtimes), np_d_event_value, "ok")
        plt.grid(True)

        plt.xlabel("Time, " + self.__lst_cl_sgs[idx].str_eu_x)
        plt.xlim(self.__lst_cl_sgs[idx].xlim_tb)
        plt.xticks(np.linspace(self.__lst_cl_sgs[idx].xlim_tb[0],
                               self.__lst_cl_sgs[idx].xlim_tb[1],
                               self.__lst_cl_sgs[idx].i_x_divisions_tb))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.ylabel("Amplitude, " + self.__lst_cl_sgs[idx].str_eu)
        plt.ylim(self.__lst_cl_sgs[idx].ylim_tb)
        plt.yticks(np.linspace(self.__lst_cl_sgs[idx].ylim_tb[0],
                               self.__lst_cl_sgs[idx].ylim_tb[1],
                               self.__lst_cl_sgs[idx].i_y_divisions_tb))

        plt.legend(['as-acquired', 'eventtimes'])
        plt.title(self.__str_plot_desc + ' Amplitude and eventtimes vs. time')

        # Save the handle prior to showing
        plot_handle = plt.gcf()

        # Show the plot
        plt.show()

        return [plot_handle, self.np_d_eventtimes(idx=idx_eventtimes)]

    # Plotting method for the eventtimes, interpreted as RPM
    def plt_rpm(self):
        """Plot rpm data in time.

        Return values:
        list: [handle to the plot, np array of RPM values]
        """

        # Put up the the plot time
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()

        ax1.plot(self.__lst_cl_sgs[0].d_time, self.np_d_sig)
        ax2.plot(self.np_d_eventtimes, self.__np_d_rpm, "ok")
        ax1.set_xlabel('Time, seconds')
        ax1.set_ylabel('Amplitude, ' + self.__lst_cl_sgs[0].str_eu)
        ax2.set_ylabel('Event speed, RPM')
        plt.legend(['as-acquired', 'RPM'])
        plt.title('Amplitude and eventtimes vs. time')
        plt.show()

        plot_handle = plt.gcf()
        return [plot_handle, self.__np_d_rpm]

    # Plotting method, time domain signals.
    def plt_apht(self, str_plot_apht_desc=None, idx=0):
        """Plot out amplitude and phase versus time ("apht") format

        Parameters
        ----------
        str_plot_apht_desc : string
            Description of data to be included in apht plot title
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        Return values:
        handle to the plot
        """

        # Parse inputs
        if str_plot_apht_desc is not None:
            # Update class attribute
            self.__lst_cl_sgs[idx].str_plot_apht_desc = str_plot_apht_desc

        return self.__lst_cl_sgs[idx].plt_apht(str_plot_apht_desc)

    # Call polar plotting method.
    def plt_polar(self, str_plot_polar_desc=None, idx=0):
        """Plot out amplitude in phase in polar format

        Parameters
        ----------
        str_plot_polar_desc : string
            Additional title text for the plot. If 'None' then method uses class attribute.
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        Return values:
        handle to the plot
        """

        return self.__lst_cl_sgs[idx].plt_polar(str_plot_polar_desc)

    # Method to estimate the RPM values
    def d_est_rpm(self, d_events_per_rev=1):
        """
        Estimate the RPM from the signal using eventtimes which must have
        calculated from a previous call to the method np_d_est_triggers.
        """

        # Store the new value in the object
        self.__d_events_per_rev = d_events_per_rev

        # Calculate the RPM using the difference in event times
        self.__np_d_rpm = 60. / (np.diff(self.__lst_cl_sgs[0].np_d_eventtimes) * float(d_events_per_rev))

        # To keep the lengths the same, append the last sample
        self.__np_d_rpm = np.append(
            self.__np_d_rpm, self.__np_d_rpm[len(self.__np_d_rpm) - 1])

        return self.__np_d_rpm

    # Save the data
    def b_save_data(self, str_data_prefix='test_class', idx_file=1):
        """
        Save the data in the object to a .csv file

        Parameters
        ----------
        str_data_prefix : string
            String with file prefix (defaults to 'test_class')
        idx_file : integer
            File name index (defaults to 1)

        Return values:
        True if write succeeds

        """

        # Construct the file name and open it
        self.__str_file = str_data_prefix + '_' '%03.0f' % idx_file + '.csv'
        file_data = open(self.__str_file, 'w+')

        # Construct the header
        self.__i_header_rows = 4
        str_header = 'X'
        str_fs = 'Sampling Frequency (Hz)'
        str_delta_t = 'Delta Time (seconds)'
        str_units = 'Sequence'
        for idx, class_signal in enumerate(self.__lst_cl_sgs):
            str_header = str_header + "," + class_signal.str_point_name
            str_fs = str_fs + "," + '%0.6f' % class_signal.d_fs
            str_delta_t = str_delta_t + "," + '%0.8f' % (self.__lst_cl_sgs[idx].d_time[1] -
                                                         self.__lst_cl_sgs[idx].d_time[0])

            str_units = str_units + class_signal.str_eu + ","

        str_header = str_header + '\n'
        str_fs = str_fs + '\n'
        str_delta_t = str_delta_t + '\n'
        str_units = str_units + '\n'
        file_data.write(str_header)
        file_data.write(str_fs)
        file_data.write(str_delta_t)
        file_data.write(str_units)

        for idx_line in range(0, self.i_ns):

            # line number
            str_line = str(idx_line)

            # add samples from each signal to the file
            for cl_obj in self.__lst_cl_sgs:
                str_line = str_line + ',' + '%0.8f' % cl_obj.np_d_sig[idx_line]

            # terminate the line
            file_data.write(str_line + '\n')

        file_data.close()

        return True

    # Retrieve the data for the whole file
    def b_read_data_as_df(self, str_filename=None):

        """
        Read the entire file in as a pandas dataframe

        Parameters
        ----------
        str_filename : string
            Filename, including .csv extension,  to read. If None then filename stored
            in the class is used

        Returns
        --------
        lst_data : list
                pandas dataframe : dataframe with all data from the file
                numpy array : vector with signal sampling rates
                numpy array : vector with delta time interval for each signal

        """

        # Pull the filename from the object if nothing is specified
        if str_filename is None:
            str_filename = self.str_file

        # Open the file and read the file headers in
        file_handle = open(str_filename)
        csv_reader = csv.reader(file_handle)
        csv_header = next(csv_reader)
        csv_fs = next(csv_reader)
        csv_delta_t = next(csv_reader)
        file_handle.close()

        # Parse the header information
        i_signals = len(csv_fs)
        assert len(csv_header) == i_signals, 'Inconsistent number of channels in file'
        assert len(csv_delta_t) == i_signals, 'Inconsistent number of channels in file'
        d_fs = np.array(list(map(float, csv_fs[1:i_signals])))
        d_delta_t = np.array(list(map(float, csv_delta_t[1:i_signals])))

        # Read the file as a dataframe
        df_sig = pd.read_csv(str_filename, header=None,
                             skiprows=self.__i_header_rows, names=csv_header[0:5])

        # Return the data
        return [df_sig, d_fs, d_delta_t]
