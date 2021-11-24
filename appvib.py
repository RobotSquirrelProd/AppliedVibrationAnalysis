import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy.fft import rfft, rfftfreq
import abc as abc


class ClSig(abc.ABC):
    """Class to manage signals. Abstract base class"""

    @property
    @abc.abstractmethod
    def np_sig(self):
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

    @abc.abstractmethod
    def set_ylim_tb(self, ylim_tb):
        pass


class ClSigReal(ClSig):
    """
    Class for storing, plotting, and manipulating real-valued signals

    """

    def __init__(self, np_sig, d_fs):
        super(ClSigReal, self).__init__()
        self.__b_complex = False
        self.__np_sig = np_sig
        self.__d_fs = d_fs
        self.__i_ns = self.__get_num_samples()
        self.__ylim_tb = [0]
        self.set_ylim_tb(self.__ylim_tb)

        # Setup the s-g array and filtering parameters
        self.__np_sig_filt_sg = np_sig
        self.__i_win_len = 31
        self.__i_poly_order = 1
        self.__str_filt_sg_desc = 'No Savitsky-Golay filtering'
        self.__str_filt_sg_desc_short = 'No S-G Filter'
        self.__b_update_filt_sg = True

        # Setup the butterworth FIR filtered signal vector and parameters
        self.__np_sig_filt_butter = np_sig
        self.__i_poles = 1
        self.__d_wn = 0.
        self.__str_filt_butter_desc = 'No Butterworth filtering'
        self.__str_filt_butter_desc_short = 'No Butter'

        # Final step: since this is instantiation, flag new signal in class
        self.__set_new_sig(True)

    @property
    def np_sig(self):
        """Numpy array containing the signal"""
        return self.__np_sig

    @np_sig.setter
    def np_sig(self, np_sig):
        """
        Update the signal vector. This update forces a recalculation of all derived parameters.

        Parameters
        ----------
        np_sig : numpy array
            Vector with the signal of interest. Must be real-valued.

        """
        # With a new signal, all the filtering will have to be done
        if np.iscomplexobj(np_sig):
            raise Exception("Must be a real-valued signal vector")

        # Store the vector into the object, reset filtering state, and update related features
        self.__np_sig = np_sig
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
            self.__b_update_filt_sg = True

    @property
    def b_complex(self):
        return self.__b_complex

    # Calculate the number of samples in the signal
    def __get_num_samples(self):
        """Calculate number of samples in the signal"""
        return len(self.__np_sig)

    @property
    def i_ns(self):
        self.__i_ns = self.__get_num_samples()
        return self.__i_ns

    @property
    def ylim_tb(self):
        """Real-valued Timebase vertical limits
        @return: plot y-limits
        """
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
                [np.max(self.__np_sig), np.min(self.__np_sig)])

    @property
    def str_filt_sg_desc(self):
        """Long Savitsky-Golay description"""
        return self.__str_filt_sg_desc

    @property
    def str_filt_sg_desc_short(self):
        """Short Savitsky-Golay description"""
        return self.__str_filt_sg_desc_short

    @property
    def np_sig_filt_sg(self):
        """ Return the signal, filtered with Savitsky-Golay"""

        # Does the filter need to be applied (signal updated) or can
        # we return the prior instance?
        if self.__b_update_filt_sg:

            # If there are enough samples, filter
            if self.i_ns > self.__i_win_len:
                self.__np_sig_filt_sg = sig.savgol_filter(self.np_sig,
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
                self.__np_sig_filt_sg = self.np_sig
                self.__str_filt_sg_desc = 'No Savitsky-Golay filtering'
                self.__str_filt_sg_desc_short = 'No S-G Filter'

            # Flag that the filtering is done
            self.__b_update_filt_sg = False

        return self.__np_sig_filt_sg

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
    def np_sig_filt_butter(self):
        """ Return the signal, filtered with butterworth FIR filter"""

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
        self.__np_sig_filt_butter = sig.sosfilt(sos, self.np_sig)

        # Generate the plain text descriptions for the plots
        self.__str_filt_butter_desc = ('Butterworth | Poles: ' +
                                       '%2.f' % self.__i_poles +
                                       ' | Lowpass corner (Hz): ' +
                                       '%0.2f' % self.__d_wn)
        self.__str_filt_butter_desc_short = 'Butter'
        return self.__np_sig_filt_butter


class ClSigComp(ClSig):
    """Class for storing, plotting, and manipulating complex-valued
       signals"""

    def __init__(self, np_sig, d_fs):
        super(ClSigComp, self).__init__()
        self.__b_complex = True
        self.__np_sig = np_sig
        self.__d_fs = d_fs
        self.__i_ns = self.__get_num_samples()
        self.__ylim_tb = [0]
        self.set_ylim_tb(self.__ylim_tb)

    @property
    def np_sig(self):
        """Numpy array containing the signal"""
        self.__i_ns = self.__get_num_samples()
        return self.__np_sig

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
        return len(self.__np_sig)

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
                [np.max(self.__np_sig), np.min(self.__np_sig)])


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
        timebase_scale: 1.000
        i_ns:   3
        d_t_del: 4.000
        d_time[0. 4. 8.]

        Attributes
        ----------

        Methods
        -------
    """

    def __init__(self, np_sig, d_fs):
        """
        Parameters
        ----------
        np_sig : numpy array
            Vector with the signal of interest. Can be real- or complex-valued.
        d_fs : double
            Describes the sampling frequency in samples/second (hertz).
        """
        # Instantiation of class so begin list and add first signal
        self.__lst_cl_sgs = []
        self.__lst_b_active = []

        # Instantiate and save common signal features to this class
        self.idx_add_sig(np_sig, d_fs_in=d_fs)
        self.__d_time = self.__get_d_time
        self.__np_d_rpm = np.zeros_like(self.__lst_cl_sgs[0].np_sig)

        self.__d_thresh = np.NaN
        self.__d_events_per_rev = np.NaN
        self.__str_eu = 'volts'
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
        return self.__lst_cl_sgs[0].np_sig

    def get_np_d_sig(self, idx=0):
        """Numpy array containing arbitrary signal"""
        return self.__lst_cl_sgs[idx].np_sig

    @np_d_sig.setter
    def np_d_sig(self, lst_in):
        np_sig_in = lst_in[0]
        idx = lst_in[1]
        self.__lst_cl_sgs[idx].np_sig = np_sig_in
        self.__lst_b_active[idx] = True

    def idx_add_sig(self, np_sig_in, d_fs_in):
        """Add another signal to this object.
        returns index to the newly added signal.
        """

        # TO DO: try/catch might be a better option here
        # Does the incoming signal have the same number of
        # samples as the ones already present?
        if len(self.__lst_cl_sgs) > 0:

            if len(np_sig_in) != self.i_ns:
                raise Exception('Cannot add signal with different number of samples')

        # Add the signals, looking for complex and real
        self.__lst_cl_sgs.append(ClSigReal(np_sig_in, d_fs_in))
        if np.iscomplexobj(np_sig_in):
            self.__lst_cl_sgs[0] = ClSigComp(np_sig_in, d_fs_in)

        # Mark this one as active
        self.__lst_b_active.append(True)

        # Success, return True
        return len(self.__lst_cl_sgs) - 1

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
        return 1.0 / self.d_fs(idx)

    @property
    def __get_d_time(self):
        return np.linspace(0, (self.i_ns - 1), self.i_ns) * self.d_t_del()

    @property
    def d_time(self):
        """Numpy array with time values, in seconds"""
        self.__d_time = self.__get_d_time
        return self.__d_time

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
        d_fs : double
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

    @property
    def np_d_eventtimes(self):
        """Numpy array of trigger event times"""
        return self.__np_d_eventtimes

    @property
    def d_thresh(self):
        """Trigger threshold value"""
        return self.__d_thresh

    @property
    def np_d_rpm(self):
        """Estimated RPM values"""
        return self.__np_d_rpm

    @property
    def d_events_per_rev(self):
        """Events per revolution"""
        return self.__d_events_per_rev

    @property
    def str_eu(self):
        """Engineering unit descriptor"""
        return self.__str_eu

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

    @str_eu.setter
    def str_eu(self, str_eu):
        self.__str_eu = str_eu

    @str_plot_desc.setter
    def str_plot_desc(self, str_plot_desc):
        self.__str_plot_desc = str_plot_desc

    # Method for calculating the spectrum for a real signal
    def d_fft_real(self):
        """Calculate the half spectrum since this is a real-valued signal"""
        d_y = rfft(self.np_d_sig)
        self.__i_ns_rfft = len(d_y)

        # Scale the fft. I'm using the actual number
        # of points to scale.
        d_y = d_y / (self.__i_ns_rfft - 1)

        # Calculate the frequency scale
        d_ws = rfftfreq(self.i_ns, 1. / self.d_fs())

        # Return the values
        return [d_ws, d_y]

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
        ax1.plot(self.d_time, self.get_np_d_sig(idx=0))
        ax1.plot(self.d_time, self.__lst_cl_sgs[0].np_sig_filt_sg)
        ax1.plot(self.d_time, self.__lst_cl_sgs[0].np_sig_filt_butter)
        ax1.grid()
        ax1.set_xlabel("Time, seconds")
        ax1.set_ylabel("Channel output, " + self.__str_eu)
        ax1.set_ylim(self.__lst_cl_sgs[0].ylim_tb)
        ax1.set_title(self.__str_plot_desc + " Timebase")
        ax1.legend(['as-acquired', self.str_filt_sg_desc_short(),
                    self.str_filt_butter_desc_short()])

        # Channel 2
        if len(self.__lst_b_active) > 1:
            if self.__lst_b_active[1]:
                i_ch = 1
                axs[i_ch].plot(self.d_time, self.get_np_d_sig(idx=1))
                ax1.plot(self.d_time, self.__lst_cl_sgs[i_ch].np_sig_filt_sg)
                ax1.plot(self.d_time, self.__lst_cl_sgs[i_ch].np_sig_filt_butter)
                axs[i_ch].grid()
                axs[i_ch].set_xlabel("Time, seconds")
                axs[i_ch].set_ylabel("Channel output, " + self.__str_eu)
                axs[i_ch].set_ylim(self.__lst_cl_sgs[0].ylim_tb)
                axs[i_ch].set_title(self.__str_plot_desc + " Timebase")
                ax1.legend(['as-acquired', self.str_filt_sg_desc_short(idx=i_ch),
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
        handle to the plot
        """
        self.__spec = self.d_fft_real()
        d_mag = np.abs(self.__spec[1])
        plt.figure()
        plt.plot(self.__spec[0], d_mag)
        plt.grid()
        plt.xlabel("Frequency, hertz")
        plt.ylabel("Channel amplitude, " + self.__str_eu)
        plt.title(self.__str_plot_desc + " Spectrum")

        # Annotate the peak
        if self.__b_spec_peak:
            idx_max = np.argmax(d_mag)
            d_ws_peak = self.__spec[0][idx_max]
            d_ws_span = (self.__spec[0][-1] - self.__spec[0][0])
            d_mag_peak = d_mag[idx_max]
            plt.plot(d_ws_peak, d_mag_peak, 'ok')
            str_label = ('%0.3f' % d_mag_peak + ' ' +
                         self.__str_eu + ' @ ' + '%0.2f' % d_ws_peak + ' Hz')
            plt.annotate(str_label, [
                d_ws_peak + (0.02 * d_ws_span), d_mag_peak * 0.95])

        # Save off the handle to the plot
        self.__plot_handle = plt.gcf()

        # Show the plot, creating a new figure.
        plt.show()

        return [self.__plot_handle, self.__spec[0], self.__spec[1]]

    # Plotting method for the eventtimes
    def plt_eventtimes(self):
        """Plot event data in time.

        Return values:
        list: [handle to the plot, np array of eventtimes]
        """

        # The eventtimes all should have threshold value for voltage
        self.__np_d_eventvalue = np.ones_like(
            self.__np_d_eventtimes) * self.d_thresh

        # Put up the the plot time
        plt.figure()
        plt.plot(self.__d_time, self.np_d_sig)
        plt.plot(self.np_d_eventtimes, self.__np_d_eventvalue, "ok")
        plt.xlabel('Time, seconds')
        plt.ylabel('Amplitude, ' + self.__str_eu)
        plt.legend(['as-aquired', 'eventtimes'])
        plt.title(self.__str_plot_desc + ' Amplitude and eventtimes vs. time')
        self.__plot_handle = plt.gcf()
        return [self.__plot_handle, self.__np_d_eventtimes]

    # Plotting method for the eventtimes
    def plt_rpm(self):
        """Plot rpm data in time.

        Return values:
        list: [handle to the plot, np array of RPM values]
        """

        # Put up the the plot time
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()

        ax1.plot(self.__d_time, self.np_d_sig)
        ax2.plot(self.np_d_eventtimes, self.__np_d_rpm, "ok")
        ax1.set_xlabel('Time, seconds')
        ax1.set_ylabel('Amplitude, ' + self.__str_eu)
        ax2.set_ylabel('Event speed, RPM')
        plt.legend(['as-aquired', 'RPM'])
        plt.title('Amplitude and eventtimes vs. time')
        plt.show()

        self.__plot_handle = plt.gcf()
        return [self.__plot_handle, self.__np_d_rpm]

    # Estimate triggers for speed,  public method
    def np_d_est_triggers(self, i_direction=0, d_thresh=0,
                          d_hyst=0.1, i_kernel=5, b_verbose=False):
        """
        This method estimates speed by identifying trigger points in time,
        a given threshold and hysteresis. When the signal level crosses
        the threshold, the trigger holds off. The trigger holds off
        until the signal crosses the hysteresis level. Hysteresis is
        defined relative to the threshold voltage.

        The trigger times can be used to estimate the rotating speed.

        Keyword arguments:
        i_direction -- 0 to search for threshold on rising signal, 1 to search
                        on a falling signal.
        d_thresh  -- Threshold value (default: 0.0 volts for zero crossings)
        d_hyst -- Hysteresis value (default: 0.1 volts)
        i_kernel -- Number of samples to consider in estimating slope,
                        must be an odd number (default: 5)
        b_verbose -- Print the intermediate steps (default: False). Useful
                        for stepping through the method to troubleshoot or
                        understand it better.

        Return values:
        np_d_eventtimes -- numpy array with list of trigger event times
        """

        # Store to local private member, it gets used in other places
        # in the class
        self.__d_thresh = d_thresh

        # Initialize trigger state to hold off: the trigger will be active
        # once the signal crosses the hysteresis
        b_trigger_hold = True

        # half kernel get used a lot
        self.__i_half_kernel = int((i_kernel - 1) / 2.)

        # Use smoothing and derivative functions of S-G filter for
        # estimating rise/fall
        self.__np_d_ch1_dir = sig.savgol_filter(self.np_d_sig,
                                                i_kernel, 1, deriv=1)

        # Initiate state machine: one state for rising signal,
        # 'up', (i_direction = 0) and another for falling signal,
        # 'down', (i_direction = 1)
        self.__d_hyst_abs = 0.
        idx_event = 0
        self.__np_d_eventtimes = np.zeros_like(self.np_d_sig)
        if i_direction == 0:

            # Define the absolute hysteretic value, rising
            self.__d_hyst_ab = self.__d_thresh - d_hyst

            # Loop through the signal
            for idx, x in enumerate(self.np_d_sig):

                # Intermediate results
                if b_verbose:
                    print('idx: ' + '%2.f' % idx + ' | x: ' + '%0.5f' % x +
                          ' | s-g: ' + '%0.4f' % self.__np_d_ch1_dir[idx])

                # The trigger leaves 'hold-off' state if the slope is
                # negative and we fall below the threshold
                if (x <= self.__d_hyst_ab and self.__np_d_ch1_dir[idx] < 0 and
                        b_trigger_hold):
                    # Next time the signal rises above the threshold, trigger
                    # will be set to hold-off state
                    b_trigger_hold = False

                # If we are on the rising portion of the signal and there is
                # no hold off state on the trigge then trigger, and change
                # state
                if (x >= self.__d_thresh and self.__np_d_ch1_dir[idx] > 0 and
                        not (b_trigger_hold)):

                    # Change state to hold off
                    b_trigger_hold = True

                    # Estimate time of crossing with interpolation
                    if idx > 0:

                        # Interpolate to estimate the actual crossing from
                        # the 2 nearest points
                        xp = np.array(
                            [self.np_d_sig[idx - 1], self.np_d_sig[idx]])
                        fp = np.array(
                            [self.__d_time[idx - 1], self.__d_time[idx]])
                        self.__np_d_eventtimes[idx_event] = np.interp(
                            d_thresh, xp, fp)

                        # More intermediate results
                        if b_verbose:
                            print('xp: ' + np.array2string(xp) + ' | fp: ' +
                                  np.array2string(fp) + ' | d_thresh: ' +
                                  '%0.4f' % d_thresh + ' | eventtimes: ' +
                                  '%0.4f' % self.__np_d_eventtimes[idx_event])

                        # Increment the eventtimes index
                        idx_event += 1

        else:

            # Define the absolute hysteretic value, falling
            self.__d_hyst_ab = self.__d_thresh + d_hyst

            # Loop through the signal
            for idx, x in enumerate(self.np_d_sig):

                # Intermediate results
                if b_verbose:
                    print('idx: ' + '%2.f' % idx + ' | x: ' + '%0.5f' % x +
                          ' | s-g: ' + '%0.4f' % self.__np_d_ch1_dir[idx])

                # The trigger leaves 'hold-off' state if the slope is
                # positive and we rise above the threshold
                if (x >= self.__d_hyst_ab and self.__np_d_ch1_dir[idx] > 0 and
                        b_trigger_hold):
                    # Next time the signal rises above the threshold, trigger
                    # will be set to hold-off state
                    b_trigger_hold = False

                # If we are on the falling portion of the signal and
                # there is no hold off state on the trigger then trigger
                # and change state
                if (x <= self.__d_thresh and self.__np_d_ch1_dir[idx] < 0 and
                        not (b_trigger_hold)):

                    # Change state to hold off
                    b_trigger_hold = True

                    # Estimate time of crossing with interpolation
                    if idx > 0:

                        # Interpolate to estimate the actual crossing from
                        # the 2 nearest points
                        xp = np.array(
                            [self.np_d_sig[idx - 1], self.np_d_sig[idx]])
                        fp = np.array(
                            [self.__d_time[idx - 1], self.__d_time[idx]])
                        self.__np_d_eventtimes[idx_event] = np.interp(
                            d_thresh, xp, fp)

                        # More intermediate results
                        if b_verbose:
                            print('xp: ' + np.array2string(xp) + ' | fp: ' +
                                  np.array2string(fp) + ' | d_thresh: ' +
                                  '%0.4f' % d_thresh + ' | eventtimes: ' +
                                  '%0.4f' % self.__np_d_eventtimes[idx_event])

                        # Increment the eventtimes index
                        idx_event += 1

        # Remove zero-valued element
        self.__np_d_eventtimes = np.delete(
            self.__np_d_eventtimes, np.where(self.__np_d_eventtimes == 0))

        return self.__np_d_eventtimes

    # Method to estimate the RPM values
    def d_est_rpm(self, d_events_per_rev=1):
        """
        Estimate the RPM from the signal using eventtimes which must have
        calculated from a previous call to the method np_d_est_triggers.
        """

        # Store the new value in the object
        self.__d_events_per_rev = d_events_per_rev

        # Calculate the RPM using the difference in event times
        self.__np_d_rpm = 60. / \
                          (np.diff(self.np_d_eventtimes) * float(d_events_per_rev))

        # To keep the lengths the same, append the last sample
        self.__np_d_rpm = np.append(
            self.__np_d_rpm, self.__np_d_rpm[len(self.__np_d_rpm) - 1])

        return self.__np_d_rpm

    # Save the data
    def b_save_data(self, str_data_prefix='testclass', idx_data=1):
        """
        Save the data in the object to a .csv file

        Keyword arguments:
        str_data_prefix -- String with file prefix (defaults to 'testclass')
        idx_data -- File index (defaults to 1)

        Return values:
        True if write succeeds

        """
        self.__str_file = str_data_prefix + '_' '%03.0f' % idx_data + '.csv'
        file_data = open(self.__str_file, 'w+')
        str_header = 'X,'
        str_units = 'Sequence,'
        idx_ch = 1
        for b_obj in self.__lst_b_active:
            str_header = str_header + 'CH' + '%0.0f' % idx_ch + ','
            str_units = str_units + 'Volt,'
            idx_ch = idx_ch + 1

        str_header = str_header + 'Start,Increment,\n'
        str_units = str_units + 'Volt,0.000000e-03,' + str(self.d_t_del) + '\n'
        file_data.write(str_header)
        file_data.write(str_units)

        for idx_line in range(0, self.i_ns):
            str_line = str(idx_line)

            # add samples from each signal to the file
            for cl_obj in self.__lst_cl_sgs:
                str_line = str_line + ',' + '%0.5f' % cl_obj.np_sig[idx_line]

            file_data.write(str_line + '\n')

        file_data.close()

        return True
