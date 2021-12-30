import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import transforms
from matplotlib.artist import Artist
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as mticker
import math
import numpy as np
import csv
import pandas as pd
import scipy.signal as sig
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from datetime import datetime, timedelta
from dateutil import tz
import abc as abc


class ClassPlotSupport:

    @staticmethod
    def get_dt_str_utc_conv(dt_timestamp):
        """
        Method to format date and time for plots, files, etc. with a conversion
        from UTC to local.

        Parameters
        ----------
        dt_timestamp : datetime
            Datetime for conversion

        Returns
        -------
        str_dt_timestamp : string
            Formatted date-time string

        """
        # Convert from UTC to local time and then to string
        dt_timestamp_working = dt_timestamp
        dt_timestamp_working = dt_timestamp_working.replace(tzinfo=tz.tzutc())
        dt_local = dt_timestamp_working.astimezone(tz.tzlocal())
        str_dt_timestamp = ClassPlotSupport.get_dt_str(dt_local)

        return str_dt_timestamp

    @staticmethod
    def get_dt_str(dt_timestamp):
        """
        Method to format date and time for plots, files, etc.

        Parameters
        ----------
        dt_timestamp : datetime
            Datetime for conversion

        Returns
        -------
        str_dt_timestamp : string
            Formatted date-time string

        """
        # Convert from datetime object to string
        str_dt_timestamp = dt_timestamp.isoformat(sep=' ', timespec='milliseconds')

        return str_dt_timestamp

    @staticmethod
    def get_idx_by_dt(np_d_time, dt_timestamp, dt_timestamp_mark):
        """
        Method to format date and time for plots, files, etc.

        Parameters
        ----------
        np_d_time : numpy array, double
            Time stamp for each sample, assumed to be seconds from the dt_timestamp value
        dt_timestamp : datetime
            Date time value for the first time sample in np_d_time
        dt_timestamp_mark : datetime
            Date time value for closest index

        Returns
        -------
        integer : index of the closest value to dt_timestamp_mark

        """

        # Construct datetime array
        np_dt_series = np.full(len(np_d_time), dt_timestamp, dtype=datetime)
        for idx, dt in enumerate(np_dt_series):
            np_dt_series[idx] = dt+timedelta(seconds=np_d_time[idx])

        return (np.abs(np_dt_series-dt_timestamp_mark)).argmin()

    @staticmethod
    def get_plot_round(d_num):
        """
        Why another round function? Indeed, this one is designed to give a reasonable
        look to the grid lines. This function is mocked up in RoundingTest.ipynb

        Parameters
        ----------
        d_num : double
            Number to be rounded

        Return
        ------
        list
            double : number rounded off
            string : format string
        """
        d_exp = np.log10(abs(d_num))
        i_round = int(round(d_exp, 0)) - 1

        if i_round <= 0:
            str_format = '%0.' + '%0.0f' % -i_round + 'f'
        else:
            str_format = '%' + '%0.0f' % i_round + '.0f'

        return [round(d_num, -i_round), str_format]

    @staticmethod
    def get_plot_setup_rows():
        """
        Global function to provide common number of grid rows

        @return:
            integer : number of rows


        """
        return 18

    @staticmethod
    def get_plot_setup_row_sig():
        """
        Global function to provide common starting point for rectilinear time-based plot data

        @return:
            integer : row at which to start plotting signal data


        """
        return 5 + ClassPlotSupport.get_plot_setup_rows_header_buffer()

    @staticmethod
    def get_plot_setup_row_sparklines():
        return ClassPlotSupport.get_plot_setup_row_sig() - + ClassPlotSupport.get_plot_setup_rows_header_buffer()

    @staticmethod
    def get_plot_setup_rows_header_buffer():
        return 1

    @staticmethod
    def get_plot_setup_row_sig_span():
        """
        Global function to provide common row span for rectilinear time-based plot data

        @return:
            integer : row span for plotting signal data


        """
        return 10

    @staticmethod
    def set_plot_setup_sig_axis(ax, x_limit_sig=np.array([0.0, 1.0]), y_limit_sig=np.array([0.0, 1.0]),
                                str_eu='volts', str_yaxis_desc='Amplitude'):
        """Set up the signal plotting axis

        Parameters
        ----------
        ax : matplotlib axes object
            Axis object for the signal plotting area
        x_limit_sig : list, double
            x-axis minimum and maximum values
        y_limit_sig : list, double
            y-axis minimum and maximum values
        str_eu : string
            y-axis units
        str_yaxis_desc : string
            y-axis description

        """

        # grid features
        i_xaxis_minor = 5

        # from: https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Set up the grid
        ax.grid(True, which='major', color="#666666ff", lw=0.5)
        ax.grid(True, which='minor', color="#666666ff", lw=0.25, ls=':')

        # x-axis grid and labels
        d_spacing = (max(x_limit_sig) - min(x_limit_sig)) / 5.0
        lst_round = ClassPlotSupport.get_plot_round(d_spacing / float(i_xaxis_minor))
        d_spacing_rounded = lst_round[0] * 5.0
        str_format = lst_round[1]
        ax.xaxis.set_major_locator(MultipleLocator(d_spacing_rounded))
        ax.xaxis.set_minor_locator(AutoMinorLocator(i_xaxis_minor))

        # Remove horizontal axis tick marks
        for tick in ax.xaxis.get_major_ticks():
            # Left bottom tick marker
            tick.tick1line.set_visible(False)

        for tick in ax.xaxis.get_minor_ticks():
            # Left bottom tick marker
            tick.tick1line.set_visible(False)

        # Set the x-ticks, just labelling the first point
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        lst_labels = [str_format % x for x in ticks_loc]
        for idx in range(2, len(lst_labels)):
            lst_labels[idx] = ' '
        ax.set_xticklabels(lst_labels)

        # Spacing is not quite right with ticks removed
        ax.tick_params(axis='x', pad=-2)

        # X-axis label
        str_suffix = 'sec'
        d_spacing_rounded_minor = d_spacing_rounded / float(i_xaxis_minor)
        if d_spacing_rounded_minor < 1.0:
            d_spacing_rounded_minor = d_spacing_rounded_minor * 1000
            str_suffix = 'ms'
        lst_format_xaxis_division = ClassPlotSupport.get_plot_round(d_spacing_rounded_minor)
        str_xaxis_description = ('Horizontal: ' + lst_format_xaxis_division[1] % d_spacing_rounded_minor +
                                 ' ' + str_suffix + '/division')
        ax.text(1, -0.1, str_xaxis_description, horizontalalignment='right', verticalalignment='top',
                fontweight='bold', transform=ax.transAxes)

        # y-axis grid and labels.
        d_mid_span = (max(y_limit_sig) - min(y_limit_sig)) / 2.0
        d_spacing = 1.1 * d_mid_span

        # Set up the spacing and number format for the major axis
        lst_round = ClassPlotSupport.get_plot_round(d_spacing)
        d_spacing_rounded = lst_round[0]
        str_format = lst_round[1]
        ax.yaxis.set_major_locator(MultipleLocator(d_spacing_rounded))
        ax.yaxis.set_minor_locator(AutoMinorLocator(i_xaxis_minor))

        # Remove vertical axis tick marks
        for tick in ax.yaxis.get_major_ticks():
            # Left/bottom tick marker
            tick.tick1line.set_visible(False)

        for tick in ax.yaxis.get_minor_ticks():
            # Left/bottom tick marker
            tick.tick1line.set_visible(False)

        # Spacing is not quite right with ticks removed
        ax.tick_params(axis='y', pad=-2)

        # Y-axis label
        d_spacing_rounded_minor = d_spacing_rounded / float(i_xaxis_minor)
        str_yaxis_description = ('Vertical: ' + str_format % d_spacing_rounded_minor +
                                 ' ' + str_eu + '/division')
        ax.text(0, -0.1, str_yaxis_description, horizontalalignment='center', verticalalignment='top',
                fontweight='bold', transform=ax.transAxes)
        ax.text(0, 1.09, str_yaxis_desc, horizontalalignment='center', verticalalignment='top',
                fontweight='bold', transform=ax.transAxes)

        return

    @staticmethod
    def get_plot_setup_cols():
        """
        Global function to provide common number of grid columns

        @return:
            integer : number of columns


        """
        return 8

    @staticmethod
    def get_plot_sparkline_desc(str_format, d_value, str_eu, str_eu_secondary):
        """
        Helper function to provide consistent formatting for the spark line descriptions

        Parameters
        str_format : string
            Formatting string for the value
        d_value : float
            Numerical value to be included in the header
        str_eu : string
            Primary engineering unit descriptor (volts, mils, g's, etc.)
        str_eu_secondary : string
            Secondary engineering unit descriptor (pp, pk, rms, etc.)

        Returns
            string : sparkline description
        """
        return str_format % d_value + ' ' + str_eu + ' ' + str_eu_secondary

    @staticmethod
    def get_font_plots():
        """
        Global function to provide common fonts across multiple plots

        @return:
            string : font name

        """
        return 'Garamond'

    @staticmethod
    def get_trac_color(i_trace):
        """
         Global function that returns trace colors

         Parameters
         ----------
             i_trace : integer
                 0-based index to the trace on the plot

         Returns
         -------
             str_color : string with the color code to use

         """
        match i_trace:
            case 0:
                # Blue hue
                return "#619fccff"
            case 1:
                # Red hue
                return "#af4d57ff"
            case 2:
                # Light green hue
                return "#9dae87ff"
            case 3:
                # Light orange hue
                return "#d99f77ff"
            case _:
                # Charcoal
                return "#1a1a1aff"

    @staticmethod
    def get_text_trunc(str_data):
        """Truncate text string"""
        return (str_data[:35] + '...') if len(str_data) > 35 else str_data

    @staticmethod
    def set_plot_header_desc(i_rows, i_cols, i_row_offset, str_plot_desc):
        """
        Global function that creates the description field and value in the header

         Parameters
         ----------
             i_rows : integer
                 Number of rows in the grid
             i_cols : integer
                 Number of columns in the grid
             i_row_offset : integer
                Number of rows to skip; useful for multi-pane plots
            str_plot_desc : string
                Description string

        Returns:

            axes_desc : handle to description axis

        """
        # Header pane, starting with the description
        axs_desc = plt.subplot2grid((i_rows, i_cols), (i_row_offset, 0), colspan=int(i_cols / 4), rowspan=1)
        axs_desc.axis('off')
        axs_desc.text(0, 1, 'Description:', horizontalalignment='right', verticalalignment='top',
                      fontweight='bold')
        axs_desc.text(0, 1, ' ' + ClassPlotSupport.get_text_trunc(str_plot_desc), horizontalalignment='left',
                      verticalalignment='top',
                      fontweight='bold')

    @staticmethod
    def set_plot_header_machine(i_rows, i_cols, i_row_offset, str_machine_name):
        """
        Global function that creates the description field and value in the header

         Parameters
         ----------
             i_rows : integer
                 Number of rows in the grid
             i_cols : integer
                 Number of columns in the grid
             i_row_offset : integer
                Number of rows to skip; useful for multi-pane plots
            str_machine_name : string
                Machine name string

        Returns:

            axes_desc : handle to machine name axis

        """
        # Machine description
        axs_mach = plt.subplot2grid((i_rows, i_cols), (i_row_offset, 0), colspan=int(i_cols / 4), rowspan=1)
        axs_mach.axis('off')
        axs_mach.text(0, 1, 'Machine:', horizontalalignment='right', verticalalignment='top',
                      fontweight='bold')
        axs_mach.text(0, 1, ' ' + ClassPlotSupport.get_text_trunc(str_machine_name), horizontalalignment='left',
                      verticalalignment='top',
                      fontweight='bold')

    @staticmethod
    def draw_multi_color_text(x, y, lst_text, str_delim=', ', b_newline=False, **kw):
        """
        Draw a list of strings in color order of the plot traces. Inspired by
        a post on SO: https://stackoverflow.com/questions/9169052/partial-coloring-of-text-in-matplotlib

        Parameters
        ----------

            x : double
                Horizontal location of text
            y : double
                Vertical location of text
            lst_text : list, string
                List of string
            str_delim : string
                Delimiter to be used
            b_newline : boolean
                Set to true to force a newline at the end of each data.
                Defaults to False.

        """

        t = plt.gca().transData
        fig = plt.gcf()
        i_items = len(lst_text)

        for idx_str, str_label in enumerate(lst_text):

            # Draw the text
            text = plt.text(x, y, str_label, color=ClassPlotSupport.get_trac_color(idx_str), transform=t, **kw)
            text.draw(fig.canvas.get_renderer())
            ex = text.get_window_extent()
            d_ex_width = ex.width
            t = transforms.offset_copy(text._transform, x=1 * d_ex_width, units='dots')

            # Punctuation
            if idx_str < (i_items - 1):
                text = plt.text(x, y, str_delim, color='black', transform=t, **kw)
                text.draw(fig.canvas.get_renderer())
                ex = text.get_window_extent()
                if b_newline:
                    t = transforms.offset_copy(text._transform, x=-1 * d_ex_width, y=-1 * ex.height, units='dots')
                else:
                    t = transforms.offset_copy(text._transform, x=1 * ex.width, units='dots')

    @staticmethod
    def set_plot_header_point(i_rows, i_cols, i_row_offset, lst_str_point_name):
        """
        Global function that creates the point name field and value in the header

         Parameters
         ----------
             i_rows : integer
                 Number of rows in the grid
             i_cols : integer
                 Number of columns in the grid
             i_row_offset : integer
                Number of rows to skip; useful for multi-pane plots
            lst_str_point_name : list, string
                Lit of point name strings, assumed to be in order of plotted traces

        Returns:

            axes_desc : handle to point name axis

        """
        # Point name
        axs_point = plt.subplot2grid((i_rows, i_cols), (i_row_offset, 0), colspan=int(i_cols / 4), rowspan=1)
        axs_point.axis('off')
        axs_point.text(0, 1, 'Point(s):', horizontalalignment='right', verticalalignment='top',
                       fontweight='bold')
        lst_str_point_name[0] = " " + lst_str_point_name[0]
        ClassPlotSupport.draw_multi_color_text(0, 1, lst_str_point_name, horizontalalignment='left',
                                               verticalalignment='top',
                                               fontweight='bold')

    @staticmethod
    def set_plot_header_date(i_rows, i_cols, i_row_offset, lst_dt_timestamp):
        """
        Global function that creates the date field and value in the header

         Parameters
         ----------
             i_rows : integer
                 Number of rows in the grid
             i_cols : integer
                 Number of columns in the grid
             i_row_offset : integer
                Number of rows to skip; useful for multi-pane plots
            lst_dt_timestamp : list, datetime
                Lit of timestamps, assumed to be in order of plotted traces

        Returns:

            axes_desc : handle to timestamp name axis

        """
        # Date value
        axs_date = plt.subplot2grid((i_rows, i_cols), (i_row_offset, 0), colspan=int(i_cols / 4), rowspan=1)
        axs_date.axis('off')
        axs_date.text(0, 1, 'Date(s):', horizontalalignment='right', verticalalignment='top',
                      fontweight='bold')
        lst_str_timestamp = list(map(lambda x: ClassPlotSupport.get_dt_str_utc_conv(x), lst_dt_timestamp))
        lst_str_timestamp = list(map(lambda x: ' ' + x, lst_str_timestamp))
        ClassPlotSupport.draw_multi_color_text(0, 1, lst_str_timestamp, str_delim=', ', b_newline=True,
                                               horizontalalignment='left',
                                               verticalalignment='top',
                                               fontweight='bold')

    @staticmethod
    def set_plot_sparkline(i_rows, i_cols, i_row_offset, np_cl_spark, dt_timestamp_mark):
        """
        Global function that creates the description field for the sparklines

         Parameters
         ----------
             i_rows : integer
                 Number of rows in the grid
             i_cols : integer
                 Number of columns in the grid
             i_row_offset : integer
                Number of rows to skip; useful for multi-pane plots
            np_cl_spark : numpy array, ClSigCompUneven
                Array with signals to be plotted in the sparklines
            dt_timestamp_mark : datetime
                Date and time to place marker

        Returns:

            axes_desc : handle to description axis

        """

        # Header pane, sparklines
        i_col_offset = int(ClassPlotSupport.get_plot_setup_cols() / 2)

        # How many sparklines to plot?
        i_sparklines = min([ClassPlotSupport.get_plot_setup_row_sparklines(), len(np_cl_spark)])

        # Iterate through each sparkline that has data and plot it
        dt_timestamp_start = []
        dt_timestamp_end = []
        for idx_spk in range(i_sparklines):
            # Sparkline
            i_row_plot = (i_row_offset + ClassPlotSupport.get_plot_setup_row_sparklines() - idx_spk - 1)
            axs_spk1 = plt.subplot2grid((i_rows, i_cols), (i_row_plot, i_col_offset),
                                        colspan=i_col_offset - 1, rowspan=1)
            ln_full = axs_spk1.plot(np_cl_spark[idx_spk].np_d_time, np_cl_spark[idx_spk].np_d_sig, 'k', linewidth=0.5)
            axs_spk1.axis('off')
            axs_spk1.axvline(c='black', lw=0.25)
            d_max_time = np.max(np_cl_spark[idx_spk].np_d_time)
            axs_spk1.axvline(x=d_max_time, c='black', lw=0.25)
            Artist.remove(ln_full[0])
            np_d_y = 0.9 * np_cl_spark[idx_spk].np_d_sig
            axs_spk1.plot(np_cl_spark[idx_spk].np_d_time, np_d_y,
                          'k', linewidth=0.35)

            # Set the vertical limits on the sparkline. This is really important
            # to have the correct aspect ratio so critical features surface, but
            # computational noise does not. By default, the ylim_tb are set to
            # None on instantiation. In that case, make a good faith effort to
            # get the aspect ratio correct based on the signal amplitude.
            if (np_cl_spark[idx_spk].ylim_tb is None) or (len(np_cl_spark[idx_spk].ylim_tb) < 2):
                d_ylim_min = 0.9 * np.min(np_d_y)
                d_ylim_max = 1.1 * np.max(np_d_y)
                # Very small signal show up with numerical noise, this is an attempt
                # to get the scaling large enough to show significant changes in trends,
                # but not show numerical noise
                # TODO: Ideally the scaling would be set to optimize feature detection,
                #  something like the 45 degrees suggested by Tufte
                if abs(d_ylim_max - d_ylim_min) < 0.5:
                    d_ylim_min = d_ylim_min - 0.5
                    d_ylim_max = d_ylim_max + 0.5

            # The user as set the limits, use those values
            else:
                d_ylim_min = min(np_cl_spark[idx_spk].ylim_tb)
                d_ylim_max = max(np_cl_spark[idx_spk].ylim_tb)

            # Set the y-axis limits
            axs_spk1.set_ylim([d_ylim_min, d_ylim_max])

            # Add the marker, colored to match the first trace on the plot
            dt_timestamp_start = np_cl_spark[idx_spk].dt_timestamp
            dt_timestamp_end = dt_timestamp_start + timedelta(seconds=np_cl_spark[idx_spk].np_d_time[-1] -
                                                                      np_cl_spark[idx_spk].np_d_time[0])
            dt_time_series = np.array([dt_timestamp_start + timedelta(seconds=dt_iter)
                                       for dt_iter in np_cl_spark[idx_spk].np_d_time])
            idx_marker = abs(dt_time_series - dt_timestamp_mark).argmin()
            d_x = np_cl_spark[idx_spk].np_d_time[idx_marker]
            d_y = 0.9 * np_cl_spark[idx_spk].np_d_sig[idx_marker]
            axs_spk1.plot(d_x, d_y,
                          '.', color=ClassPlotSupport.get_trac_color(0), ms=3)

            # Description
            ClassPlotSupport.set_plot_spark_desc(i_rows, i_cols, i_row_plot,
                                                 np_cl_spark[idx_spk].str_point_name)

        # Add the time labels to this last sparkline
        idx_spk = ClassPlotSupport.get_plot_setup_row_sparklines() - 1
        axs_spk1 = plt.subplot2grid((i_rows, i_cols), (i_row_offset + idx_spk + 1, i_col_offset),
                                    colspan=i_col_offset - 1, rowspan=1)
        axs_spk1.axis('off')
        axs_spk1.text(0, 0, ClassPlotSupport.get_dt_str_utc_conv(dt_timestamp_start),
                      horizontalalignment='center', verticalalignment='bottom', fontsize='small')
        axs_spk1.text(1, 0, ClassPlotSupport.get_dt_str_utc_conv(dt_timestamp_end),
                      horizontalalignment='center', verticalalignment='bottom', fontsize='small')

    @staticmethod
    def set_plot_spark_desc(i_rows, i_cols, i_row_offset, str_spark_desc):
        """
        Global function that creates the description field for the sparklines

         Parameters
         ----------
             i_rows : integer
                 Number of rows in the grid
             i_cols : integer
                 Number of columns in the grid
             i_row_offset : integer
                Number of rows to skip; useful for multi-pane plots
            str_spark_desc : string
                Description string

        Returns:

            axes_desc : handle to description axis

        """
        # Header pane, starting with the description
        axs_desc = plt.subplot2grid((i_rows, i_cols), (i_row_offset, ClassPlotSupport.get_plot_setup_cols() - 1),
                                    colspan=1, rowspan=1)
        axs_desc.axis('off')
        axs_desc.text(-0.20, 1, ' ' + str_spark_desc, horizontalalignment='left', verticalalignment='top',
                      fontweight='bold')


class ClSignalFeaturesEst:
    """This class has mostly static methods used to estimate signal features"""

    @staticmethod
    def np_d_est_amplitude(np_d_sig=np.array([1., 2., 3.])):
        """
        This method estimates the signal envelope (peak values) by analytic signals

        Parameters
        ----------
        np_d_sig : numpy array
            Vector with the signal of interest. Assumed to be real-valued.

        Return
        ------

        np_d_amp : numpy array, double
            Vector with signal envelope (amplitude)

        """

        np_d_analytic = hilbert(np_d_sig - np.mean(np_d_sig))
        return np.abs(np_d_analytic)

    @staticmethod
    def np_d_est_pk():
        """TODO: Derive and implement peak detector function"""
        return True

    @staticmethod
    def np_d_est_rms(np_d_sig=np.array([1., 2., 3.]), i_break=128, i_kernel=200):
        """
        This method estimates the root-mean square (RMS) value of a signal. For
        signal with less than i_break samples a single RMS value is calculated.
        For signals with more than i_break samples a rolling RMS is calculated
        where i_kernel defines the kernel sample count.

        Parameters
        ----------
        np_d_sig : numpy array
            Vector with the signal of interest. Assumed to be real-valued.
        i_break : integer
            Signal with samples less than i_break have a single RMS value
            calculated. Samples with more than i_break samples have a rolling
            RMS calculate where the kernel length is defined by i_kernel
        i_kernel : integer
            Number of samples in the rolling RMS kernel

        Return
        ------
        np_d_amp : numpy array, double
            Vector with rms


        """

        # Condition the signal
        np_d_sig = np_d_sig - np.mean(np_d_sig)

        # If the signal is short, a single RMS value is calculated for the entire signal.
        # for longer signals, a rolling RMS is needed
        np_d_rms = np.ones_like(np_d_sig)
        i_ns = len(np_d_sig)
        if len(np_d_sig) < i_break:

            # One value for the signal, replicated for the length of the signal
            d_rms = np.sqrt(np_d_sig.dot(np_d_sig) / np_d_sig.size)
            np_d_rms = np_d_rms * d_rms

        else:

            # Rolling rms value
            xc = np.cumsum(abs(np_d_sig) ** 2)
            np_d_rms_rolling = np.sqrt((xc[i_kernel:] - xc[:-i_kernel]) / i_kernel)

            # It takes i_kernel samples to accumulate the RMS value
            np_d_rms[i_kernel:i_ns] = np_d_rms_rolling
            np_d_rms[0:i_kernel] = np.mean(np_d_rms_rolling)

        return np_d_rms

    @staticmethod
    def np_d_est_mean(np_d_sig=np.array([1., 2., 3.]), i_break=500, i_kernel=1200):
        """
        This method estimates the signal average value. For a
        signal with less than i_break samples a single mean value is calculated.
        For signals with more than i_break samples a rolling mean is calculated
        where i_kernel defines the kernel sample count.

        Parameters
        ----------
        np_d_sig : numpy array
            Vector with the signal of interest. Assumed to be real-valued.
        i_break : integer
            Signal with samples less than i_break have a single mean value
            calculated. Samples with more than i_break samples have a rolling
            mean calculate where the kernel length is defined by i_kernel
        i_kernel : integer
            Number of samples in the rolling mean kernel

        Return
        ------
        np_d_amp : numpy array, double
            Vector with rms


        """

        # If the signal is short, a single mean value is calculated for the entire signal.
        # for longer signals, a rolling mean is needed
        np_d_avg = np.ones_like(np_d_sig)
        i_ns = len(np_d_sig)

        # If the signal is the same length as i_kernel, avoid recursion
        if i_ns == i_kernel:
            i_break = i_ns + 10

        # Process the signal
        if len(np_d_sig) < i_break:

            # One value for the signal, replicated for the length of the signal
            np_d_avg = np_d_avg * np.mean(np_d_sig)

        else:

            # Rolling average
            np_d_avg_kernel = np.cumsum(np_d_sig, dtype=float)
            np_d_avg_kernel[i_kernel:] = np_d_avg_kernel[i_kernel:] - np_d_avg_kernel[:-i_kernel]
            np_d_avg_kernel = np_d_avg_kernel[i_kernel:] / float(i_kernel)

            # It takes i_kernel samples to accumulate the mean value
            np_d_avg[i_kernel:i_ns] = np_d_avg_kernel
            np_d_avg[0:i_kernel] = np.mean(np_d_avg_kernel)

        return np_d_avg


class ClSig(abc.ABC):
    """

    Abstract base class to manage signals

    """

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
        """Point name for the signal"""
        pass

    @property
    @abc.abstractmethod
    def str_machine_name(self):
        """Machine name"""
        pass

    @property
    @abc.abstractmethod
    def dt_timestamp(self):
        """signal timestamp"""
        pass

    @abc.abstractmethod
    def set_ylim_tb(self, ylim_tb):
        pass


class ClSigReal(ClSig, ClSignalFeaturesEst):
    """
    Class for storing, plotting, and manipulating real-valued signals

    ...

    Attributes
    ----------

    Methods
    ----------

    """

    def __init__(self, np_d_sig, d_fs, str_eu='volts', str_point_name='CH1', str_machine_name='Machine',
                 dt_timestamp=datetime.fromisoformat('1990-01-01T00:00:00-00:00'),
                 dt_timestamp_mark=datetime.fromisoformat('1990-01-01T00:00:00-00:00')):
        """
        Parameters
        ----------
        np_d_sig : numpy array
            Vector with real-valued signal of interest
        d_fs : double
            Sampling frequency, hertz
        str_eu : string
            Engineering units. Defaults to 'volts'
        str_point_name : string
            Signal point name
        str_machine_name : string
            Machine train name
        dt_timestamp : datetime
            Signal timestamp in datetime stamp. This is the timestamp when the first sample was acquired.
            Defaults to 1-Jan-1970 UTC Timezone
        dt_timestamp_mark : datetime
            Timestamp of feature measurement in the sparklines. Defaults to 1-Jan-1970 UTC Timezone
        """
        # Parent class
        super(ClSigReal, self).__init__()

        # Store the signal and set the data status flags using the setter method.
        self.np_d_sig = np_d_sig

        # Signal metadata
        self.__b_complex = False
        self.__np_d_nx = np.zeros_like(np_d_sig)
        self.__d_fs = d_fs
        self.__b_is_stale_fs = True
        self.__str_eu = str_eu
        self.__str_point_name = str_point_name
        self.__str_machine_name = str_machine_name
        self.__dt_timestamp = dt_timestamp
        self.__str_timedate_format = "%m/%d/%Y, %H:%M:%S.%f"

        # Derived features for the signal
        self.__i_ns = self.__get_num_samples()
        self.__d_time = np.array([0, 1])
        self.__d_time_max = 0.0
        self.__d_time_min = 0.0
        np_x = np.linspace(0, 127, 128)
        self.__np_sparklines = np.array([], dtype=ClSigCompUneven)
        for idx_spark in range(ClassPlotSupport.get_plot_setup_row_sparklines()):
            self.__np_sparklines = np.append(self.__np_sparklines,
                                             [ClSigCompUneven(np.random.rand(np.size(np_x)), np_x)])
            self.__np_sparklines[idx_spark].str_point_name = 'Sparkline ' + '%0.0f' % idx_spark
            self.__np_sparklines[idx_spark].ylim_tb = None
        self.__dt_timestamp_mark = dt_timestamp_mark
        self.__b_is_stale_sparkline = True

        # Timebase plot attributes. Some/many are derived from the signal
        # itself, and they need to be in this object, even though other
        # objects could be generating the plot.
        self.__d_time_plot = self.__d_time
        self.__ylim_tb = [0]
        self.set_ylim_tb(self.__ylim_tb)
        self.__i_x_divisions_tb = 12
        self.__i_y_divisions_tb = 9
        self.__str_eu_x = 'seconds'
        self.__str_plot_desc = ' '

        # Set up the s-g array and filtering parameters
        self.__np_d_sig_filt_sg = np_d_sig
        self.__i_win_len = 31
        self.__i_poly_order = 1
        self.__str_filt_sg_desc = 'No Savitsky-Golay filtering'
        self.__str_filt_sg_desc_short = 'No S-G Filter'
        self.__b_is_stale_filt_sg = True

        # Set up the butterworth FIR filtered signal vector and parameters
        self.__np_d_sig_filt_butter = np_d_sig
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
        self.__np_d_eventtimes = np.zeros_like(np_d_sig)
        self.__idx_events = np.zeros_like(np_d_sig)
        self.__b_is_stale_eventtimes = True

        # Attributes related to rpm calculation
        self.__d_events_per_rev = np.NaN
        self.__np_d_rpm = np.zeros_like(np_d_sig)
        self.__b_is_stale_rpm = True

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
            Set to True to re-calculate all functions dependent on either signal values or
            sampling frequency.

        """
        if b_state:
            self.__b_is_stale_fs = True
            self.__b_is_stale_filt_sg = True
            self.__b_is_stale_filt_butter = True
            self.__b_is_stale_eventtimes = True
            self.__b_is_stale_rpm = True
            self.__b_is_stale_sparkline = True

    @property
    def str_eu(self):
        return self.__str_eu

    @str_eu.setter
    def str_eu(self, str_eu_in):
        self.__str_eu = str_eu_in

    @property
    def dt_timestamp(self):
        return self.__dt_timestamp

    @dt_timestamp.setter
    def dt_timestamp(self, dt_timestamp):
        self.__dt_timestamp = dt_timestamp

    @property
    def dt_timestamp_mark(self):
        return self.__dt_timestamp_mark

    @dt_timestamp_mark.setter
    def dt_timestamp_mark(self, dt_timestamp_mark):
        self.__dt_timestamp_mark = dt_timestamp_mark

    @property
    def str_point_name(self):
        return self.__str_point_name

    @str_point_name.setter
    def str_point_name(self, str_point_name):
        self.__str_point_name = str_point_name

    @property
    def str_machine_name(self):
        return self.__str_machine_name

    @str_machine_name.setter
    def str_machine_name(self, str_machine_name):
        self.__str_machine_name = str_machine_name

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
        if self.__b_is_stale_filt_butter:

            # This is a guess of the filter corner, useful for general vibration
            # analysis of physical displacements.
            # TODO: This needs to be own method, should not be setting this here
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

    @d_threshold.setter
    def d_threshold(self, d_threshold):
        """Update the threshold level"""
        self.__d_threshold = d_threshold

    # The value of this attribute can be read, but it should
    # not be set, outside of the estimate crossings methods, since
    # any change requires a re-calculation of the eventtimes
    @property
    def d_hysteresis(self):
        return self.__d_hysteresis

    @property
    def i_direction(self):
        return self.__i_direction

    @property
    def idx_events(self):

        # Update the eventtimes if stale
        if self.__b_is_stale_eventtimes:
            self.np_d_est_triggers(np_d_sig=None, i_direction=None, d_threshold=None,
                                   d_hysteresis=None, b_verbose=False)
        if self.__b_is_stale_nx:
            self.calc_nx()

        return self.__idx_events

    # This is effectively set with the estimate crossings methods
    @property
    def np_d_eventtimes(self):

        # Update the eventtimes if stale
        if self.__b_is_stale_eventtimes:
            self.np_d_est_triggers(np_d_sig=None, i_direction=None, d_threshold=None,
                                   d_hysteresis=None, b_verbose=False)
        return self.__np_d_eventtimes

    # Interpolation of points for instantaneous frequency estimation
    def calc_interpolate_crossing(self, np_d_sig, idx, b_verbose=False):

        """
        This method estimates time of crossing using linear interpolation

        Parameters
        ----------
        idx  : integer
            This is the sample index immediately after the trigger changed to active. The
            function assumes the prior sample was taken before the trigger state changed.
        np_d_sig : numpy array
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
        xp = np.array([np_d_sig[idx], np_d_sig[idx + 1]])
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
    def np_d_est_triggers(self, np_d_sig=None, i_direction=None, d_threshold=None,
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
        np_d_sig : numpy array, None
            Signal to be evaluated for crossings. It can be any signal, but the class is designed
            for the input to be one of the signals already defined in the class so that an example
            looks like: np_d_sig=class_test_real.np_d_sig. This defaults to 'None' and assigns the
            class attribute 'np_sig' to 'np_d_sig'
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
        if np_d_sig is None:

            # Copy the class vector into this method
            np_d_sig = self.np_d_sig
        else:
            # User is possibly adding a new signal, force recalculation
            self.__b_is_stale_eventtimes = True
            self.__b_is_stale_nx = True
            self.__b_is_stale_rpm = True

        if i_direction is not None:
            # User is possibly adding a new direction, force recalculation
            self.__i_direction = i_direction
            self.__b_is_stale_eventtimes = True
            self.__b_is_stale_nx = True
            self.__b_is_stale_rpm = True
            if b_verbose:
                print('i_direction: ' + '%1.0f' % self.__i_direction)

        if d_threshold is not None:
            # User is possibly adding a new threshold, force recalculation
            self.__d_threshold = d_threshold
            self.__b_is_stale_eventtimes = True
            self.__b_is_stale_nx = True
            self.__b_is_stale_rpm = True
            if b_verbose:
                print('d_threshold: ' + '%0.4f' % self.__d_threshold)

        if d_hysteresis is not None:
            # User is possibly adding a new hysteresis, force recalculation
            self.__d_hysteresis = d_hysteresis
            self.__b_is_stale_eventtimes = True
            self.__b_is_stale_nx = True
            self.__b_is_stale_rpm = True
            if b_verbose:
                print('d_threshold: ' + '%0.4f' % self.__d_threshold)

        # Validate attributes
        assert (np.min(np_d_sig) < self.d_threshold < np.max(np_d_sig)), \
            'Threshold is outside signal range'

        # Run the calculation if stale data is present
        if self.__b_is_stale_eventtimes:

            # Initialize trigger state to hold off: the trigger will be active
            # once the signal crosses the hysteresis
            b_trigger_hold = True

            # Initiate state machine: one state for rising signal,
            # 'up', (i_direction = 0) and another for falling signal,
            # 'down', (i_direction = 1)
            idx_event = 0
            self.__np_d_eventtimes = np.zeros_like(np_d_sig)
            if self.__i_direction == 0:

                # Define the absolute hysteretic value, rising
                d_hysteresis_abs = self.d_threshold - self.d_hysteresis
                if b_verbose:
                    print('Direction is 0 (rising edge)')
                    print('d_hysteresis_abs: ' + '%0.4f' % d_hysteresis_abs)

                # Loop through the signal, excluding the last point
                for idx, x in enumerate(np_d_sig[0:-1]):

                    # Intermediate results
                    if b_verbose:
                        print('idx: ' + '%2.f' % idx + ' | x: ' + '%0.5f' % x +
                              ' | np_d_sig: ' + '%0.4f' % np_d_sig[idx])

                    # Only the sign matters so subtract this point from next to
                    # get sign of slope
                    d_slope = np_d_sig[idx + 1] - np_d_sig[idx]

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
                    if (x < self.d_threshold <= np_d_sig[idx + 1]) and d_slope > 0 and not b_trigger_hold:
                        # Change state to hold off
                        b_trigger_hold = True
                        if b_verbose:
                            print('Triggered, rising')

                        # Estimate time of crossing with interpolation
                        self.__np_d_eventtimes[idx_event] = self.calc_interpolate_crossing(np_d_sig, idx)

                        # Increment the eventtimes index
                        idx_event += 1

            else:

                # Define the absolute hysteretic value, falling
                d_hysteresis_abs = self.d_threshold + self.d_hysteresis

                # Loop through the signal
                for idx, x in enumerate(np_d_sig[0:-1]):

                    # Intermediate results
                    if b_verbose:
                        print('idx: ' + '%2.f' % idx + ' | x: ' + '%0.5f' % x +
                              ' | s-g: ' + '%0.4f' % np_d_sig[idx])

                    # Only the sign matters so subtract this point from next to
                    # get sign of slope
                    d_slope = np_d_sig[idx + 1] - np_d_sig[idx]

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
                    if (x > self.d_threshold >= np_d_sig[idx + 1]) and d_slope < 0 and not b_trigger_hold:
                        # Change state to hold off
                        b_trigger_hold = True
                        if b_verbose:
                            print('Triggered, falling')

                        # Estimate time of crossing with interpolation
                        self.__np_d_eventtimes[idx_event] = self.calc_interpolate_crossing(np_d_sig, idx)

                        # Increment the eventtimes index
                        idx_event += 1

            # Remove zero-valued element
            self.__np_d_eventtimes = np.delete(
                self.__np_d_eventtimes, np.where(self.__np_d_eventtimes == 0))

            # Freshly updated eventtimes
            self.__b_is_stale_eventtimes = False

            # Since the eventtimes were calculated the nX vectors have to marked
            # as stale. The RPM values also have to be marked as stale.
            self.__b_is_stale_nx = True
            self.__b_is_stale_rpm = True

        # Create vector of indexes to the nearest points, needed for plotting
        self.__idx_events = np.round(self.__np_d_eventtimes * self.d_fs, 0).astype(int)

        # Return the list of eventtimes.
        return self.__np_d_eventtimes

    @property
    def np_d_rpm(self):

        # Is the data stale?
        if self.__b_is_stale_rpm:
            self.np_d_est_triggers(self.np_d_sig)
            self.np_d_est_rpm()

        """Estimated RPM values"""
        return self.__np_d_rpm

    @property
    def d_events_per_rev(self):
        """Number of events per revolution"""
        return self.__d_events_per_rev

    @d_events_per_rev.setter
    def d_events_per_rev(self, d_events_per_rev):
        """Update the number of events in the object"""
        # Force a re-calculation of RPM
        self.__b_is_stale_rpm = True
        self.__d_events_per_rev = d_events_per_rev

    # Method to estimate the RPM values
    def np_d_est_rpm(self, d_events_per_rev=1.0):
        """
        Estimate the RPM from the signal using eventtimes which must have
        calculated from a previous call to the method np_d_est_triggers.

        Parameters
        ---------
        d_events_per_rev : double
            Number of events per revolution. It must be a real value to for hunting tooth gear
            combinations. Defaults to 1

        Returns
        -------
        numpy array, double : array of RPM values

        """

        # Store the new value in the object
        self.__d_events_per_rev = d_events_per_rev

        # Validate internal attributes
        assert (len(self.np_d_eventtimes) > 1), 'Not enough trigger events to estimate RPM'

        # Calculate the RPM using the difference in event times
        self.__np_d_rpm = 60. / (np.diff(self.np_d_eventtimes) * d_events_per_rev)

        # To keep the lengths the same, append the last sample
        self.__np_d_rpm = np.append(
            self.__np_d_rpm, self.__np_d_rpm[len(self.__np_d_rpm) - 1])

        return self.__np_d_rpm

    @property
    def np_d_nx(self):
        return self.__np_d_nx

    # Estimate nX vectors, given trigger events and a signal
    def calc_nx(self, np_d_sig=None, np_d_eventtimes=None, b_verbose=False):
        """
        This method estimates the 1X vectors, given trigger event times. The
        phase reported in this estimation is intended to be used for balancing
        so phase lag is positive (spectral phase lag is negative). Since this is 
        implemented in the real signal class, the method assumes the signal
        is also real. 

        Parameters
        ----------
        np_d_sig : numpy array
            Signal to be evaluated for crossings. Should reference a signal already loaded
            into the object (i.e. np_d_sig = {ClSigReal}.np_d_sig). Setting 'np_d_sig' to None
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
        if np_d_sig is None:

            # Copy the class vector into this method
            np_d_sig = self.np_d_sig

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
                d_np_y = rfft(np_d_sig[idx_start:idx_end])
                i_ns_rfft = len(d_np_y)

                # Scale the fft using the actual number
                # of points to scale.
                d_np_y = d_np_y / float(i_ns_rfft)

                # Grab the second bin since it is the best estimate
                # of the sinusoid with the same frequency as the
                # spacing of eventtimes
                d_nx[idx] = d_np_y[1] * (-1.0 + 0.0j)

                # This is needed to correct for offset introduced by a non-zero threshold setting
                d_cor = -2.0 * (np.angle(d_nx[idx]) - np.deg2rad(90.0))
                d_nx[idx] = d_nx[idx] * np.exp(d_cor * 1j)

                # Print summary
                if b_verbose:
                    print('idx_start: ' + '%5.0f' % idx_start + ' | idx_end: ' +
                          '%5.0f' % idx_end + ' | nX mag: ' + '%2.6f' % abs(d_nx[idx]) +
                          ' | %2.6f' % np.rad2deg(np.angle(d_nx[idx])) + ' deg.')

            # Pad the end
            if len(d_nx) > 0:
                d_nx[-1] = d_nx[-2]

            # Update calculation status
            self.__b_is_stale_nx = False

            # Update the complex class
            self.__class_sig_comp = ClSigCompUneven(d_nx, self.__np_d_eventtimes)

        # Save to class attribute
        self.__np_d_nx = self.__class_sig_comp.np_d_sig

        # Return the values
        return self.__np_d_nx

    @property
    def np_sparklines(self):
        # If the signal has been updated, but not the sparklines,
        # this variable will be set to True and the method estimates
        # sparklines for just the signal itself
        if self.__b_is_stale_sparkline:

            # First sparkline is the pk-pk value
            np_d_pk = ClSignalFeaturesEst.np_d_est_amplitude(self.np_d_sig)
            np_d_pkpk = 2.0 * np_d_pk
            d_pkpk_mean = np.mean(np_d_pkpk)
            lst_fmt = ClassPlotSupport.get_plot_round(d_pkpk_mean)
            str_point_spark = ClassPlotSupport.get_plot_sparkline_desc(lst_fmt[1],
                                                                       d_pkpk_mean,
                                                                       self.str_eu,
                                                                       'pp')
            self.__np_sparklines[0] = ClSigCompUneven(np_d_pkpk, self.d_time, str_eu=self.str_eu,
                                                      str_point_name=str_point_spark,
                                                      str_machine_name=self.str_machine_name,
                                                      dt_timestamp=self.dt_timestamp)

            # Second sparkline is the rms value
            np_d_rms = ClSignalFeaturesEst.np_d_est_rms(self.np_d_sig)
            d_rms_avg = np.mean(np_d_rms)
            lst_fmt = ClassPlotSupport.get_plot_round(d_rms_avg)
            str_point_spark = ClassPlotSupport.get_plot_sparkline_desc(lst_fmt[1],
                                                                       d_rms_avg,
                                                                       self.str_eu,
                                                                       'rms')
            self.__np_sparklines[1] = ClSigCompUneven(np_d_rms, self.d_time, str_eu=self.str_eu,
                                                      str_point_name=str_point_spark,
                                                      str_machine_name=self.str_machine_name,
                                                      dt_timestamp=self.dt_timestamp)

            # Third sparkline is the crest value
            np_d_crest = np.divide(np_d_pk, np_d_rms)
            d_crest_avg = np.mean(np_d_crest)
            lst_fmt = ClassPlotSupport.get_plot_round(d_crest_avg)
            str_point_spark = ClassPlotSupport.get_plot_sparkline_desc(lst_fmt[1],
                                                                       d_crest_avg,
                                                                       'crest factor',
                                                                       '')
            self.__np_sparklines[2] = ClSigCompUneven(np_d_crest, self.d_time, str_eu=self.str_eu,
                                                      str_point_name=str_point_spark,
                                                      str_machine_name=self.str_machine_name,
                                                      dt_timestamp=self.dt_timestamp)

            # Fourth is average
            np_d_mean = ClSignalFeaturesEst.np_d_est_mean(self.np_d_sig)
            d_mean_avg = np.mean(np_d_mean)
            lst_fmt = ClassPlotSupport.get_plot_round(d_mean_avg)
            str_point_spark = ClassPlotSupport.get_plot_sparkline_desc(lst_fmt[1],
                                                                       d_mean_avg,
                                                                       self.str_eu,
                                                                       'mean')
            self.__np_sparklines[3] = ClSigCompUneven(np_d_mean, self.d_time, str_eu=self.str_eu,
                                                      str_point_name=str_point_spark,
                                                      str_machine_name=self.str_machine_name,
                                                      dt_timestamp=self.dt_timestamp)

            # Fifth is rpm. Sparklines look wierd if the lines start at different spots
            np_d_spark_rpm = np.array([-10., -10.])
            # There is a bit of a catch 22 with this sparkline. RPM values can only be estimated
            # if the threshold and hysteresis are set correctly. The analyst often needs
            # to see the signal to set the values. Using a try-catch structure that allows
            # a sparkline to be plotted so the signal can also be plotted.
            try:
                np_d_spark_eventtimes = np.insert(self.np_d_eventtimes, 0, self.d_time[0])
                np_d_spark_rpm = np.insert(self.np_d_rpm, 0, self.np_d_rpm[0])
            except AssertionError:
                np_d_spark_eventtimes = np.array([self.d_time[0], self.d_time[-1]])

            d_mean_rpm = np.mean(np_d_spark_rpm)
            lst_fmt = ClassPlotSupport.get_plot_round(d_mean_rpm)
            str_point_spark = ClassPlotSupport.get_plot_sparkline_desc(lst_fmt[1],
                                                                       d_mean_rpm,
                                                                       'RPM',
                                                                       '')
            self.__np_sparklines[4] = ClSigCompUneven(np_d_spark_rpm, np_d_spark_eventtimes, str_eu=self.str_eu,
                                                      str_point_name=str_point_spark,
                                                      str_machine_name=self.str_machine_name,
                                                      dt_timestamp=self.dt_timestamp)

        return self.__np_sparklines

    @np_sparklines.setter
    def np_sparklines(self, np_sparklines):

        # New data, sparkline is updated and does not require further processing
        self.__b_is_stale_sparkline = False

        # Return the value
        self.__np_sparklines = np_sparklines

    @property
    def str_plot_desc(self):
        return self.__str_plot_desc

    @str_plot_desc.setter
    def str_plot_desc(self, str_plot_desc):
        self.__str_plot_desc = str_plot_desc
        self.__class_sig_comp.str_plot_desc = str_plot_desc

    @property
    def __str_format_dt(self):
        """
        Method to format date and time for plots, files, etc.

        Parameters
        ----------

        Returns
        -------
        str_dt_timestamp : string
            Formatted date-time string

        """
        # Convert from UTC to local time and then to string
        dt_timestamp_working = self.dt_timestamp
        dt_timestamp_working = dt_timestamp_working.replace(tzinfo=tz.tzutc())
        dt_local = dt_timestamp_working.astimezone(tz.tzlocal())
        str_dt_timestamp = dt_local.isoformat(sep=' ', timespec='milliseconds')

        return str_dt_timestamp

    # Call the method to plot the apht plot
    def plt_apht(self, str_plot_apht_desc=None, b_verbose=False):

        """
        Plot out the vector data (amplitude and phase) in trend format

        Parameters
        ----------
        str_plot_apht_desc : string
            Signal metadata description for plot title. Defaults to None which uses the
            plot description as-found in the object.
        b_verbose : boolean
            Print the intermediate steps (default: False). Useful for stepping through the
            method to troubleshoot or understand it better.

        Returns
        -------
        handle to the plot

        """

        # Create plot title from class attributes
        str_meta = str_plot_apht_desc
        if str_plot_apht_desc is None:
            str_meta = self.str_plot_desc + '\n' + 'apht' + ' | ' + self.str_point_name + \
                       ' | ' + self.__str_format_dt

        return self.__class_sig_comp.plt_apht(str_plot_apht_desc=str_meta, b_verbose=b_verbose)

    # Call polar plotting method.
    def plt_polar(self, str_plot_desc=None):
        """Plot out amplitude in phase in polar format

        Parameters
        ----------
        str_plot_desc : string
            Additional title text for the plot. If 'None' then method uses class attribute.

        Return values:
        handle to the plot
        """

        return self.__class_sig_comp.plt_polar(str_plot_desc)


class ClSigComp(ClSig):
    """Class for storing, plotting, and manipulating complex-valued
       signals"""

    def __init__(self, np_sig, d_fs, str_eu='volts', str_point_name='CH1', str_machine_name='Machine',
                 dt_timestamp=datetime.fromisoformat('1990-01-01T00:00:00-00:00')):

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
        str_machine_name : string
            Machine train name
        dt_timestamp : datetime
            Signal timestamp in datetime stamp. This is the timestamp when the first sample was acquired.
            Defaults to 1-Jan-1970 UTC Timezone
        """

        super(ClSigComp, self).__init__()
        self.__b_complex = True
        self.__np_d_sig = np_sig

        # Signal metadata
        self.__str_eu = str_eu
        self.__d_fs = d_fs
        self.__i_ns = self.__get_num_samples()
        self.str_point_name = str_point_name
        self.str_machine_name = str_machine_name
        self.dt_timestamp = dt_timestamp
        self.__str_timedate_format = "%m/%d/%Y, %H:%M:%S"

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
    def dt_timestamp(self):
        return self.__dt_timestamp

    @dt_timestamp.setter
    def dt_timestamp(self, dt_timestamp):
        self.__dt_timestamp = dt_timestamp

    @property
    def str_point_name(self):
        return self.__str_point_name

    @str_point_name.setter
    def str_point_name(self, str_point_name):
        self.__str_point_name = str_point_name

    @property
    def str_machine_name(self):
        return self.__str_machine_name

    @str_machine_name.setter
    def str_machine_name(self, str_machine_name):
        self.__str_machine_name = str_machine_name

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

    def __init__(self, np_sig, np_d_time, str_eu='volts', str_point_name='CH1', str_machine_name='Machine',
                 dt_timestamp=datetime.fromisoformat('1990-01-01T00:00:00-00:00')):

        """
        Parameters
        ----------
        np_sig : numpy array, double
            Vector with real-valued signal of interest
        np_d_time : numpy array, double
            Time stamp for each sample, assumed to be seconds from the dt_timestamp value
        str_eu : string
            Engineering units. Defaults to 'volts'
        str_point_name : string
            Signal point name
        str_machine_name : string
            Machine train name
        dt_timestamp : datetime
            Signal timestamp in datetime stamp. This is the timestamp when the first sample was acquired.
            Defaults to 1-Jan-1970 UTC Timezone
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
        self.str_machine_name = str_machine_name
        dt_timestamp_utc = dt_timestamp.astimezone(tz.tzutc())
        self.dt_timestamp = dt_timestamp_utc
        self.__str_timedate_format = "%m/%d/%Y, %H:%M:%S.%f"

        # Plotting attributes
        self.__ylim_mag = [0]
        self.set_ylim_mag(self.__ylim_mag)
        self.__ylim_tb = [0]
        self.set_ylim_tb(self.__ylim_tb)
        self.__str_plot_desc = '-'
        self.__str_plot_desc = '-'

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

            # Only change the limits if the signal is valid
            if len(self.__np_d_sig) > 0:
                self.__ylim_mag = np.array(
                    [1.05 * np.max(np.abs(self.__np_d_sig)), 0.95 * np.min(np.abs(self.__np_d_sig))])

    def set_ylim_tb(self, ylim_tb):
        """Y limits for magnitude timebase plot"""
        # Only use limits if they are valid. First check is if they
        # are set to None
        if ylim_tb is not None:
            if len(ylim_tb) == 2:
                self.__ylim_tb = ylim_tb
        else:
            # Only change limits if the signal is valid and has samples
            if len(self.__np_d_sig) > 0:
                self.__ylim_tb = np.array(
                    [np.max(np.abs(self.__np_d_sig)), np.abs(np.min(self.__np_d_sig))])

    @property
    def str_eu(self):
        return self.__str_eu

    @str_eu.setter
    def str_eu(self, str_eu_in):
        self.__str_eu = str_eu_in

    @property
    def dt_timestamp(self):
        return self.__dt_timestamp

    @dt_timestamp.setter
    def dt_timestamp(self, dt_timestamp):
        self.__dt_timestamp = dt_timestamp

    @property
    def str_point_name(self):
        return self.__str_point_name

    @str_point_name.setter
    def str_point_name(self, str_point_name):
        self.__str_point_name = str_point_name

    @property
    def str_machine_name(self):
        return self.__str_machine_name

    @str_machine_name.setter
    def str_machine_name(self, str_machine_name):
        self.__str_machine_name = str_machine_name

    @property
    def str_plot_desc(self):
        return self.__str_plot_desc

    @str_plot_desc.setter
    def str_plot_desc(self, str_plot_desc):
        self.__str_plot_desc = str_plot_desc

    @property
    def __str_format_dt(self):
        """
        Method to format date and time for plots, files, etc.

        Parameters
        ----------

        Returns
        -------
        str_dt_timestamp : string
            Formatted date-time string

        """
        # Convert from UTC to local time and then to string
        dt_timestamp_working = self.dt_timestamp
        dt_timestamp_working = dt_timestamp_working.replace(tzinfo=tz.tzutc())
        dt_local = dt_timestamp_working.astimezone(tz.tzlocal())
        str_dt_timestamp = dt_local.isoformat(sep=' ', timespec='milliseconds')

        return str_dt_timestamp

    # Plotting method, apht plots.
    def plt_apht(self, str_plot_apht_desc=None, b_verbose=False):
        """

        Plot out amplitude in phase in apht format

        Parameters
        ----------
        str_plot_apht_desc : string
            Signal metadata description for plot title. Defaults to ''
        b_verbose : boolean
            Print the intermediate steps (default: False). Useful for stepping through the
            method to troubleshoot or understand it better.

        Return values:
        handle to the plot
        """

        # Parse inputs
        if str_plot_apht_desc is not None:
            # Update class attribute
            self.__str_plot_desc = str_plot_apht_desc

        # Format and concatenate
        str_plot_desc_local = self.__str_plot_desc + '\n' + 'apht' + ' | ' + self.str_point_name + \
                               ' | ' + self.__str_format_dt

        # Figure with subplots
        plt.rcParams["font.family"] = ClassPlotSupport.get_font_plots()
        fig, axs = plt.subplots(2)

        if b_verbose:
            print(np.angle(self.__np_d_sig))

        # Plot the phase
        axs[0].plot(self.__np_d_time, np.rad2deg(np.angle(self.__np_d_sig)), color=ClassPlotSupport.get_trac_color(0))
        axs[0].grid()
        axs[0].set_xlabel("Time, seconds")
        axs[0].set_ylabel("Phase, degrees")
        axs[0].set_ylim([-360.0, 360.0])
        axs[0].set_title(str_plot_desc_local)

        # Plot the magnitude
        axs[1].plot(self.__np_d_time, np.abs(self.__np_d_sig), color=ClassPlotSupport.get_trac_color(0))
        axs[1].grid()
        axs[1].set_xlabel("Time, seconds")
        axs[1].set_ylabel("Magnitude, " + self.str_eu)
        axs[1].set_ylim([0, max(self.ylim_mag)])
        axs[1].set_title(self.__str_plot_desc)

        # Set the layout
        plt.tight_layout()

        # Save off the handle to the plot
        plot_handle = plt.gcf()

        # Show the plot, creating a new figure. This command resets the graphics context
        # so the plot handle has to be saved first.
        plt.show()

        return plot_handle

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
            self.__str_plot_desc = str_plot_polar_desc

        # Figure with subplots
        plt.rcParams["font.family"] = ClassPlotSupport.get_font_plots()
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # Polar plot
        ax.plot(np.angle(self.__np_d_sig), np.abs(self.__np_d_sig), color=ClassPlotSupport.get_trac_color(0))
        ax.set_rmax(np.max(self.ylim_mag))
        d_tick_radial = np.round(np.max(self.ylim_mag) / 4.0, decimals=1)
        ax.set_rticks([d_tick_radial, d_tick_radial * 2.0, d_tick_radial * 3.0, d_tick_radial * 4.0])
        ax.set_rlabel_position(-22.5)
        ax.grid(True)
        ax.set_title(self.__str_plot_desc)

        # Set the layout
        plt.tight_layout()

        # Save off the handle to the plot
        plot_handle = plt.gcf()

        # Show the plot, creating a new figure. This command resets the graphics context
        # so the plot handle has to be saved first.
        plt.show()

        return plot_handle


class ClSigFeatures(ClassPlotSupport):
    """
    Class to manage signal features on scope data and other signals

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

    def __init__(self, np_d_sig, d_fs, str_point_name='CH1', str_machine_name='Machine',
                 dt_timestamp=datetime.fromisoformat('1990-01-01T00:00:00-00:00')):
        """
        Parameters
        ----------
        np_d_sig : numpy array
            Vector with the signal of interest. Can be real- or complex-valued.
        d_fs : double
            Describes the sampling frequency in samples/second (hertz).
        str_point_name : string
            Signal point name
        str_machine_name : string
            Machine train name
        dt_timestamp : datetime
            Signal timestamp in datetime stamp. This is the timestamp when the first sample was acquired.
            Defaults to 1-Jan-1970 UTC Timezone.
        """

        # These are helper objects
        super(ClassPlotSupport, self).__init__()

        # Instantiation of class so begin list and add first signal
        self.__lst_cl_sgs = []
        self.__lst_b_active = []

        # Instantiate and save common signal features to this class
        self.idx_add_sig(np_d_sig, d_fs=d_fs, str_point_name=str_point_name, str_machine_name=str_machine_name,
                         dt_timestamp=dt_timestamp)

        # Attributes related to file save/read behavior
        self.__str_file = ''
        self.__i_header_rows = 10

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
        np_d_sig = lst_in[0]
        idx = lst_in[1]
        self.__lst_cl_sgs[idx].np_d_sig = np_d_sig
        self.__lst_b_active[idx] = True

    def idx_add_sig(self, np_d_sig, d_fs, str_point_name, str_machine_name='Machine',
                    dt_timestamp=datetime.fromisoformat('1990-01-01T00:00:00-00:00')):
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
        str_machine_name : string
            Machine train name
        dt_timestamp : datetime
            Signal timestamp in datetime stamp. This is the timestamp when the first sample was acquired.
            Defaults to 1-Jan-1970 UTC Timezone
        """

        # TODO: try/catch might be a better option here
        # This is an implicit dataframe structure so all signals need to be the same length.
        # Does the incoming signal have the same number of samples as the ones already present? The
        # reason this restriction is enforced is that there a number of ways to interpolate or pad
        # a signal to bring different sampling rates to the same length and it is not appropriate to
        # assume a method since it will change the signal content.
        if len(self.__lst_cl_sgs) > 0:

            if len(np_d_sig) != self.i_ns:
                raise Exception('Cannot add signal with different number of samples')

        # cast to numpy array
        np_d_sig = np.array(np_d_sig)

        # Add the signals, looking for complex and real
        dt_timestamp_utc = dt_timestamp.astimezone(tz.tzutc())
        if np.iscomplexobj(np_d_sig):
            self.__lst_cl_sgs.append(ClSigComp(np_d_sig, d_fs, dt_timestamp=dt_timestamp_utc))
        else:
            self.__lst_cl_sgs.append(ClSigReal(np_d_sig, d_fs, dt_timestamp=dt_timestamp_utc))

        # signal index number
        idx_class = len(self.__lst_cl_sgs) - 1

        # add signal meta data
        self.__lst_cl_sgs[idx_class].str_point_name = str_point_name
        self.__lst_cl_sgs[idx_class].str_machine_name = str_machine_name

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

    def d_threshold(self, idx=0):
        """
        Return the threshold level

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        return self.__lst_cl_sgs[idx].d_threshold

    def d_threshold_update(self, d_threshold, idx=0):
        """
        Set the threshold level

        Parameters
        ---------
        d_threshold : double
            Describes the trigger threshold level

        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        self.__lst_cl_sgs[idx].d_threshold = d_threshold

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
    def np_d_est_triggers(self, np_d_sig, i_direction=0, d_threshold=0.0,
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

    def np_d_rpm(self, idx=0):
        """
        Numpy array of rpm values

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        return self.__lst_cl_sgs[idx].np_d_rpm

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

    def str_eu(self, idx=0):
        """
        Return engineering unit descriptor

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        Returns
        -------
        str_eu : string
            String describing engineering units

        """
        return self.__lst_cl_sgs[idx].str_eu

    def str_eu_set(self, str_eu, idx=0):
        """
        Set engineering unit descriptor

        Parameter
        ---------
        str_eu : string
            String describing engineering units
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        self.__lst_cl_sgs[idx].str_eu = str_eu

    def str_point_name(self, idx=0):
        """
        Return signal point name

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        Returns
        -------
        str_eu : string
            Engineering unit string

        """
        return self.__lst_cl_sgs[idx].str_point_name

    def str_point_name_set(self, str_point_name, idx=0):
        """
        Set signal point name

        Parameter
        ---------
        str_point_name : string
            Signal point name
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        self.__lst_cl_sgs[idx].str_point_name = str_point_name

    def str_machine_name(self, idx=0):
        """
        Return machine name

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        Returns
        -------
        str_machine_name : string
            Machine name string

        """
        return self.__lst_cl_sgs[idx].str_machine_name

    def str_machine_name_set(self, str_machine_name, idx=0):
        """
        Set machine point name

        Parameter
        ---------
        str_machine_name : string
            Machine name string
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        self.__lst_cl_sgs[idx].str_machine_name = str_machine_name

    def dt_timestamp(self, idx=0):
        """
        Signal date and time

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        return self.__lst_cl_sgs[idx].dt_timestamp

    def dt_timestamp_mark(self, idx=0):
        """
        Feature measurement date and time in sparkline

        Parameter
        ---------
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        """
        return self.__lst_cl_sgs[idx].dt_timestamp_mark

    def dt_timestamp_mark_update(self, dt_timestamp_mark, idx=0):
        """
        Set feature measurement date and time in sparklines

        Parameter
        ---------
        dt_timestamp_mark : datetime
            Timestamp of feature measurement in the sparklines. Defaults to 1-Jan-1970 UTC Timezone
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)
        """
        self.__lst_cl_sgs[idx].dt_timestamp_mark = dt_timestamp_mark

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

    def ylim_tb(self, ylim_tb_in=None, idx=0):
        """
        Interface for the vertical plotting limits

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

    def __str_format_dt(self, idx=0):
        """
        Method to format date and time for plots, files, etc.

        Parameters
        ----------
        idx: integer
           Index of signal to pull description. Defaults to 0 (first signal)

        Returns
        -------
        str_dt_timestamp : string
            Formatted date-time string

        """
        # Convert from UTC to local time and then to string
        return ClassPlotSupport.get_dt_str_utc_conv(self.dt_timestamp(idx=idx))

    def __str_plt_support_title_meta(self, str_plot_type='timebase', idx=0):
        """

        Method to concatenate and format the signal meta data to string format for plot titles

        Parameters
        ----------
        str_plot_type : string
            String describing the plot title. Defaults to 'timebase'
        idx: integer
           Index of signal to pull description. Defaults to 0 (first signal)

        Returns
        -------
        str_meta : string
            String with meta data, formatted for title
        """

        # Format and concatenate
        str_meta = self.__str_plot_desc + '\n' + str_plot_type + ' | ' + self.str_point_name(idx=idx) + \
                   ' | ' + self.__str_format_dt(idx=idx)

        return str_meta

    def d_time_plot(self, idx=0):
        """
        Method to access time series for trace identified by idx

        Parameters
        ----------
        idx: integer
           Index of signal to pull time series. Defaults to 0 (first signal)

        Returns
        -------
        numpy array, double : NumPy array of double values representing the time in seconds

        """
        return self.__lst_cl_sgs[idx].d_time_plot

    def np_sparklines(self, idx=0):
        """
        Method to access sparkline array for trace identified by idx

        Parameters
        ----------
        idx: integer
           Index of signal to pull sparklines. Defaults to 0 (first signal)

        Returns
        -------
        numpy array, ClSigCompUneven : NumPy array of ClSigCompUneven objects used to create sparklines

        """
        return self.__lst_cl_sgs[idx].np_sparklines

    def np_sparklines_update(self, np_sparklines, idx=0):
        """
        Method to access sparkline array for trace identified by idx

        Parameters
        ----------
        np_sparklines : numpy array, ClSigCompUneven
            NumPy array of ClSigCompUneven objects used to create sparklines
        idx: integer
           Index of signal to set sparklines. Defaults to 0 (first signal)

        Returns
        -------

        """

        self.__lst_cl_sgs[idx].np_sparklines = np_sparklines

    # Plotting method, time domain signals.
    def plt_sigs(self, b_verbose=False, b_plot_sg=False, b_plot_filt=False):
        """
        Plot out the data in this signal feature class in the time domain

        Parameters
        ----------
        b_verbose : boolean
            Print the intermediate steps (default: False). Useful for stepping through the
            method to troubleshoot or understand it better.
        b_plot_sg : boolean
            Set to True to overlay the Savitsky-Golay filtered signal. Defaults to false
        b_plot_filt : boolean
            Set to True to overlay the FIR filtered signal. Defaults to false

        Returns
        -------
        plot_handle : handle to the plot

        """

        # How many plots, assuming 1 is given?
        i_plots = 0
        for b_obj in self.__lst_b_active:
            if b_obj:
                i_plots += 1

        # Figure with subplots
        plt.rcParams["font.family"] = ClassPlotSupport.get_font_plots()
        i_rows = ClassPlotSupport.get_plot_setup_rows() * i_plots
        i_cols = ClassPlotSupport.get_plot_setup_cols()

        # Step through the channels
        for idx_ch, _ in enumerate(self.__lst_cl_sgs):

            # Verbose documentation
            if b_verbose:
                print('idx_ch: ' + '%1.0f' % idx_ch)

            # Offset on a channel-by-channel basis
            i_row_offset = (idx_ch * ClassPlotSupport.get_plot_setup_rows())

            # Main signal pane, beginning with the signal
            idx_trace = 0
            i_row = (ClassPlotSupport.get_plot_setup_row_sig() + i_row_offset)
            axs_sig = plt.subplot2grid((i_rows, i_cols), (i_row, 0), colspan=i_cols,
                                       rowspan=ClassPlotSupport.get_plot_setup_row_sig_span())
            axs_sig.plot(self.__lst_cl_sgs[idx_ch].d_time_plot, self.get_np_d_sig(idx=idx_ch),
                         color=ClassPlotSupport.get_trac_color(idx_trace), linewidth=3.5)
            idx_trace = idx_trace + 1

            # If requested, add the S-G filtered signal
            if b_plot_sg:
                axs_sig.plot(self.__lst_cl_sgs[idx_ch].d_time_plot, self.__lst_cl_sgs[idx_ch].np_d_sig_filt_sg,
                             color=ClassPlotSupport.get_trac_color(idx_trace), linewidth=2.5)
                idx_trace = idx_trace + 1

            # If requested, add the FIR filtered signal
            if b_plot_filt:
                axs_sig.plot(self.__lst_cl_sgs[idx_ch].d_time_plot, self.__lst_cl_sgs[idx_ch].np_d_sig_filt_butter,
                             color=ClassPlotSupport.get_trac_color(idx_trace), linewidth=1.5)
                idx_trace = idx_trace + 1

            # Grid, labels, ticks, and other plot features
            axs_sig.set_xlim(self.__lst_cl_sgs[idx_ch].xlim_tb)
            axs_sig.set_xticks(np.linspace(self.__lst_cl_sgs[idx_ch].xlim_tb[0],
                                           self.__lst_cl_sgs[idx_ch].xlim_tb[1],
                                           self.__lst_cl_sgs[idx_ch].i_x_divisions_tb))
            axs_sig.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs_sig.set_ylim(self.__lst_cl_sgs[idx_ch].ylim_tb)
            axs_sig.set_yticks(np.linspace(self.__lst_cl_sgs[idx_ch].ylim_tb[0],
                                           self.__lst_cl_sgs[idx_ch].ylim_tb[1],
                                           self.__lst_cl_sgs[idx_ch].i_y_divisions_tb))
            ClassPlotSupport.set_plot_setup_sig_axis(axs_sig, self.__lst_cl_sgs[idx_ch].xlim_tb,
                                                     self.__lst_cl_sgs[idx_ch].ylim_tb,
                                                     self.__lst_cl_sgs[idx_ch].str_eu,
                                                     "Channel output, " + self.__lst_cl_sgs[idx_ch].str_eu)

            # After the plots and signal have been plotted (forcing re-calculation of extracted
            # features) create the header, starting with the description
            ClassPlotSupport.set_plot_header_desc(i_rows, i_cols, i_row_offset, self.__str_plot_desc)
            ClassPlotSupport.set_plot_header_machine(i_rows, i_cols, i_row_offset + 1,
                                                     self.str_machine_name(idx=idx_ch))
            lst_points = [self.str_point_name(idx=idx_ch)]
            lst_dates = [self.dt_timestamp(idx_ch)]

            # Set up the point and date lists based on the selected traces
            if b_plot_sg:
                lst_points.append(self.str_filt_sg_desc_short(idx=idx_ch))
                lst_dates.append(self.dt_timestamp(idx_ch))
            if b_plot_filt:
                lst_points.append(self.str_filt_butter_desc_short(idx=idx_ch))
                lst_dates.append(self.dt_timestamp(idx_ch))

            # Add the point and date information to the header
            ClassPlotSupport.set_plot_header_point(i_rows, i_cols, i_row_offset + 2, lst_points)
            ClassPlotSupport.set_plot_header_date(i_rows, i_cols, i_row_offset + 3, lst_dates)

            # Header pane, sparklines
            d_offset = self.__lst_cl_sgs[idx_ch].d_time_plot[-1] - self.__lst_cl_sgs[idx_ch].d_time_plot[0]
            dt_timestamp_end = (self.dt_timestamp(idx=idx_ch) + timedelta(seconds=d_offset))
            ClassPlotSupport.set_plot_sparkline(i_rows, i_cols, i_row_offset,
                                                self.__lst_cl_sgs[idx_ch].np_sparklines,
                                                self.dt_timestamp_mark(idx=idx_ch))

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

        # Get the spectral data
        spec = self.__lst_cl_sgs[0].d_fft_real()
        d_mag = np.abs(spec[1])

        # Open the plot
        plt.rcParams["font.family"] = ClassPlotSupport.get_font_plots()
        plt.figure()

        # Plot the spectrum
        plt.plot(spec[0], d_mag, color=ClassPlotSupport.get_trac_color(0))
        plt.grid()
        plt.xlabel("Frequency, hertz")
        plt.ylabel("Channel amplitude, " + self.__lst_cl_sgs[0].str_eu)
        plt.title(self.__str_plt_support_title_meta(str_plot_type='Spectrum', idx=0))

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

    # Plotting methods including eventtimes need different
    # x-axis limits methods
    def __get_x_limit_events(self, idx_eventtimes=0, idx=0):
        """
        This method returns x-limits modified to actual events

        Parameters
        ----------
        idx_eventtimes: integer
            Index to signal of interest
        idx : integer
            Index of signal to be plotted. Defaults to 0 (first signal)

        Returns
        -------
            list, double : [x-limit start, x-limit end]

        """

        # The x-axis needs to be bounded by either x-limits or eventtimes, but should
        # not display data outside the events
        np_d_eventtimes = self.np_d_eventtimes(idx=idx_eventtimes)

        d_xlim_start = self.__lst_cl_sgs[idx].xlim_tb[0]
        if self.__lst_cl_sgs[idx].xlim_tb[0] < np_d_eventtimes[0]:
            d_xlim_start = np_d_eventtimes[0]

        d_xlim_end = self.__lst_cl_sgs[idx].xlim_tb[1]
        if self.__lst_cl_sgs[idx].xlim_tb[1] > np_d_eventtimes[-1]:
            d_xlim_end = np_d_eventtimes[-1]

        return [d_xlim_start, d_xlim_end]

    # Plotting method for the eventtimes
    def plt_eventtimes(self, idx_eventtimes=0, idx=0):
        """
        Plot a signal and overlay event data in timebase format.

        Parameters
        ---------
        idx_eventtimes : integer
            Index of signal eventtimes. Defaults to 0 (first signal)
        idx : integer
            Index of signal to be plotted. Defaults to 0 (first signal)

        Returns
        -------
        list: [handle to the plot, np array of eventtimes]

        """

        # The x-axis needs to be bounded by either x-limits or eventtimes, but should
        # not display data outside the events
        np_d_eventtimes = self.np_d_eventtimes(idx=idx_eventtimes)
        [d_xlim_start, d_xlim_end] = self.__get_x_limit_events(idx_eventtimes=idx_eventtimes, idx=idx)

        # Put up the the plot time
        plt.rcParams["font.family"] = ClassPlotSupport.get_font_plots()
        plt.figure()
        plt.plot(self.__lst_cl_sgs[idx].d_time, self.__lst_cl_sgs[idx].np_d_sig,
                 color=ClassPlotSupport.get_trac_color(0))
        plt.plot(np_d_eventtimes,
                 self.__lst_cl_sgs[idx].np_d_sig[self.__lst_cl_sgs[idx_eventtimes].idx_events], "ok")
        plt.grid(True)

        plt.xlabel("Time, " + self.__lst_cl_sgs[idx].str_eu_x)
        # plt.xlim(self.__lst_cl_sgs[idx].xlim_tb)
        plt.xlim([d_xlim_start, d_xlim_end])
        plt.xticks(np.linspace(d_xlim_start,
                               d_xlim_end,
                               self.__lst_cl_sgs[idx].i_x_divisions_tb))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.ylabel("Amplitude, " + self.__lst_cl_sgs[idx].str_eu)
        plt.ylim(self.__lst_cl_sgs[idx].ylim_tb)
        plt.yticks(np.linspace(self.__lst_cl_sgs[idx].ylim_tb[0],
                               self.__lst_cl_sgs[idx].ylim_tb[1],
                               self.__lst_cl_sgs[idx].i_y_divisions_tb))

        plt.legend(['as-acquired', 'eventtimes'])
        plt.title(self.__str_plt_support_title_meta(str_plot_type='Triggered Timebase', idx=0))

        # Save the handle prior to showing
        plot_handle = plt.gcf()

        # Show the plot
        plt.show()

        return [plot_handle, self.np_d_eventtimes(idx=idx_eventtimes)]

    # Plotting method for the eventtimes, interpreted as RPM
    def plt_rpm(self, idx_eventtimes=0, idx=0, d_events_per_rev=1.0):
        """
        Plot rpm data in timebase format.

        Parameters
        ---------
        idx_eventtimes : integer
            Index of signal eventtimes. Defaults to 0 (first signal)
        idx : integer
            Index of signal to be plotted. Defaults to 0 (first signal)
        d_events_per_rev : double
            Number of events per revolution. It must be a real value to for hunting tooth gear
            combinations. Defaults to 1

        Returns
        -------
        list: [handle to the plot, np array of RPM values]

        """

        # Put up the plot, timebase is the foundation
        plt.rcParams["font.family"] = ClassPlotSupport.get_font_plots()
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()

        # Local variables  to simplify code; update RPM trend
        np_d_eventtimes = self.np_d_eventtimes(idx=idx_eventtimes)
        self.__lst_cl_sgs[idx].np_d_est_rpm(d_events_per_rev=d_events_per_rev)
        [d_xlim_start, d_xlim_end] = self.__get_x_limit_events(idx_eventtimes=idx_eventtimes, idx=idx)

        lns1 = ax1.plot(self.__lst_cl_sgs[idx].d_time, self.np_d_sig, color=ClassPlotSupport.get_trac_color(0),
                        label='Signal')
        lns2 = ax2.plot(np_d_eventtimes, self.__lst_cl_sgs[idx].np_d_rpm, color=ClassPlotSupport.get_trac_color(1),
                        label='RPM',
                        marker='.',
                        ms=20)
        plt.grid(True)
        ax1.set_xlabel('Time, seconds')
        plt.xlim([d_xlim_start, d_xlim_end])
        plt.xticks(np.linspace(d_xlim_start,
                               d_xlim_end,
                               self.__lst_cl_sgs[idx].i_x_divisions_tb))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.set_ylabel('Amplitude, ' + self.__lst_cl_sgs[0].str_eu)
        ax2.set_ylabel('Event speed, RPM')
        ax2.set_ylim([0, round(1.05 * np.max(self.__lst_cl_sgs[0].np_d_rpm))])

        # Aggregate the plot handles and labels
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs)

        plt.title(self.__str_plt_support_title_meta(str_plot_type='RPM vs. time', idx=0))

        # Save off the handle
        plot_handle = plt.gcf()

        # show the plot
        plt.show()

        return [plot_handle, self.__lst_cl_sgs[0].np_d_rpm]

    # Plotting method, nX plots.
    def plt_nx(self, str_plot_desc=None, b_overlay=True):
        """

        Plot out the synthesized nx and timebase. This call assumes that the first
        signal (idx=0) has the eventtimes that will be applied to all signals in
        the object for nx vector estimation.

        Parameters
        ----------
        str_plot_desc : string
            Signal metadata description for plot title. Defaults to None
        b_overlay : boolean
            Overlay the nX on top of the original signal. Defaults to True

        Return values:
        handle to the plot

        """

        # Parse inputs
        if str_plot_desc is not None:
            # Update class attribute
            self.__str_plot_desc = str_plot_desc

        # In this method, the eventtimes are always fixed to the first signal (idx=0)
        idx_event_source = 0

        # array of index events
        idx_events = self.__lst_cl_sgs[idx_event_source].idx_events
        np_d_eventtimes = self.__lst_cl_sgs[idx_event_source].np_d_eventtimes

        # How many plots, assuming 1 is given?
        i_plots = 0
        lst_nx = []
        for idx_ch, b_obj in enumerate(self.__lst_b_active):

            # The signal could be inactive
            if b_obj:

                # Setup arrays
                np_d_nx_tb = np.zeros_like(self.np_d_sig)
                np_d_time = self.__lst_cl_sgs[idx_ch].d_time_plot

                # Synthesize the nx vector for each revolution
                for idx, idx_active in enumerate(idx_events[0:-1]):
                    # Use the eventtimes from the first signal to calculate the nx vectors
                    # for this signal
                    self.__lst_cl_sgs[idx_ch].calc_nx(np_d_eventtimes=np_d_eventtimes)

                    # Vector phase and angle
                    d_amp = abs(self.__lst_cl_sgs[idx_ch].np_d_nx[idx])
                    d_ang = np.angle(self.__lst_cl_sgs[idx_ch].np_d_nx[idx])
                    d_freq_law = 1.0 / (np_d_eventtimes[idx + 1] - np_d_eventtimes[idx])

                    # Define starting and ending index
                    idx_start = int(idx_active)
                    idx_end = int(idx_events[idx + 1]) + 1

                    # Synthesize the 1X, subtracting phase to correct from balance phase to spectral phase
                    np_d_time_nx = np_d_time[idx_start:idx_end] - np_d_time[idx_start]

                    # Synthesize the 1X, subtracting phase to correct from balance phase to spectral phase
                    np_d_nx_tb[idx_start:idx_end] = d_amp * np.cos(2 * math.pi * d_freq_law * np_d_time_nx - d_ang)

                lst_nx.append(np_d_nx_tb)
                i_plots += 1

        # Figure with subplots
        plt.rcParams["font.family"] = ClassPlotSupport.get_font_plots()
        fig, axs = plt.subplots(i_plots)

        # A single plot returns handle to the axis which isn't iterable. Rather than branch to support
        # this behavior, cast the single axis object to a list with one element
        if not isinstance(axs, (list, tuple, np.ndarray)):
            axs = [axs]

        # Step through the channels and plot out the signals
        for idx_ch, _ in enumerate(self.__lst_cl_sgs):
            axs[idx_ch].plot(self.__lst_cl_sgs[idx_ch].d_time_plot, self.get_np_d_sig(idx=idx_ch),
                             color=ClassPlotSupport.get_trac_color(
                                 0))
            if b_overlay:
                axs[idx_ch].plot(self.__lst_cl_sgs[idx_ch].d_time_plot, lst_nx[idx_ch],
                                 color=ClassPlotSupport.get_trac_color(1))

            axs[idx_ch].plot(self.np_d_eventtimes(idx=idx_event_source),
                             lst_nx[idx_ch][self.__lst_cl_sgs[idx_event_source].idx_events], "ok")

            axs[idx_ch].grid()
            axs[idx_ch].set_xlabel("Time, " + self.__lst_cl_sgs[idx_ch].str_eu_x)
            [d_xlim_start, d_xlim_end] = self.__get_x_limit_events(idx_eventtimes=idx_event_source, idx=idx_ch)
            axs[idx_ch].set_xlim([d_xlim_start, d_xlim_end])
            axs[idx_ch].set_xticks(np.linspace(d_xlim_start,
                                               d_xlim_end,
                                               self.__lst_cl_sgs[idx_ch].i_x_divisions_tb))
            axs[idx_ch].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs[idx_ch].set_ylabel("Channel output, " + self.__lst_cl_sgs[idx_ch].str_eu)
            axs[idx_ch].set_ylim(self.__lst_cl_sgs[idx_ch].ylim_tb)
            axs[idx_ch].set_yticks(np.linspace(self.__lst_cl_sgs[idx_ch].ylim_tb[0],
                                               self.__lst_cl_sgs[idx_ch].ylim_tb[1],
                                               self.__lst_cl_sgs[idx_ch].i_y_divisions_tb))
            axs[idx_ch].set_title(self.__str_plt_support_title_meta(str_plot_type='nX Timebase', idx=idx_ch))
            if b_overlay:
                axs[idx_ch].legend(['as-acquired', 'nX Vector'])

        # Set the layout
        plt.tight_layout()

        # Save off the handle to the plot
        plot_handle = plt.gcf()

        # Show the plot, creating a new figure. This command resets the graphics context
        # so the plot handle has to be saved first.
        plt.show()

        return plot_handle

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

        else:
            # Update the signal class attribute with the metadata description from this object
            self.__lst_cl_sgs[idx].str_plot_apht_desc = \
                self.__str_plt_support_title_meta(str_plot_type='APHT Plot', idx=0)

        return self.__lst_cl_sgs[idx].plt_apht(str_plot_apht_desc)

    # Call polar plotting method.
    def plt_polar(self, str_plot_desc=None, idx=0):
        """
        Plot out amplitude in phase in polar format

        Parameters
        ----------
        str_plot_desc : string
            Additional title text for the plot. If 'None' then method uses class attribute.
        idx : integer
            Index of signal to pull description. Defaults to 0 (first signal)

        Return values:
        handle to the plot

        """

        # Parse inputs
        str_plot_desc_meta = ''
        if str_plot_desc is not None:
            # Update the signal class attribute with the metadata description from this object
            str_plot_desc_meta = \
                self.__str_plt_support_title_meta(str_plot_type='Polar Plot | ' + str_plot_desc, idx=idx)

        else:
            # Update the signal class attribute with the metadata description from this object
            str_plot_desc_meta = \
                self.__str_plt_support_title_meta(str_plot_type='Polar Plot', idx=idx)

        return self.__lst_cl_sgs[idx].plt_polar(str_plot_desc=str_plot_desc_meta)

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

        # Construct the header, self.__i_header_rows must be updated
        # TODO: convert to a linked list, this implementation is clunky
        str_header = 'X'
        str_datetime = 'Date and Time'
        str_fs = 'Sampling Frequency (Hz)'
        str_delta_t = 'Delta Time (seconds)'
        str_units = 'Sequence'
        str_machine_name = 'Machine'
        for idx, class_signal in enumerate(self.__lst_cl_sgs):
            str_header = str_header + "," + class_signal.str_point_name
            str_fs = str_fs + "," + '%0.6f' % class_signal.d_fs
            str_datetime = str_datetime + "," + self.__str_format_dt(idx=idx)
            str_delta_t = str_delta_t + "," + '%0.8f' % (self.__lst_cl_sgs[idx].d_time[1] -
                                                         self.__lst_cl_sgs[idx].d_time[0])

            str_units = str_units + "," + class_signal.str_eu
            str_machine_name = str_machine_name + "," + class_signal.str_machine_name

        str_header = str_header + '\n'
        str_datetime = str_datetime + '\n'
        str_fs = str_fs + '\n'
        str_delta_t = str_delta_t + '\n'
        str_units = str_units + '\n'
        str_machine_name = str_machine_name + '\n'

        # Write the header
        file_data.write(str_header)
        file_data.write(str_datetime)
        file_data.write(str_fs)
        file_data.write(str_delta_t)
        file_data.write(str_units)
        file_data.write(str_machine_name)
        file_data.write('\n')
        file_data.write('\n')
        file_data.write('\n')
        file_data.write('\n')

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
                numpy array, datetime : vector with date and timestamps
                numpy array, double : vector with signal sampling rates
                numpy array, double : vector with delta time interval for each signal
                list, string :  list with engineering units descriptions
                list, string : machine name

        """

        # Pull the filename from the object if nothing is specified
        if str_filename is None:
            str_filename = self.str_file

        # Open the file and read the file headers in
        file_handle = open(str_filename)
        csv_reader = csv.reader(file_handle)
        csv_header = next(csv_reader)
        csv_dt = next(csv_reader)
        csv_fs = next(csv_reader)
        csv_delta_t = next(csv_reader)
        csv_units = next(csv_reader)
        csv_machine_name = next(csv_reader)
        file_handle.close()

        # Parse the header information
        i_signals = len(csv_fs)
        assert len(csv_header) == i_signals, 'Inconsistent number of channels in file'
        assert len(csv_delta_t) == i_signals, 'Inconsistent number of channels in file'
        np_dt_timestamps = np.array(list(map(datetime.fromisoformat, csv_dt[1:i_signals])))
        d_fs = np.array(list(map(float, csv_fs[1:i_signals])))
        d_delta_t = np.array(list(map(float, csv_delta_t[1:i_signals])))

        # Read the file as a dataframe
        df_sig = pd.read_csv(str_filename, header=None,
                             skiprows=self.__i_header_rows, names=csv_header[0:5])

        # Return the data
        return [df_sig, np_dt_timestamps, d_fs, d_delta_t, csv_units, csv_machine_name]
