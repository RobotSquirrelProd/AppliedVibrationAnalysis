from ds1054z import DS1054Z
import time
import numpy as np

def b_set_trigger(scope_con, d_trigger_level = 1e-01):
    """Set the trigger configuration
    
    Keyword arguments:
    scope_con -- Connection to scope. Usually
        the value returned from the 'DS1054Z('192.168.1.206')' call.
    d_trigger_level -- Voltage level to trigger scope (default: 0.1 volts)
    
    Return values:
    [None]
    """
    
    scope_con.write(':trigger:edge:source CHAN1')
    scope_con.write(':trigger:edge:level ' + format(d_trigger_level))
    scope_con.single()

def b_setup_scope(scope_con, lst_ch_active = [True, False, False, False],
                  lst_ac_coupled = [True, True, True, True],
                  lst_ch_scale=[5.e-1, 1., 1., 1.], timebase_scale=5e-2, 
                  d_trigger_level = 1e-01, b_single = True):
    """Setup Rigol ds1054z to read data from one or more channels
    
    Keyword arguments:
    scope_con -- Connection to scope. Usually
        the value returned from the 'DS1054Z('192.168.1.206')' call.
    lst_ch_active -- List of booleans describing active channels
        (default: [True, False, False, False] which sets only channel 1
        active).
    lst_ac_coupled -- Boolean list describing ac-coupled state for 
        each channel (default: [True, True, True, True], all channels
        set to AC-coupled)
    lst_ch_scale  -- List with channel scale (default: [5.e-1, 1., 1., 1.] volts)
    timebase_scale -- Time scale for data (default: 0.005 seconds)
    d_trigger_level -- Voltage level to trigger scope (default: 0.1 volts)
    b_trigger -- If true, then use trigger levels and capture only
        a single shot (default: True)
    
    Return values:
    d_ch1_scale_actual -- The closest value chosen by the scope
    """

    # Setup horizontal scale and place scope in run mode
    scope_con.timebase_scale = timebase_scale
    scope_con.run()

    # Setup channel 1
    scope_con.display_channel(1,enable=lst_ch_active[0])
    scope_con.set_probe_ratio(1,1)
    scope_con.set_channel_scale(1,"{:e}".format(lst_ch_scale[0]) +'V')
    if lst_ac_coupled[0]:
        scope_con.write(':CHANnel1:COUPling AC')
    else:
        scope_con.write(':CHANnel1:COUPling DC')

    # Setup channel 2
    scope_con.display_channel(2,enable=lst_ch_active[1])
    if lst_ch_active[1]:
        scope_con.set_probe_ratio(2,1)
        scope_con.set_channel_scale(2,"{:e}".format(lst_ch_scale[1]) +'V')
        if lst_ac_coupled[1]:
            scope_con.write(':CHANnel2:COUPling AC')
        else:
            scope_con.write(':CHANnel2:COUPling DC')

    # Setup channel 3
    scope_con.display_channel(3,enable=lst_ch_active[2])
    if lst_ch_active[2]:
        scope_con.set_probe_ratio(3,1)
        scope_con.set_channel_scale(3,"{:e}".format(lst_ch_scale[2]) +'V')
        if lst_ac_coupled[2]:
            scope_con.write(':CHANnel3:COUPling AC')
        else:
            scope_con.write(':CHANnel3:COUPling DC')


    # Setup channel 4
    scope_con.display_channel(4,enable=lst_ch_active[3])
    if lst_ch_active[3]:
        scope_con.set_probe_ratio(4,1)
        scope_con.set_channel_scale(4,"{:e}".format(lst_ch_scale[3]) +'V')
        if lst_ac_coupled[3]:
            scope_con.write(':CHANnel4:COUPling AC')
        else:
            scope_con.write(':CHANnel4:COUPling DC')
    
    # Do we need a trigger?
    if b_single:
        
        # Set the scope to capture after trigger
        b_set_trigger(scope_con, d_trigger_level)
        
    else:
        
        # No trigger, useful for seeing the scope data when you aren't sure
        # what the signal looks like
        scope_con.write(":TRIGger:SWEep AUTO")
        
    return scope_con.get_channel_scale(1)

# This one is a little tricky because it can take time to acquire the signal so there 
# are pause statements to allow data to accumulate at the scope. If the acquisition 
# terminates before the sampling is complete there will be NaN's in the list. In this 
# case the NaN's are converte zeros to allow processing to continue. It can be helpful 
# to see a partial waveform to troubleshoot timing at the scope.
def d_get_data(scope_con, lst_ch_active = [True, False, False, False], timebase_scale=5e-2):
    """Get data from the scope
    
    Keyword arguments:
    scope_con -- Scope connection handle (Required). Usually
        the value returned from the 'DS1054Z('192.168.1.206')' call.
    lst_ch_active -- List of booleans describing active channels
        (default: [True, False, False, False] which sets only channel 1
        active). This should exactly match the values used to setup the
        scope.
    timebase_scale -- Scope time scale (default: 5e-2)
    
    Return values:
    lst_d_ch -- list of numpy array of values from the scope:
        [np_d_ch1, np_d_ch2, np_d_ch3, np_d_ch4]. If channel
        was not selected for data capture it will be empty ([])
    """

    # Initialize local variables
    d_ch1 = []
    d_ch2 = []
    d_ch3 = []
    d_ch4 = []
    
    # Calculate the delay time
    d_time_delay = timebase_scale*32 + 1.
    
    # Acquire the data
    time.sleep(d_time_delay)
    if lst_ch_active[0]:
        d_ch1 = scope_con.get_waveform_samples(1, mode='NORM')
    if lst_ch_active[1]:
        d_ch2 = scope_con.get_waveform_samples(2, mode='NORM')
    if lst_ch_active[2]:
        d_ch3 = scope_con.get_waveform_samples(3, mode='NORM')
    if lst_ch_active[3]:
        d_ch4 = scope_con.get_waveform_samples(4, mode='NORM')
        
    time.sleep(d_time_delay)

    # Back to run mode 
    scope_con.run()
    
    # Convert the list to a numpy array and replace NaN's with zeros
    np_d_ch1 = np.array(d_ch1) 
    np_d_ch1 = np.nan_to_num(np_d_ch1)
    np_d_ch2 = np.array(d_ch2) 
    np_d_ch2 = np.nan_to_num(np_d_ch2)
    np_d_ch3 = np.array(d_ch3) 
    np_d_ch3 = np.nan_to_num(np_d_ch3)
    np_d_ch4 = np.array(d_ch4) 
    np_d_ch4 = np.nan_to_num(np_d_ch4)
    
    return [np_d_ch1, np_d_ch2, np_d_ch3, np_d_ch4]
