# AppliedVibrationAnalysis

## Introduction

Computation tools for looking at real-world signals for [applied vibration analysis](https://robotsquirrelproductions.com/applied-vibration-analysis/):

![Vibration analysis tools](https://robotsquirrelproductions.com/wp-content/uploads/2024/07/VibAnalysisProcess_2k.png)

As of this writing, the package is a work-in-progress and heavily in development. Structure and syntax will likely change prior to a formal release.

If you do scare up the courage to install this and try commands, I would love your feedback, please log issues and concerns here in the GitHub repository.

## Installation

In this early stage I clone the repository to a local folder and then:

-   Create and activate the dev environment:

    `virtualenv dev_env`

    `source dev_env/bin/activate`

-   Navigate to the local folder with the cloned contents from Github:

    `cd “C:\Local Documents\Github\AppliedVibrationAnalysis”`

-   Use `pip` to install the package (note the ‘`.`’ at the end):

    `pip install -e .`

I verified these instructions in Windows 11, but it should be similar for other operating systems.

## Quick Examples

This section has some examples to get you startedd.

### Timebase plot

A timebase plot shows an amplitude versus time waveform in a rectangular grid for one data series. This plot format can be thought of as a high-resolution trend with 1000s of points over a few hundred milliseconds. 

It takes about 3 steps to get the plot up:

-   Load the libraries:
    `import matplotlib.pyplot as plt`
    `plt.rcParams['figure.dpi'] = 300`
    `import numpy as np`
    `import math`
    `from datetime import datetime, timedelta, timezone`
    `from scipy import signal`
    `import matplotlib` 
    `from appvib import ClSigFeatures`	
	
-   Synthesize the signal:

    `d_fs_even = 2048`
    `i_ns = (d_fs_even*2)`
    `d_freq_sig = 10./(float(i_ns)/d_fs_even)`
    `print('Signal frequency, hertz: ' + '%0.10f' % d_freq_sig)`
    `d_time_ext = np.linspace(0,(i_ns-1),i_ns)/float(d_fs_even)`
    `np_d_test_even = np.sin(2 * math.pi * d_freq_sig * d_time_ext )`
	
'-   Create the plot:

    `plt.rcParams['figure.figsize'] = [8, 4.5]`
    `lst_sig_even = ClSigEven.plt_sigs()`
    `lst_sig_even.savefig('HelloWorld_Sinusoid.pdf')`
	
The code should produce a plot like this:

![timebase plot](https://robotsquirrelproductions.com/wp-content/uploads/2024/11/Timebase-HelloWorld_Sinusoid_2k_241127_170919.png)