"""This script tries to access neurosky to record eeg data. LSL version.
Based on: https://github.com/D1o0g9s/EEGFaceDetection/blob/master/mindwave_code/LSLCollectRawData_without_handler.py
Notes:
- requires mindwave.py
- change the line labeled "mac version" to run it on other OS
- run by typing 'python lsl_neurosky.py' in the terminal or console
- configure your experimental settings under SETTINGS
- trial_permutations are randomly sampled from TARGETS by default
- the recorded data are saved under ./[SUBJECT_NUMBER]/[SESSION_NUMBER]/
"""

## IMPORTS
import mindwave, time
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
import time
import pandas as pd
import numpy as np
from random import choices
from threading import Thread
import os
from os.path import join as pjoin

##########################################################################
##########################################################################

## SETTINGS
SUBJECT_NUMBER = 0
SESSION_NUMBER = 0
TRIAL_DURATION = 2000 # ms
NUM_TRIALS = 2 # number of trials for each target
SAMPLING_FREQUENCY = 128 # Hz
INTER_TRIAL_INTERVAL = 1000 # ms, between targets

TARGETS = {0:'rest'}  # classication targets, eg. {1:'task1', 2:'task2'}
                      # NOTE: 0 is reserverd for 'rest'. Unless you just
                      # want to record 'rest' for the entire session, you
                      # should not include the key 0 in this dictionary

### PATHS
BASE_PATH = "./"
DATA_PATH = pjoin(pjoin(BASE_PATH, str(SUBJECT_NUMBER)),str(SESSION_NUMBER))
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

##########################################################################
##########################################################################

class MindwaveLSLRecorder: 
    def __init__(self): 
        # Create eeg outlet [raw eeg, attention, meditation, blink]
        self.__Fs = 128 # 128Hz 
        info_eeg = StreamInfo(name='Mindwave EEG', 
            type='EEG', channel_count=4, nominal_srate=self.__Fs, 
            channel_format='float32', source_id='eeg_thread')
        self.__eeg_outlet = StreamOutlet(info_eeg)
        self.currentTimestamp = None

    def run(self):
        print("Connecting")
        headset = mindwave.Headset('/dev/tty.MindWaveMobile-SerialPo') # mac version
        print("Connected!")

        try:
            while (headset.poor_signal > 5):
                print("Headset signal noisy %d. Adjust the headset and the earclip." % (headset.poor_signal))
                time.sleep(0.1)
                
            print("Writing output to LSL stream" )
            stime = time.time()
            prevTime = 0
            while True:
                cycle_start_time = time.time()
                if headset.poor_signal > 5 :
                    print("Headset signal noisy %d. Adjust the headset and the earclip." % (headset.poor_signal))
                    self.__eeg_outlet.push_sample(np.array([0, 0, 0, 0]))
                else :
                    self.__eeg_outlet.push_sample(np.array([headset.raw_value, headset.attention, headset.meditation, headset.blink]))

                # Print "second elapsed" every second
                timeDiff = int(time.time()-stime)
                if(timeDiff != prevTime) : 
                    print("seconds elapsed: " + str(timeDiff))
                    prevTime = timeDiff
                
                time.sleep((1/(self.__Fs)) - (time.time() - cycle_start_time + 0.0005))
        finally:
            print("Closing!")
            headset.stop()

def get_eeg_lsl(save_variable, types = ['EEG']):
    """ 'get eeg from lsl' (by eeg I mean all specified 'types' of data) and
        save them onto 'save_variable'
    Continuously collect specified channels data from the Lab Streaming Layer(LSL),
    not necessarily just the EEG(a little bit of a misnomer).
    (LSL is commonly used in EEG labs for different devices to push and pull
    data from a shared network layer to ensure good synchronization and timing
    across all devices)

    Parameters
    ----------
    save_variable : empty list --> [timepoints by channels]
                                    where channels = 
                                    [timestamp, types[0]*num_channels_of_type[0] ...]
        the variable to save the data onto
    types: len(types) list of str
        specifies the source types of the streams you want to get data from, eg. ['EEG', 'AUX']
    

    Returns
    -------
    inlets : some length list of pylsl.StreamInlet objects
    pull_thread : the thread instance that pulls data from the LSL constantly

    Note
    ----
    To properly end the pull_thread, call all inlet.close_stream() right before
    you call board.stop_stream() If this isn't done, the program could freeze or 
    show error messages. Do not lose the inlets list

    Examples
    --------
    >>> save_variable = []
    >>> inlets, _ = get_eeg_lsl(save_variable) # to start pulling data from lsl
    ...
    >>> for inlet in inlets:\
    >>>     inlet.close_stream()
    >>> print(save_variable)
    """
    streams = []
    inlets = []
    for stream_type in types:
        streams.extend(resolve_stream('type', stream_type))
    for stream in streams:
        inlets.append(StreamInlet(stream))
    if inlets == None or len(inlets) == 0:
                raise Exception("Error: no stream found.")

    def save_sample(inlets, save_variable):
        while True:
            row_data = [0]
            inlet_idx = 0
            while inlet_idx < len(inlets): # iterate through the inlets
                sample, timestamp = inlets[inlet_idx].pull_sample()
                if (sample, timestamp) != (None, None):
                    if inlet_idx is (len(inlets) - 1):
                        row_data[0] = timestamp
                    row_data.extend(sample)
                    inlet_idx += 1 # move on to next inlet
                else:
                    time.sleep(0.0001) # wait for 0.1 ms if the data is not there yet
                                       # just to save some processing power
            save_variable.append(row_data)

    pull_thread = Thread(target = save_sample, args=(inlets, save_variable))
    pull_thread.daemon = True
    pull_thread.start()
    return inlets, pull_thread

##########################################################################
##########################################################################
# if this script is run as a script rather than imported
if __name__ == "__main__": 
    mlslr = MindwaveLSLRecorder()
    mlslr.run()
    eeg = []
    inlets, _ = get_eeg_lsl(eeg)
    # for inlet in inlets:
    #     inlet.close_stream()