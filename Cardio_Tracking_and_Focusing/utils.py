'''
***
'''

import numpy as np
import pickle
import _warnings
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

# basic settings for your data collection
settings = {
    "path": 'path_to_bin_file', # Path for the .bin file
    "save_path": 'save_path', # Save as this name
    "numFrames": 12000, # used for check the load_bin result
    "start_time": 0,
    "numADCSamples": 256, # No. of samples per Chirp
    "numTxAntennas": 3,
    "numRxAntennas": 4,
    "numLoopsPerFrame": 2,
    "startFreq": 77, # GHz
    "sampleRate": 5000, # ksps
    "freqSlope": 64.985e12, # Hz/s 注意单位
    "idleTime": 10, # us
    "rampEndTime": 60, # us
    "numAOA": 64,
    "Frame_period": 0.005, # 200Hz
    "wavelength": 0.0039, # The wavelength of the radar signal  
}
def load_bin(settings):
    '''
    input : path for the .bin file

    output: data + parameters of radar raw data - [no.frame, no.chirps, Tx, Rx, no.ADCsample (complex number)]
    read raw .bin files and output [No. of Frames, IQ (2 channels), Tx (3 channels), Rx (4 channels), ADCsample (256)]
    '''
    path = settings["path"]
    loaded = np.array([])
    loaded = np.concatenate([loaded, np.fromfile(path, dtype = np.int16)])
    # Calculate the number of frame based on the length of data and other parameters
    # 2*2 -> 2 for IQ data component and 2 loops
    numFrames = len(loaded)//(2 * settings['numLoopsPerFrame'] * settings['numADCSamples'] * settings['numTxAntennas'] * settings['numRxAntennas'])
    # print("Number of frames: ", numFrames)
    try:
        assert numFrames == settings["numFrames"]
    except AssertionError:
        print("Number of frames does not match the settings")


    # Shape the data so we can fit all frames
    loaded = loaded.reshape(numFrames, -1)
    # First reshape
    loaded = np.reshape(
        loaded,
        (
            -1, # No. of frames
            settings['numLoopsPerFrame'],
            settings['numTxAntennas'],
            settings['numRxAntennas'],
            settings['numADCSamples'] // 2, # One set of IQ pais
            2, # IQ components
            2, # I part and Q part
        ),
    )
    # Change the order of the last two dimension so that we can combine them in the next step
    loaded = np.transpose(loaded, (0, 1, 2, 3, 4, 6, 5))
    # Reducing the last dimension and combine the IQ complex values
    loaded = np.reshape(
        loaded,
        (
            # -1,
            -1,
            settings['numLoopsPerFrame'],
            settings['numTxAntennas'],
            settings['numRxAntennas'],
            settings['numADCSamples'],
            2, # IQ components
        ),
    )
    loaded = (
        1j * loaded[:, :, :, :, :, 0] + loaded[:, :, :, :, :, 1]
    ).astype(np.complex64) # Imaginary I component and Real Q component
    loaded = loaded[:, :, :, :, :] # Takes only the first clips frames
    return loaded
    
def calculation(dataCube, voxel_location, antenna_location, info, c = 299792458):
    '''
    Given the dataCube from beamforming, calculate the intensity of each voxel.

    return: The indexes of the voxel and its intensity
    '''
    s = np.zeros(info["clip"], dtype=np.complex64)
    for frame_idx in range(info["clip"]): # Traverse all the frames
        for channel in range(1, 12 + 1):
            r = np.linalg.norm(np.array(voxel_location) - np.array(antenna_location[channel])).astype(np.float64) * 2 # Calculating distance
            t = np.linspace(2e-7, 51.2e-6, num=256)
            phase_shift = np.exp(1j * 2 * np.pi * info["freqSlope"] * r * t / c) * np.exp(1j * 2 * np.pi * r / info["waveLength"])
            
            # Antennas 1、3、5、7、9、11 need to inverse the phase
            # if channel in [1, 3, 5, 7, 9, 11]:
            #     phase_shift *= -1
            if channel in [1,3,9,11]:
                phase_shift *= -1
            
            for chirp_idx in range(1):  # 1 chirp
                tx_idx = (channel - 1) // info["numRxAntennas"]
                rx_idx = (channel - 1) % info["numRxAntennas"]

                y_nt = dataCube[frame_idx, chirp_idx, tx_idx, rx_idx, :]

                s[frame_idx] += np.sum(y_nt * phase_shift) # Summing the intensity
    return s

wavelength = 0.0039 # The wavelength of the radar signal
antenna_loc = {  # Define the location of each antenna in the form of (x,y,z)
            1: (-wavelength/4, 0, -wavelength/4),    2: (wavelength/4, 0, -wavelength/4), 3: (3*wavelength/4, 0, -wavelength/4), 4: (5*wavelength/4, 0, -wavelength/4),
            5: (3*wavelength/4, 0, wavelength/4),   6: (5*wavelength/4, 0, wavelength/4), 7: (7*wavelength/4, 0, wavelength/4), 8: (9*wavelength/4, 0, wavelength/4),
            9: (7*wavelength/4, 0, -wavelength/4), 10:(9*wavelength/4, 0, -wavelength/4),11:(11*wavelength/4, 0, -wavelength/4),12:(13*wavelength/4, 0, -wavelength/4),
        }

if __name__ == "__main__":
    settings['path'] = '/Users/zyy/Desktop/PhD/Code/Data_Collection/Mesh-Adaptive-Cardiac-Focusing-/data/final_data/zyy/data_2025-02-24_15_05_51_05/adc_data_24_15_05_50_640_Raw_0.bin'
    load_bin(settings)