import csv

import soundfile as sf
from scipy.signal import lfilter
import cv2
import os
import librosa
from tqdm import tqdm
import time
from statistics import mean
import matplotlib.pyplot as plt

from custom_spectrogram import *
import pandas as pd
import numpy as np

datasetDir = 'Dataset'

wav_folder = datasetDir + '/' + '03 WAV files'
csv_folder = datasetDir + '/' + '04 CSV files'

data_peaks = []

def compare_magnitude(S, specific_value):
    # compare the magnitude of each element in the array S with a specific value
    # and return either the original value or the specific value based on the comparison
    result = np.where(np.abs(S) > specific_value, S, specific_value)
    return result


def find_landmarks(song_data, dilate_size):
    ##GETTING THE SPECTROGRAM
    # First of, all, we instantiate the parameters we're going to use to get the
    # spectrogram. These parameters are defined in the instructions of the
    # project
    fs = 8000  # All the songs we'll be working on will be sampled at an 8KHz rate
    twindow = 64e-3  # The window must be long enough to get 64ms of the signal
    nwindow = fs * twindow  # Number of elements the window must have
    window = np.transpose(np.hamming(nwindow))  # Window used in the spectrogram
    nfft = 512
    NOverlap = nwindow / 2  # We want a 50% overlap
    [S, F, T] = custom_spectrogram(song_data, window, NOverlap, nfft)
    S = S[0: -1][:]
    F = F[0: -1][:]

    ## PROCESSING THE SPECTROGRAM
    # First of all, we're going to take small peaks out
    S = compare_magnitude(S, np.max(S) / 1e6)

    # Now, we're going to pre-process the spectrogram, in order to make it easier to work with.
    # Then, we get the logarithmic value
    S = 10 * np.log10(np.absolute(np.array(S)), where=np.absolute(np.array(S)) > 0)  # [[j for j in i] for i in S]

    # Now, we substract the average of the image
    S = S - np.mean(S)

    # Finally, we filter it
    B = [1, -1]
    A = [1, -0.98]
    S = lfilter(B, A, np.transpose(S), axis=0)
    S = np.transpose(S)

    ## FINDING THE MAXES
    # The next step will be to locate the maxes on the spectrogram. In order to
    # do this, we'll get the dilatation of the spectrogram and then look for the
    # points whose value is the same in both the pre-dilated and the processed
    # spectrogram

    # We begin by creating the Structuring Element.
    se = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_size)
    if S.shape[1] == 0:
        return []

    # Now we dilate the image
    sdilated = cv2.dilate(S, se)

    # And finally, we get the coordinates whose values coincide using the find command.
    [I, J] = np.where(
        sdilated == S)  # Due to the fact that indexes start from 1 in matlab, the results are reduced by 1 in python
    sort = np.lexsort((I, J))
    I = I[sort]
    J = J[sort]

    # It's important to realise that [I, J] are the row and the column where the
    # maxes are, but what we want to know is its value.
    # ALTERNATIVELY, we could get the time and the frequency
    # corresponding to each row and column and store it {e.g. maxes = [F(I), T(J)'];}
    maxes = np.transpose(np.array([I, J]))

    """##PAIRING THE MAXES
    # Now that we have the maxes, it's time to pair them.
    # We want to find all the pairs whose time difference is less than 64
    # columns (~2 seconds) and whose frequency differs is less than 32 (500 Hz).
    # Besides, if for a maximum we find more than 3 pairs, we'll store only the
    # 3 that are closer in time.

    # We'll store all this data in a matrix L, in which we'll use a row for each
    # pair and whose structure will be the following:
    # Lrow = [initialTime, initialFreq, finalFreq-initialFreq, finalTime-initialTime]

    # As we'll store up to three pairs for each point, the maximum length L can
    # have is 3*sizeof(maxes).
    L = np.zeros((len(maxes) * 3, 4))
    pairs = 0  # Number of pairs found

    # For each max that we've found, we'll look for the ones whose frequency isn't
    # more than 500Hz or 2.048 seconds away from it. In case that we found more than
    # three maxes that meet these conditions, we'll get only the nearest three.
    for i in range(len(maxes) - 1):
        f1 = I[i]
        t1 = J[i]

        # Find indices where the conditions are satisfied
        indices = np.where((np.abs(I - f1) < 32) & ((J - t1) > 0) & ((J - t1) < 64))[0]

        # Take the first 3 indices
        indices = indices[:3]

        #We store [initial_time, initial_frequency, freq_diff, time_diff] in the matrix L.
        for j in indices:
            Lrow = np.array([t1, f1, I[j] - f1, J[j] - t1])
            pairs += 1
            L[pairs - 1, :] = Lrow

    # We'll return only those rows of L that contain a pair, and get rid of those
    # that we've created with zeros at the beginning of this part but haven't
    # been used
    L = L[:pairs, :]

    return L, final_maxes"""
    return maxes.tolist()


"""def landmark2hash(L):
    # We reserve memory for H, so we can operate with it without altering its size
    # inside the loop
    H = np.zeros((len(L), 1), dtype=np.uint32)

    for i in range(len(H)):
        # We get the values we're going to use
        initialFreq = L[i, 1] + 1
        Fdiff = L[i, 2]
        initialTime = L[i, 0]
        tDiff = L[i, 3]

        # Creating the hash
        hash_value = (initialFreq - 1) * 2 ** 12 + Fdiff * 2 ** 6 + tDiff

        # We store the row in H with all the needed values
        H[i] = hash_value

    return H """


def add_tracks_simplified(datasetfile):
    start_overall_time = time.time()  # Start timing for overall processing
    processing_times = []  # List to store individual processing times

    apk_samples = pd.read_csv(datasetDir + '/' + datasetfile + '.csv')

    create_folder(csv_folder)
    data = []  # List to collect all the processed data

    dilate_size = [20, 20]
    length_folder = len([name for name in os.listdir(wav_folder) if os.path.isfile(os.path.join(wav_folder, name))])
    s = np.empty(length_folder, dtype=object)

    for idx, name in enumerate(os.listdir(wav_folder)):
        s[idx] = os.path.join(wav_folder, name)

    # Use tqdm to iterate over files with a progress bar
    for name in tqdm(os.listdir(wav_folder), desc='Processing audio files and extracting peaks', unit='file'):
        start_time = time.time()  # Start timing for individual track processing

        filepath = os.path.join(wav_folder, name)
        [c, fs] = sf.read(filepath)
        c = np.array([c])
        c8000 = librosa.resample(y=c, orig_sr=fs, target_sr=8000, res_type="polyphase")
        """L, maxes] = find_landmarks(c8000, dilate_size)

        H = landmark2hash(L)
        create_hashes_table(H, name, datasetfile)"""

        maxes = find_landmarks(c8000, dilate_size)
        processing_time = time.time() - start_time  # End timing for individual track processing
        processing_times.append(processing_time)  # Store the processing time

        hash = name.replace('.dex.data.wav', '')
        if hash in apk_samples['Hash'].values:
            # Retrieve the corresponding 'sample' value
            sample = apk_samples.loc[apk_samples['Hash'] == hash, 'Sample'].values[0] if not pd.isna(
                apk_samples.loc[apk_samples['Hash'] == hash, 'Sample'].values[0]) else ''
            type = apk_samples.loc[apk_samples['Hash'] == hash, 'Type'].values[0] if not pd.isna(
                apk_samples.loc[apk_samples['Hash'] == hash, 'Type'].values[0]) else ''
            family = apk_samples.loc[apk_samples['Hash'] == hash, 'Family'].values[0] if not pd.isna(
                apk_samples.loc[apk_samples['Hash'] == hash, 'Family'].values[0]) else ''
            packed_sample = apk_samples.loc[apk_samples['Hash'] == hash, 'packed_sample'].values[0] if not pd.isna(
                apk_samples.loc[apk_samples['Hash'] == hash, 'packed_sample'].values[0]) else ''

            # Collect the row data
            data.append({
                'Hash': hash,
                'Sample': sample,
                'Type': type,
                'Family': family,
                'packed_sample': packed_sample,
                'Peaks': maxes
            })

    # Convert the collected data to a DataFrame
    df = pd.DataFrame(data)

    # Reset the index and rename it to 'sn'
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'sn'}, inplace=True)

    # Save the DataFrame to a compressed CSV file
    csvPeaks_filename = datasetfile + '_peaks.csv.xz'
    df.to_csv(os.path.join(csv_folder, csvPeaks_filename), sep='\t', encoding='utf-8', index=False, compression='xz')

    """"# Calculate and print statistics
    avg_processing_time = mean(processing_times)
    min_processing_time = min(processing_times)
    max_processing_time = max(processing_times)
    print(f"Average Peak Extraction Processing Time: {avg_processing_time:.2f} seconds")
    print(f"Minimum Peak Extraction Processing Time: {min_processing_time:.2f} seconds")
    print(f"Maximum Peak Extraction Processing Time: {max_processing_time:.2f} seconds")

    overall_duration_seconds = time.time() - start_overall_time  # End timing for overall processing
    overall_duration_hours = overall_duration_seconds // 3600
    overall_duration_minutes = (overall_duration_seconds % 3600) // 60
    print(
        f"Overall Peak Extraction Processing Time: {int(overall_duration_hours)} hours, {int(overall_duration_minutes)} minutes")

    # Generate and save histogram
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(processing_times, color='blue', edgecolor='black')
    ax.set_xlabel('Processing Time (s)')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Peak Extraction Processing Time')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for count, bin_center in zip(counts, bin_centers):
        ax.text(bin_center, count, f"{int(count)}", ha='center', va='bottom')
    plt.savefig('Peak extraction duration distribution.png')
    plt.close() """

def audio_fingerprinting(datasetfile):
    add_tracks_simplified(datasetfile)

def create_folder(folder):
    if not (os.path.isdir(folder)):
        os.mkdir(folder)
    return folder