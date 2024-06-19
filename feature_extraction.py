import hashlib
import io
import os
import wave
import zipfile
import zlib
from scipy.io import wavfile

from tqdm import tqdm

datasetDir = 'Dataset'

apk_folder = datasetDir + '/' + '01 APK files'
log_folder = 'CustomLogs'


def sonification():
    #create_folder(dex_data_section_folder)
    create_folder(log_folder)

    extract_data_section(apk_folder)

def create_folder(folder):
    if not (os.path.isdir(folder)):
        os.mkdir(folder)
    return folder

def extract_data_section(apk_dir):

    apk_files = os.listdir(apk_dir)
    num_apks_processed = 0
    num_multi_dex = 0
    num_single_dex = 0
    num_apks_corrupted = 0
    num_dex_corrupted = 0
    num_dataSection_corrupted = 0
    corrupted_apk_files = []
    corrupted_dex_files = []
    corrupted_dataSection_files = []
    single_dex_files = []
    multi_dex_files = []
    audio_files_with_no_length = []

    for apk_file in tqdm(apk_files, desc="Extracting data section from DEX files of samples"):
        apk_path = os.path.join(apk_dir, apk_file)
        # num_apks_processed += 1
        if not zipfile.is_zipfile(apk_path) or not apk_path.endswith(".apk"):
            continue
        apk_data = open(apk_path, "rb").read()
        calculated_sha256_signature = hashlib.sha256(apk_data).hexdigest()
        if calculated_sha256_signature.casefold() in os.path.splitext(apk_file)[0].casefold():
            # Increment the number of APKs processed
            num_apks_processed += 1
            with zipfile.ZipFile(apk_path) as apk:
                # Check if the APK file is multi-dex
                    if "classes2.dex" in apk.namelist():
                        num_multi_dex += 1
                        multi_dex_files.append(apk_file)
                    else:
                        num_single_dex += 1
                        single_dex_files.append(apk_file)

                        # Extract the single dex file
                        #dex_filename = os.path.splitext(apk_file)[0] + ".dex"
                        #dex_path = os.path.join(dex_dir, dex_filename)
                        with apk.open("classes.dex") as dex_in:#, open(dex_path, "wb") as dex_out:
                            dex_data = dex_in.read()
                            # Verify the dex file format
                            extracted_alder32_checksum = hex(int.from_bytes(dex_data[0x08:0x0C], byteorder='little'))
                            extracted_sha1_signature = dex_data[0x0C:0x20].hex()
                            checksumPortion_data = dex_data[0x0C:]
                            calculated_alder32_checksum = hex(zlib.adler32(checksumPortion_data))

                            if (extracted_alder32_checksum == calculated_alder32_checksum):  # and (extracted_sha1_signature == calculated_sha1_signature):
                                #dex_out.write(dex_data)
                                #dataSection_file_name = os.path.splitext(apk_file)[0] + ".dex.data"
                                #dataSection_file_path = os.path.join(dex_data_section_folder, dataSection_file_name)

                                # Extract the data section
                                data_size = int.from_bytes(dex_data[0x68:0x6C], byteorder='little')
                                data_offset = int.from_bytes(dex_data[0x6C:0x70], byteorder='little')
                                if data_offset > 0 and data_offset + data_size <= len(dex_data):
                                    #with open(dataSection_file_path, 'wb') as out_file:
                                    #  out_file.write(dex_data[data_offset:data_offset + data_size])
                                    """data_section = dex_data[data_offset:data_offset + data_size]
                                    wav_data = create_wav_in_memory(data_section)

                                    if wav_data:
                                        # Now you can apply audio processing techniques to `audio_data`
                                        print(f"File name: {apk_file}")

                                        # Read the WAV data into a numpy array
                                        wav_data.seek(0)  # Ensure we are at the beginning of the BytesIO object
                                        frame_rate, audio_data = wavfile.read(wav_data)
                                        print(f"Frame rate: {frame_rate}")
                                        print(f"Audio data shape: {audio_data.shape}")

                                        ## After all operations are done, you can close the BytesIO object if desired
                                        ##wav_data.close()

                                        # Open the WAV file in read mode
                                        #with wave.open(wav_data, 'rb') as wav_file:
                                        #    # Print the header information
                                        #    print(f"Number of channels: {wav_file.getnchannels()}")
                                        #    print(f"Sample width: {wav_file.getsampwidth()} bytes")
                                        #    print(f"Frame rate (sample rate): {wav_file.getframerate()} Hz")
                                        #    print(f"Number of frames: {wav_file.getnframes()}")
                                        #    print(f"Compression type: {wav_file.getcomptype()}")
                                        #    print(f"Compression name: {wav_file.getcompname()}")

                                    else:
                                        print("No audio data was created due to an error.")
                                        audio_files_with_no_length.append(apk_file)

                                    stra = 1 """
                                else:
                                    corrupted_dataSection_files.append(apk_file)
                                    num_dataSection_corrupted += 1
                            else:
                                corrupted_dex_files.append(apk_file)
                                num_dex_corrupted += 1
                        #os.remove(dex_path)
        else:
            # Increment the number of corrupted files and add the filename to the list
            corrupted_apk_files.append(apk_file)
            num_apks_corrupted += 1

    print("num_apks_processed: ", num_apks_processed)
    print("num_multi_dex: ", num_multi_dex)
    print("num_single_dex: ", num_single_dex)
    print("num_apks_corrupted: ", num_apks_corrupted)
    print("num_dex_corrupted: ", num_dex_corrupted)
    print("num_dataSection_corrupted :", num_dataSection_corrupted)

    with open(r'CustomLogs/corrupted_apk_files.txt', 'w') as fp:
        fp.write('\n'.join(corrupted_apk_files))

    with open(r'CustomLogs/corrupted_dex_files.txt', 'w') as fp:
        fp.write('\n'.join(corrupted_dex_files))

    with open(r'CustomLogs/corrupted_dataSection_files.txt', 'w') as fp:
        fp.write('\n'.join(corrupted_dataSection_files))

    with open(r'CustomLogs/single_dex_files.txt', 'w') as fp:
        fp.write('\n'.join(single_dex_files))

    with open(r'CustomLogs/multi_dex_files.txt', 'w') as fp:
        fp.write('\n'.join(multi_dex_files))

    with open(r'CustomLogs/audio_files_with_no_length.txt', 'w') as fp:
        fp.write('\n'.join(audio_files_with_no_length))

# Convert the file to audio (wav type)
def create_wav_in_memory(data_section):
    frame_rate = check_size(data_section)
    if frame_rate != 0:
        try:
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as obj:
                obj.setnchannels(1)
                obj.setsampwidth(2)
                obj.setframerate(frame_rate)
                obj.writeframes(data_section)  # Write the data section directly
            wav_io.seek(0)

            return wav_io  # Return the processed audio data
        except Exception as e:
            print(f"Error processing WAV data: {e}")
            return None
    else:
        return None

def check_size(data_section):
    byte_size = len(data_section)
    if byte_size >= 5646:
        return 44100
    elif byte_size >= 1024:
        return 8000
    else:
        return 0