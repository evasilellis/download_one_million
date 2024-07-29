import io
import wave
import zipfile

from tqdm import tqdm
import hashlib
import zlib
import os

datasetDir = 'Dataset'

apk_folder = datasetDir + '/' + '01 APK files'
log_folder = 'CustomLogs'
dex_data_section_folder = datasetDir + '/' + '02 DEX data_section files'
wav_folder = datasetDir + '/' + '03 WAV files'

audio_files_with_no_length = []

# Get the path where the file exists
def get_path_file(filename):
    return dex_data_section_folder + '/' + filename


# Read the file in bytes
def read_file_in_bytes(filename):
    path = get_path_file(filename)
    with open(path, "rb") as f:
        file_bytes = f.read()
    converted_file_in_bytes = io.BytesIO(file_bytes)
    return converted_file_in_bytes


def check_size(filename_selected):
    byte_size = os.stat(dex_data_section_folder + '/' + filename_selected).st_size
    if byte_size >= 5646:
        return 44100
    elif byte_size >= 1024:
        return 8000
    else:
        return 0


# Convert the file to audio (wav type)
def create_wav_file(converted_file_in_bytes, filename_selected):
    frame_rate = check_size(filename_selected)
    if frame_rate != 0:
        new_wav_path = os.path.join(create_wav_folder(),
                                    create_wav_filename(filename_selected))
        obj = wave.open(new_wav_path, 'wb')
        obj.setnchannels(1)
        obj.setsampwidth(2)
        obj.setframerate(frame_rate)
        obj.writeframes(b''.join(converted_file_in_bytes))
    else:
        audio_files_with_no_length.append(filename_selected)


# Create the wav file
def create_wav_filename(filename_selected):
    return filename_selected + '.wav'


# Create wav folder, if it doesn't exist
def create_wav_folder():
    if not (os.path.isdir(wav_folder)):
        os.mkdir(wav_folder)
    return wav_folder


def convert_apk_file():
    # Get a list of APK files in the specified folder
    apk_files = [entry for entry in os.scandir(dex_data_section_folder) if
                 entry.is_file() and entry.name.endswith('.dex.data')]

    # Initialize a progress bar with the total number of APK files
    progress_bar = tqdm(total=len(apk_files), desc="Converting Android applications into audio files")

    for apk_file in apk_files:
        # Simulate processing an APK file
        # Replace this with your actual processing logic
        converted_file_in_bytes = read_file_in_bytes(apk_file.name)
        create_wav_file(converted_file_in_bytes, apk_file.name)

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    print("num_audio_files_with_no_length: ", len(audio_files_with_no_length))

    with open(r'CustomLogs/audio_files_with_no_length.txt', 'w') as fp:
        fp.write('\n'.join(audio_files_with_no_length))

def extract_data_section(apk_dir, dex_dir):

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
    multi_dex_files = []

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
                        # Extract the single dex file
                        dex_filename = os.path.splitext(apk_file)[0] + ".dex"
                        dex_path = os.path.join(dex_dir, dex_filename)
                        with apk.open("classes.dex") as dex_in, open(dex_path, "wb") as dex_out:
                            dex_data = dex_in.read()
                            # Verify the dex file format
                            extracted_alder32_checksum = hex(int.from_bytes(dex_data[0x08:0x0C], byteorder='little'))
                            extracted_sha1_signature = dex_data[0x0C:0x20].hex()
                            checksumPortion_data = dex_data[0x0C:]
                            calculated_alder32_checksum = hex(zlib.adler32(checksumPortion_data))

                            if (extracted_alder32_checksum == calculated_alder32_checksum):  # and (extracted_sha1_signature == calculated_sha1_signature):
                                dex_out.write(dex_data)
                                dataSection_file_name = os.path.splitext(apk_file)[0] + ".dex.data"
                                dataSection_file_path = os.path.join(dex_data_section_folder, dataSection_file_name)

                                # Extract the data section
                                data_size = int.from_bytes(dex_data[0x68:0x6C], byteorder='little')
                                data_offset = int.from_bytes(dex_data[0x6C:0x70], byteorder='little')
                                if data_offset > 0 and data_offset + data_size <= len(dex_data):
                                    with open(dataSection_file_path, 'wb') as out_file:
                                      out_file.write(dex_data[data_offset:data_offset + data_size])
                                else:
                                    corrupted_dataSection_files.append(dataSection_file_name)
                                    num_dataSection_corrupted += 1
                            else:
                                corrupted_dex_files.append(dex_filename)
                                num_dex_corrupted += 1
                        os.remove(dex_path)
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

    with open(r'CustomLogs/multi_dex_files.txt', 'w') as fp:
        fp.write('\n'.join(multi_dex_files))


def sonification():
    create_folder(dex_data_section_folder)
    create_folder(log_folder)

    extract_data_section(apk_folder, dex_data_section_folder)

    convert_apk_file()

def create_folder(folder):
    if not (os.path.isdir(folder)):
        os.mkdir(folder)
    return folder