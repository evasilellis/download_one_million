import hashlib
import io
import os
import wave
import zipfile
import zlib
from tqdm import tqdm

datasetDir = 'Dataset'

apk_folder = datasetDir + '/' + '01 APK files'
log_folder = 'CustomLogs'

corrupted_apk_log = os.path.join(log_folder, 'corrupted_apk_files.txt')
corrupted_dex_log = os.path.join(log_folder, 'corrupted_dex_files.txt')
corrupted_dataSection_log = os.path.join(log_folder, 'corrupted_dataSection_files.txt')
single_dex_log = os.path.join(log_folder, 'single_dex_files.txt')
multi_dex_log = os.path.join(log_folder, 'multi_dex_files.txt')

def filter_downloaded_samples():
    create_folder(log_folder)
    integrityCheck_singleDexFilter(apk_folder)
    cleanDirectory(apk_folder, single_dex_log)


def create_folder(folder):
    if not (os.path.isdir(folder)):
        os.mkdir(folder)
    return folder


def integrityCheck_singleDexFilter(apk_dir):
    apk_files = os.listdir(apk_dir)
    num_apks_processed = 0
    num_multi_dex = 0
    num_single_dex = 0
    num_apks_corrupted = 0
    num_dex_corrupted = 0
    num_dataSection_corrupted = 0

    progress_bar = tqdm(total=len(apk_files), desc="Filtering APK files", unit="file")

    for apk_file in apk_files:
        apk_path = os.path.join(apk_dir, apk_file)
        """if not zipfile.is_zipfile(apk_path) or not apk_path.endswith(".apk"):
            progress_bar.update(1)
            with open(corrupted_apk_log, 'a') as fp:
                fp.write(apk_file + '\n')
            num_apks_corrupted += 1
            continue"""

        with open(apk_path, "rb") as apk_data_file:
            apk_data = apk_data_file.read()

        calculated_sha256_signature = hashlib.sha256(apk_data).hexdigest()

        if calculated_sha256_signature.casefold() == os.path.splitext(apk_file)[0].casefold():
            num_apks_processed += 1
            try:
                with zipfile.ZipFile(apk_path) as apk:
                    if "classes2.dex" in apk.namelist():
                        num_multi_dex += 1
                        with open(multi_dex_log, 'a') as fp:
                            fp.write(apk_file + '\n')
                    else:
                        num_single_dex += 1
                        with open(single_dex_log, 'a') as fp:
                            fp.write(apk_file + '\n')
                        try:
                            with apk.open("classes.dex") as dex_in:
                                dex_data = dex_in.read()

                            extracted_alder32_checksum = hex(int.from_bytes(dex_data[0x08:0x0C], byteorder='little'))
                            extracted_sha1_signature = dex_data[0x0C:0x20].hex()
                            checksumPortion_data = dex_data[0x0C:]
                            calculated_alder32_checksum = hex(zlib.adler32(checksumPortion_data))

                            if (extracted_alder32_checksum == calculated_alder32_checksum):
                                data_size = int.from_bytes(dex_data[0x68:0x6C], byteorder='little')
                                data_offset = int.from_bytes(dex_data[0x6C:0x70], byteorder='little')
                                if data_offset > 0 and data_offset + data_size <= len(dex_data):
                                    pass  # Placeholder for actual processing logic
                                else:
                                    with open(corrupted_dataSection_log, 'a') as fp:
                                        fp.write(apk_file + '\n')
                                    num_dataSection_corrupted += 1
                            else:
                                with open(corrupted_dex_log, 'a') as fp:
                                    fp.write(apk_file + '\n')
                                num_dex_corrupted += 1
                        except (KeyError, zipfile.BadZipFile, zlib.error, zipfile.LargeZipFile, zipfile.BadZipFile) as e:
                            print(f"Error processing dex file in {apk_file}: {e}")
                            with open(corrupted_apk_log, 'a') as fp:
                                fp.write(apk_file + '\n')
                            num_apks_corrupted += 1
            except (zipfile.BadZipFile, zlib.error, zipfile.LargeZipFile) as e:
                print(f"Error processing APK file {apk_file}: {e}")
                with open(corrupted_apk_log, 'a') as fp:
                    fp.write(apk_file + '\n')
                num_apks_corrupted += 1
        else:
            with open(corrupted_apk_log, 'a') as fp:
                fp.write(apk_file + '\n')
            num_apks_corrupted += 1

        progress_bar.update(1)

    progress_bar.close()

    print("num_apks_processed: ", num_apks_processed)
    print("num_multi_dex: ", num_multi_dex)
    print("num_single_dex: ", num_single_dex)
    print("num_apks_corrupted: ", num_apks_corrupted)
    print("num_dex_corrupted: ", num_dex_corrupted)
    print("num_dataSection_corrupted :", num_dataSection_corrupted)

def cleanDirectory(apk_directory, single_dex_log):
    single_dex_files = load_filenames(single_dex_log)
    keep_filenames = single_dex_files

    all_files = [f for f in os.listdir(apk_directory)] #if f.endswith('.apk')]
    total_files = len(all_files)

    with tqdm(total=total_files, desc="Deleting multi-dex and corrupted APK files", unit="file") as pbar:
        for filename in all_files:
            sha256_filename = filename.replace('.apk', '')
            if sha256_filename not in keep_filenames:
                file_path = os.path.join(apk_directory, filename)
                os.remove(file_path)
                pbar.write(f"Deleted: {file_path}")
            pbar.update(1)

def load_filenames(file_path):
    with open(file_path, 'r') as file:
        return set(line.strip().replace('.apk', '') for line in file)