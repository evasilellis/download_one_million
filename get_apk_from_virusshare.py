import os
import shutil

import pandas as pd
import requests
import logging
import time
import hashlib
import zipfile

BUF_SIZE = 65536


def debugv(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(DEBUG_LEVELV_NUM):
        self._log(DEBUG_LEVELV_NUM, message, args, **kws)


# Adds a very verbose level of logs
DEBUG_LEVELV_NUM = 9
logging.addLevelName(DEBUG_LEVELV_NUM, "DEBUGV")
logging.Logger.debugv = debugv
log = logging.getLogger("featslist")


def logSetup(level):
    if level == 2:
        log.setLevel(DEBUG_LEVELV_NUM)
        log.info("Debug is Very Verbose.")
    elif level == 1:
        log.setLevel(logging.DEBUG)
        log.info("Debug is Verbose.")
    elif level == 0:
        log.setLevel(logging.INFO)
        log.info("Debug is Normal.")
    else:
        log.setLevel(logging.INFO)
        log.warning(
            'Logging level "{}" not defined, setting "normal" instead'.format(level)
        )


# Tries to apply colors to logs
def applyColorsToLogs():
    try:
        import coloredlogs

        style = coloredlogs.DEFAULT_LEVEL_STYLES
        style["debugv"] = {"color": "magenta"}
        coloredlogs.install(
            show_hostname=False,
            show_name=True,
            logger=log,
            level=DEBUG_LEVELV_NUM,
            fmt="%(asctime)s [%(levelname)8s] %(message)s",
        )
    except ImportError:
        log.error("Can't import coloredlogs, logs may not appear correctly.")


applyColorsToLogs()


def get_apk_from_virusshare(api_key, hash_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Api key from file
    if not os.path.exists(api_key):
        log.error(
            'The VirusShare API key file "{}" not found. Exiting'.format(api_key)
        )
        exit(404)

        # Read the CSV file skipping the header and extracting only the first column
        column_to_extract = 'sha256'  # Replace with the actual column name
        data = pd.read_csv(hash_file, usecols=[column_to_extract])

        # Convert the column values to a list
        list_hash = data[column_to_extract].tolist()

    with open(hash_file) as f:
        for line in f:
            this_hash = line.strip()
            list_hash.append(this_hash)

    total_hashes_to_download = len(list_hash)
    print("Num hashes: {}".format(total_hashes_to_download))
    print(
        "- means that this APK has already been downloaded and has a good sha256 (not corrupted)"
    )

    list_hashes_not_found = []
    num_file = 0

    for sha256 in list_hash:
        num_file += 1

        filename = output_dir + "/" + sha256 + ".apk"

        if os.path.exists(filename):
            sha256_algo = hashlib.sha256()
            with open(filename, "rb") as f:
                while True:
                    data = f.read(BUF_SIZE)
                    if not data:
                        break
                    sha256_algo.update(data)

            file_sha256 = sha256_algo.hexdigest()

            if sha256.lower() == file_sha256.lower():
                print("-", end="", flush=True)
                continue
            else:
                print(
                    "  - File corrupted: (sha256: {}) redownloading".format(file_sha256)
                )

        print(
            "\n[{} / {}] sha256 to download: {}".format(
                num_file, total_hashes_to_download, sha256
            )
        )

        url = "https://virusshare.com/apiv2/{}?apikey={}&hash={}".format(
            "download", api_key, sha256
        )
        response = requests.get(url, stream=True)

        response_status = response.status_code
        # print("Response status code: {}".format(response_status))

        if int(response_status) != 200:
            print('Error: Hash "{}" not found'.format(sha256))
            list_hashes_not_found.append(sha256)
            continue

        print("Done")

        zip_filename = filename + ".zip"
        with open(zip_filename, "wb") as fd:
            for chunk in response.iter_content(chunk_size=128):
                fd.write(chunk)

        zipped_file = ""
        with zipfile.ZipFile(zip_filename, "r") as fd:
            zipped_file = fd.namelist()[0]
            fd.extract(zipped_file, pwd=b"infected")

        os.remove(zip_filename)
        # os.rename(zipped_file, os.path.join(output_dir, zipped_file + '.apk'))
        shutil.move(zipped_file, os.path.join(output_dir, zipped_file + ".apk"))

        print("Done unzipping")
        time.sleep(16)

    print()
    n_hash_not_found = len(list_hashes_not_found)
    print("{} hashes were not found in AndroZoo".format(n_hash_not_found))

    hashes_not_found_filename = (
        ".".join(hash_file.split(".")[:-2]) + ".hashes_not_found.txt"
    )

    if n_hash_not_found != 0:
        with open(hashes_not_found_filename, "w") as fd:
            for line in list_hashes_not_found:
                fd.write(line + "\n")
        print('The list was saved as "{}"'.format(hashes_not_found_filename))