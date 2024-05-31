import os

import pandas as pd
import requests
import time
import hashlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BUF_SIZE = 65536
NUM_WORKERS = 30


def get_sha256(filename):
    sha256_algo = hashlib.sha256()
    with open(filename, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256_algo.update(data)

    return sha256_algo.hexdigest()


def get_download(tup):
    url, params, filename = tup
    expected_hash = params["sha256"]
    num_tries = 0
    while True:
        num_tries += 1
        if num_tries > 3:
            print("Three consecutive connexion errors, exiting")
            exit()
        try:
            response = requests.get(url, params=params)
            time.sleep(0.3)
        except requests.exceptions.ConnectionError:
            print("Connection Error, retrying in 1 minute")
            time.sleep(60)
            continue

        if response.status_code != 200:
            return response.status_code

        with open(filename, "wb") as fd:
            for chunk in response.iter_content(chunk_size=128):
                fd.write(chunk)

        new_file_sha256 = get_sha256(filename)

        if expected_hash.lower() == new_file_sha256.lower():
            return response.status_code
        else:
            print("Hash of downloaded file is not correct:")
            print("- Expected sha256: {}".format(expected_hash))
            print("-     File sha256: {}".format(new_file_sha256))
            print("Redownloading")
            num_tries = 0


def load_all2(urllist):
    results = []
    with tqdm(total=len(urllist), unit="apk") as pbar:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(get_download, arg): arg[1]["sha256"] for arg in urllist
            }

            for future in as_completed(futures):
                arg = futures[future]
                res = future.result()
                if res != 200:
                    results.append(arg)
                pbar.update(1)
                pbar.refresh()

    return results


def get_apk_from_androzoo(api_key, hash_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    url = "https://androzoo.uni.lu/api/download"

    print("Number of workers: {}".format(NUM_WORKERS))

    # Read the CSV file skipping the header and extracting only the first column
    column_to_extract = 'sha256'  # Replace with the actual column name
    data = pd.read_csv(hash_file, usecols=[column_to_extract])

    # Convert the column values to a list
    list_hash = data[column_to_extract].tolist()

    total_hashes_to_download = len(list_hash)
    print("Num hashes in file: {}".format(total_hashes_to_download))

    list_hashes_not_found = []

    list_url_params_filename = []

    for sha256 in list_hash:

        filename = output_dir + "/" + sha256 + ".apk"

        if os.path.exists(filename):
            file_sha256 = get_sha256(filename)

            if sha256.lower() == file_sha256.lower():
                print("-", end="", flush=True)
                continue
            else:
                print(
                    "  - File corrupted (file {} gives sha256 {}): redownloading".format(
                        filename, file_sha256
                    )
                )

        # From url string
        params = {"apikey": api_key, "sha256": sha256}

        hash_tuple = (url, params, filename)

        list_url_params_filename.append(hash_tuple)

    print("Num hashes to download: {}".format(len(list_url_params_filename)))

    list_hashes_not_found = load_all2(list_url_params_filename)

    print()
    n_hash_not_found = len(list_hashes_not_found)
    print("{} hashes were not found in AndroZoo".format(n_hash_not_found))

    hashes_not_found_filename = (
        ".".join(hash_file.split(".")[:-1]) + ".hashes_not_found.txt"
    )

    if n_hash_not_found != 0:
        with open(hashes_not_found_filename, "w") as fd:
            for line in list_hashes_not_found:
                fd.write(line + "\n")
        print('The list was saved as "{}"'.format(hashes_not_found_filename))
        return hashes_not_found_filename
    else:
        return ""