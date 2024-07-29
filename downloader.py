#import sys
import os.path
from os import path
from get_apk_from_androzoo import get_apk_from_androzoo
from get_apk_from_virusshare import get_apk_from_virusshare

#api_key_androzoo = "7c4f122a42200d4792a79df70368065ad783676746c120a92031c53123003088" # EV's key
api_key_androzoo = "cdc31a7425300cf164c9067cc9e0b6c81ba17864ee0e5a37be3f6035f95884a5" # 3A's key
api_key_virusshare = "9DZhK1uSMmgCdxwNHUIUc2Ju9E2vSW1l" # EV's key

outdir = 'Dataset'

def download_samples(datasetfile):
    print("Dataset: " + datasetfile)
    print("Files are prefixed: " + datasetfile)

    print(
        "=========================================================================="
    )
    sha256file = outdir + "/" + datasetfile + ".csv"

    datasetdownloadfolder = outdir + "/" + "01 APK files"

    if not path.exists(sha256file):
        print("ERROR: missing file " + sha256file)
        exit()

    if not path.exists(datasetdownloadfolder):
        print("Creating directory " + datasetdownloadfolder)
        os.makedirs(datasetdownloadfolder)

    print("Downloading " + sha256file)
    hashes_not_found_androzoo = get_apk_from_androzoo(
        api_key_androzoo, sha256file, datasetdownloadfolder
    )

    print(datasetdownloadfolder)

    downloaded_files = [
        name
        for name in os.listdir(datasetdownloadfolder)
        if os.path.isfile(datasetdownloadfolder + "/" + name)
    ]

    print("Counting: " + str(len(downloaded_files)))

    if path.exists(hashes_not_found_androzoo):
        print("Trying to download from VirusShare")

        get_apk_from_virusshare(
            api_key_virusshare, hashes_not_found_androzoo, datasetdownloadfolder
        )