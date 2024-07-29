from checker import filter_downloaded_samples, cleanDirectory
from downloader import download_samples
from shazam_functionality import add_tracks_simplified
from sonification import sonification

datasetfile = "CCCS-CIC-AndMal-2020"

#datasetfile = "testing"


if __name__ == "__main__":
    #STEP 1
    #download_samples(datasetfile)

    #STEP 2
    """try:
        filter_downloaded_samples()
    except Exception as e:
        print(f"Unexpected error: {e}")"""


    #STEP 3
    #sonification()

    #STEP 4
    add_tracks_simplified(datasetfile)