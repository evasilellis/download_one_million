from checker import filter_downloaded_samples, cleanDirectory
from downloader import download_samples
from shazam_functionality import add_tracks_simplified
from sonification import sonification

datasetfile = "CCCS-CIC-AndMal-2020"

#datasetfile = "testing"


if __name__ == "__main__":
    #STEP 1
    """print("STEP 1: Downloading apps")
    download_samples(datasetfile)"""

    #STEP 2
    print("STEP 2: Filtering downloaded apps")
    try:
        filter_downloaded_samples()
    except Exception as e:
        print(f"Unexpected error: {e}")


    #STEP 3
    print("STEP 3: Sonifying filtered apps")
    sonification()

    #STEP 4
    print("STEP 4: Extracting peaks from sonified apps")
    add_tracks_simplified(datasetfile)