from downloader import download_samples

#datasetfile = "one_million_sha256_part2"
from feature_extraction import sonification

#datasetfile = "testing"


if __name__ == "__main__":
    #download_samples(datasetfile)

    try:
        sonification()
    except Exception as e:
        print(f"Unexpected error: {e}")