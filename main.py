from downloader import download_samples

#datasetfile = "one_million_sha256_part1"
datasetfile = "testing"


def main():
    download_samples(datasetfile)

if __name__ == "__main__":
    main()