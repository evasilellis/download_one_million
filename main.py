import pandas as pd

from checker import filter_downloaded_samples, cleanDirectory
from downloader import download_samples
from shazam_functionality import add_tracks_simplified
from sonification import sonification

datasetfile = "one_million_sha256_part1_subpart1_single_dex_files"

if __name__ == "__main__":
    #STEP 1
    """print("STEP 1: Downloading apps")
    download_samples(datasetfile)"""

    #STEP 2
    """print("STEP 2: Filtering downloaded apps")
    try:
        filter_downloaded_samples()
    except Exception as e:
        print(f"Unexpected error: {e}")"""


    #STEP 3
    """print("STEP 3: Sonifying filtered apps")
    sonification()"""

    #STEP 4
    """print("STEP 4: Extracting peaks from sonified apps")
    add_tracks_simplified(datasetfile)"""

    """input_file = 'Dataset/one_million_sha256_part1_subpart1_single_dex_files.csv'  # replace with your input file path
    output_file = 'Dataset/one_million_sha256_part1_subpart1_single_dex_files.csv'  # replace with your desired output file path

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Rename the first column to "Hash"
    df.columns.values[0] = 'Hash'

    # Add the new columns with empty values
    df['Sample'] = ''
    df['Type'] = ''
    df['Family'] = ''
    df['packed_sample'] = ''

    # Save the processed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)"""