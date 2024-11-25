
import gdown
import os

# Create the data directory if it does not exist
if not os.path.exists('data'):
    os.makedirs('data')


# URL to the file on Google Drive
url = 'https://drive.google.com/uc?export=download&id=1KzyZzgI1Wj7msuy6_qUU2mSXd50zwUtH'
output = 'data/dataset_large.zip'

gdown.download(url, output, quiet=False)

# Unzip the file
import zipfile
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('data')
