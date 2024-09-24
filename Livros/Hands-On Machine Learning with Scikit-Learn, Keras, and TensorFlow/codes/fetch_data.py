import os
import tarfile
import urllib
import urllib.request

def fetch_data(url, path, dataset_name):
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, dataset_name + ".tgz")
    urllib.request.urlretrieve(url, tgz_path)
    dataset_tgz = tarfile.open(tgz_path)
    dataset_tgz.extractall(path=path)
    dataset_tgz.close()

"""

To download the data from chapter 2: Follow the below code

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
DOWNLOAD_PATH = os.path.join("datasets", "housing")
URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

fetch_data(URL, DOWNLOAD_PATH, "housing")

"""

