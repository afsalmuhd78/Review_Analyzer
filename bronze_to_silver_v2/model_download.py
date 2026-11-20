import os
import tarfile
import urllib.request
from tqdm import tqdm
from os import environ
from dotenv import load_dotenv

load_dotenv()

MODEL_URL= "https://code.turbolab.in/api/v4/projects/20694/packages/ml_models/3/files/robust-sentiment-analysis.tar.gz"
FILE_NAME = MODEL_URL.split("/")[-1]
FOLDER_NAME = FILE_NAME.split(".")[0]
HEADERS = {"Authorization": f"Bearer {environ['GITLAB_MODEL_TOKEN']}"}


def download():
    # Directory where models will be stored
    models_dir = os.path.expanduser("~/.models")

    # Check if model already exists
    model_path = os.path.join(models_dir, FOLDER_NAME)
    if os.path.isdir(model_path):
        print("Model already exists")
    else:
        print("Downloading model ...")
        
        # Ensure the models directory exists
        os.makedirs(models_dir, exist_ok=True)

        # Define the file path for the downloaded model
        tar_file_path = os.path.join(models_dir, FILE_NAME)

        # Function to download the file with progress bar
        def download_file(url, dest_path, headers=None):
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                total_size = int(response.getheader('Content-Length'))
                with open(dest_path, 'wb') as out_file:
                    # Use tqdm to show progress for downloading
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=FILE_NAME) as pbar:
                        while True:
                            chunk = response.read(1024)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            pbar.update(len(chunk))

        # Download the model with progress bar
        download_file(MODEL_URL, tar_file_path, HEADERS)

        # Extract the tar.gz file with tqdm progress
        with tarfile.open(tar_file_path, "r:gz") as tar_ref:
            members = tar_ref.getmembers()
            with tqdm(total=len(members), desc="Extracting") as pbar:
                for member in members:
                    tar_ref.extract(member, models_dir)
                    pbar.update(1)

        # Remove the tar.gz file after extraction
        os.remove(tar_file_path)
        
        print("Download completed")
