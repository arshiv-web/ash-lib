import os
import requests
import zipfile
import shutil
from pathlib import Path


def walk_through_dir(dir_path: str):
    """
    Walks through dir_path and prints the count of files and subdirectories.
    
    Args:
        dir_path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} files in '{dirpath}'.")


def get_data(source: str, 
             destination: str,
             remove_source: bool = True) -> Path:
    """
    Acquires data from a URL or a local ZIP file and extracts it.
    
    Args:
        source (str): URL to a zip file OR path to a local zip file.
        destination (str): Path to the folder where data should be extracted.
        remove_source (bool): If True, deletes the zip file after extraction 
                              (only applies if downloaded from URL).
    """
    target_dir = Path(destination)
    
    # If destination exists, we assume data is ready
    if target_dir.is_dir():
        print(f"[INFO] {target_dir} exists, skipping ingestion.")
        return target_dir

    print(f"[INFO] Creating {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Handle Web URL
    if source.startswith(("http://", "https://")):
        target_file = target_dir / Path(source).name
        print(f"[INFO] Downloading from URL: {source}...")
        
        with open(target_file, "wb") as f:
            request = requests.get(source)
            f.write(request.content)
            
        should_remove = remove_source # User decision applies here

    # 2. Handle Local File
    elif os.path.exists(source):
        print(f"[INFO] Using local file: {source}...")
        target_file = Path(source)
        should_remove = False # NEVER delete user's original local file automatically

    else:
        raise FileNotFoundError(f"Source not found: {source}")

    # 3. Unzip
    if zipfile.is_zipfile(target_file):
        with zipfile.ZipFile(target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping to {target_dir}...")
            zip_ref.extractall(target_dir)
    else:
        print(f"[WARN] {target_file} is not a zip file. Copied/Downloaded only.")
        if source != str(target_file): # Avoid copy if already there
             shutil.copy(source, target_dir)

    # 4. Cleanup (Only for downloads)
    if should_remove and source.startswith("http"):
        os.remove(target_file)
        
    return target_dir