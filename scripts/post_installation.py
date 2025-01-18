import os
import subprocess

from pyvisim._config import ROOT

_MODEL_FILES_DIR = ROOT / "pyvisim/res/model_files"
_ZIP_FILE_PATH = _MODEL_FILES_DIR / "clustering_models.7z"

def unpack_clustering_models():
    """
    Unpacks the clustering_models.7z file in the res/model_files directory.

    :raises FileNotFoundError: If the zipped file is not found.
    :raises Exception: If the unpacking command fails.
    """
    if not os.path.exists(_ZIP_FILE_PATH):
        raise FileNotFoundError(f"Zipped file not found: {_ZIP_FILE_PATH}")

    print(f"Unpacking {_ZIP_FILE_PATH}...")
    try:
        subprocess.run(["7z", "x", _ZIP_FILE_PATH, f"-o{_MODEL_FILES_DIR}"], check=True)
        print("Unpacking successful.")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Unpacking failed: {e}")

    os.remove(_ZIP_FILE_PATH)
    print(f"Unpacked files are available in: {_MODEL_FILES_DIR}")

if __name__ == "__main__":
    unpack_clustering_models()