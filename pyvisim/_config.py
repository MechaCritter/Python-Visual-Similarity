import logging
import logging.config
import pathlib
import os

import yaml

# -Config for the dataset- #
ROOT = pathlib.Path(__file__).parent.parent
LOG_FOLDER = ROOT / "res/logs"
os.makedirs(LOG_FOLDER, exist_ok=True)
LOG_FILE_PATH = LOG_FOLDER / "log_msgs.log"

RES_FOLDER  = ROOT / "pyvisim/res"
PICKLE_MODEL_FILES_PATH = RES_FOLDER / "model_files"
LOGGER_FILE = RES_FOLDER / "logging_config.yaml"

# - Logging - #
def setup_logging(default_path=LOGGER_FILE, default_level=logging.WARNING):
    """Setup logging configuration"""
    try:
        with open(default_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        try:
            config["handlers"]["file_handler"]["filename"] = str(LOG_FILE_PATH)
        except Exception as e:
            print(f"Error in Logging Configuration: {e}. Cannot set output path for log file.")
        logging.config.dictConfig(config)
    except Exception as e:
        print(f"Error in Logging Configuration: {e}")
        logging.basicConfig(level=default_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


