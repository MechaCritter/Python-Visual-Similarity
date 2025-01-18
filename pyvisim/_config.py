import logging
import logging.config
import pathlib

import yaml

# -Config for the dataset- #
ROOT = pathlib.Path(__file__).parent.parent
LOG_FILE_PATH = ROOT / "res/logs/log_msgs.log"
PICKLE_MODEL_FILES_PATH = ROOT / "pyvisim/res/model_files"

# - Logging - #
def setup_logging(default_path=rf"{ROOT}/res/logging_config.yaml", default_level=logging.INFO):
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


