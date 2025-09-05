import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)
pd.set_option("display.max_colwidth", None)
import os
from pathlib import Path
import platform
import logging


# ==========================
# Path Setup
# ==========================
def get_paths():
    system_name = platform.system()
    logging.info(f"Detected OS: {system_name}")

    if system_name == "Linux":
        base_dir = Path("/home/pee/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result")
    elif system_name == "Darwin":
        base_dir = Path("/Users/Jumma/git_repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result")
    elif system_name == "Windows":
        base_dir = Path("C:/Users/CAMT/repo/github_api_extractor/resources/tsdetect/test_smell_flink/latest_result")
    else:
        raise EnvironmentError("Unsupported platform")

    return base_dir, base_dir / "wtl_20_summary", base_dir / "tables"

