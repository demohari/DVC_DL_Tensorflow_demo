import os
import argparse
import shutil
import logging
from tqdm import tqdm
from src.utils.all_utils import read_yaml, create_directory

logging_str = "[%(asctime)s: %(levelname)s:%(module)s:%(lineno)d]:%(message)s"
log_dir = "logs"
# create_directory([log_dir])
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "running_logs.log"),
    level=logging.INFO,
    format=logging_str,
    filemode="a",
)


def copy_file(source_download_dir, local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    N = len(list_of_files)
    for file in tqdm(
        list_of_files,
        total=N,
        desc=f"copying_file_from {source_download_dir} to {local_data_dir}",
        colour="green",
    ):
        src = os.path.join(source_download_dir, file)
        dest = os.path.join(local_data_dir, file)
        shutil.copy(src, dest)


def get_data(config_path):
    config = read_yaml(config_path)

    source_download_dirs = config["source_download_dirs"]
    # print("source: ", source_download_dirs)
    local_data_dirs = config["local_data_dirs"]
    # print("local: ", local_data_dirs)

    for source_download_dir, local_data_dir in tqdm(
        zip(source_download_dirs, local_data_dirs),
        total=2,
        desc="list of folders",
        colour="red",
    ):
        create_directory([local_data_dir])
        # print("local: ", local_data_dirs)
        copy_file(source_download_dir, local_data_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>>>>>>>>>>stage01 started")
        get_data(config_path=parsed_args.config)
        logging.info("stage01 completed successfully data saved in local <<<<<<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e
