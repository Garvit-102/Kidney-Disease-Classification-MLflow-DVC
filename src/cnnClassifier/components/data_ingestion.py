import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)

            logger.info(
                f"Downloading file from: {dataset_url} into: {zip_download_dir}"
            )

            gdown.download(
                url=dataset_url,
                output=zip_download_dir,
                quiet=False
            )

            logger.info(
                f"Downloaded file from: {dataset_url} into: {zip_download_dir}"
            )

        except Exception as e:
            raise e


    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

   