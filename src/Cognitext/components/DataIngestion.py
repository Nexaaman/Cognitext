import os
import urllib.request as request
from Cognitext.logging import logger
from Cognitext.utils.common import get_size
import zipfile
from pathlib import Path
from Cognitext.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    def download_Data(self):
        if not os.path.exists(self.config.data_path):
            filename, headers = request.urlretrieve(url=self.config.source_URL, filename=self.config.data_path)
            logger.info(f"{filename}, Downloaded Sucessfully!")
        else:
            logger.info(f"File already exists of size {get_size(Path(self.config.data_path))}")

    def unzip_data(self):
        unzip_path = self.config.save_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.data_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)