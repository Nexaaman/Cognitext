from Cognitext.utils.common import read_yaml, create_directories
from Cognitext.constants import *
from Cognitext.entity import DataIngestionConfig, DataTransformationConfig

class ConfigurationManager:
    def __init__(self, param = PARAMS_FILE_PATH, config = CONFIG_FILE_PATH):
        self.params = read_yaml(param)
        self.config = read_yaml(config)

        create_directories([self.config.artifacts_root])

    def get_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        get_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            data_path = config.data_path,
            save_dir = config.save_dir
        )
        return get_ingestion_config
    
    def get_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.DataLoaderParams

        get_transformation_connfig = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path= config.data_path,
            tokenizer= config.tokenizer,
            max_length= params.max_length,
            stride= params.stride,
            save_dir=config.save_dir
        )
        return get_transformation_connfig