from Cognitext.config.configuration import ConfigurationManager
from Cognitext.components.DataIngestion import DataIngestion


class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_ingestion_config()
        data_ingestion = DataIngestion(config = data_ingestion_config)
        data_ingestion.download_Data()
        data_ingestion.unzip_data()