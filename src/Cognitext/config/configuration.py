from Cognitext.utils.common import read_yaml, create_directories
from Cognitext.constants import *
from Cognitext.entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, PredictionConfig

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
    
    def get_trainer_config(self) -> ModelTrainerConfig:
            config = self.config.model_training
            params = self.params.DataLoaderParams

            
            get_training_config =  ModelTrainerConfig(
                root_dir = config.root_dir,
                data_path= config.data_path,
                save_dir= config.save_dir,
                batch_size = params.batch_size,
                max_length= params.max_length,
                stride= params.stride, 
                shuffle= params.shuffle, 
                drop_last= params.drop_last, 
                num_workers= params.num_workers,
                vocab_size= params.vocab_size,      
                emb_dim= params.emb_dim,            
                context_length= params.context_length,    
                n_heads= params.n_heads,            
                n_layers= params.n_layers,           
                drop_rate= params.drop_rate,       
                ff_dim= params.ff_dim,      
                qkv_bias= params.qkv_bias,        
                learning_rate= params.learning_rate,
                tokenizer= config.tokenizer
            )
            return get_training_config
    
    def get_prediction_config(self) -> PredictionConfig:
        config = self.config.prediction
        params = self.params.DataLoaderParams

        get_prediction_config = PredictionConfig(
            model = config.model,
            tokenizer = config.tokenizer,
            max_token = params.max_token
        )
        return get_prediction_config