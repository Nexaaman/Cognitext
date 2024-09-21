from Cognitext.pipeline.Stage_Data_Ingestion import DataIngestionPipeline
from Cognitext.pipeline.Stage_Data_Transformation import DataTransformationPipeline
from Cognitext.pipeline.Stage_Model_training import ModelTrainingPipeline
from Cognitext.logging import logger

STAGE_NAME = "Data Ingestion"

try:
    logger.info(f"************{STAGE_NAME}************")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f"************{STAGE_NAME} completed Succesfully************")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation"

try:
    logger.info(f"************{STAGE_NAME}************")
    data_ingestion = DataTransformationPipeline()
    data_ingestion.main()
    logger.info(f"************{STAGE_NAME} completed Succesfully************")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training"

try:
    logger.info(f"************{STAGE_NAME}************")
    data_ingestion = ModelTrainingPipeline()
    data_ingestion.main()
    logger.info(f"************{STAGE_NAME} completed Succesfully************")

except Exception as e:
    logger.exception(e)
    raise e