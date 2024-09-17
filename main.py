from Cognitext.pipeline.Stage_Data_Ingestion import DataIngestionPipeline
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