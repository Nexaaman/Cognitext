from Cognitext.config.configuration import ConfigurationManager
from Cognitext.components.DataTransformation import DataTransformation

class DataTransformationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        get_transformation_config = config.get_transformation_config()
        transformation = DataTransformation(config = get_transformation_config)
        transformation._len_()
        transformation.save()