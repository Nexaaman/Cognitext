from Cognitext.config.configuration import ConfigurationManager
from Cognitext.components.ModelTraining import Model, DatasetFromJSON, save
import os, torch
class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        get_training_config = config.get_trainer_config()
        dataset = DatasetFromJSON(config=get_training_config)
        train_loader, val_loader = dataset.get_train_val(dataset)

        model = Model(config=get_training_config)

        if os.path.exists(os.path.join(get_training_config.save_dir , "model.safetensors")):
            print(f"Model already present at =>  {get_training_config.save_dir}")

        else:
            model = (model.train_model(model,train_loader,val_loader, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")))
            save(model, get_training_config)