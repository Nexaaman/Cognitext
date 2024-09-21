from Cognitext.components.ModelTraining import Model
import torch
import tiktoken
from Cognitext.entity import PredictionConfig
from Cognitext.config.configuration import ConfigurationManager
config = ConfigurationManager()
get_training_config = config.get_trainer_config()

from safetensors.torch import load_file

class PredictionPipleine:
    def __init__(self , config = PredictionConfig):
        self.config = config

        self.model = Model(get_training_config)
        self.tokenizer = tiktoken.get_encoding(self.config.tokenizer)
        self.max_token = self.config.max_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_weights = load_file(self.config.model)  # config.model_path should point to your .safetensors file
        self.model.load_state_dict(model_weights)     # Load the weights into the model
        
        # Move model to the appropriate device
        self.model.to(self.device)



    def predict(self,text):
        encoded = self.tokenizer.encode(text)  # Tokenize the text
        input_ids = torch.tensor([encoded])
        self.model.eval()  
        context_size = self.model.pos_emb.weight.shape[0]

        with torch.no_grad():  
            for _ in range(self.max_token):
                input_cond = input_ids[:, -context_size:]

                logits = self.model(input_cond)

                logits = logits[:, -1, :]  

                probabilities = torch.softmax(logits, dim=-1)
                next_token_id = torch.argmax(probabilities, dim=-1, keepdim=True)  

                input_ids = torch.cat([input_ids, next_token_id], dim=1)

        flat = input_ids.squeeze(0)  
        generated_text =  self.tokenizer.decode(flat.tolist())

        return generated_text


