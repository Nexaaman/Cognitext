import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch.utils.data import Dataset, DataLoader
import json, csv
from torch.amp import GradScaler
from torch.amp import autocast

from Cognitext.entity import ModelTrainerConfig
from Cognitext.utils.common import create_directories
from safetensors.torch import save_file, load_file

class DatasetFromJSON(Dataset):
    def __init__(self,config = ModelTrainerConfig):
        self.config = config
        self.tokenizer = self.config.tokenizer
        self.json_path = self.config.data_path

        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        self.input_ids = [torch.tensor(item["input_ids"]) for item in self.data]
        self.target_ids = [torch.tensor(item["target_ids"]) for item in self.data]

        self.tokenizer = self.config.tokenizer
        self.max_length = self.config.max_length
        self.stride = self.config.stride

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    def get_train_val(self , dataset):
        train_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, val_loader
    

class Model(nn.Module):
    def __init__(self, config=ModelTrainerConfig):
        super().__init__()
        self.config = config

        create_directories([self.config.root_dir])

        self.tok_emb = nn.Embedding(self.config.vocab_size , self.config.emb_dim)
        self.pos_emb = nn.Embedding(self.config.max_length , self.config.emb_dim)
        self.drop_emb = nn.Dropout(self.config.drop_rate)


        self.trf_blocks = nn.Sequential(
            *[self.TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        )

        self.final_norm = nn.LayerNorm(self.config.emb_dim)

        self.out_head = nn.Linear(self.config.emb_dim ,self.config.vocab_size, bias=False)

        self.tokenizer = self.config.tokenizer

    
    class TransformerBlock(nn.Module):

        def __init__(self, config=ModelTrainerConfig):
            super().__init__()
            self.config = config

            self.att = nn.MultiheadAttention(self.config.emb_dim, self.config.n_heads, dropout=self.config.drop_rate)
            self.ff = Model.FeedForward(self.config)

            self.norm1 = nn.LayerNorm(self.config.emb_dim)
            self.norm2 = nn.LayerNorm(self.config.emb_dim)
            self.drop = nn.Dropout(self.config.drop_rate)

        def forward(self, x):
            shortcut = x
            x = self.norm1(x)
            x, _ = self.att(x,x,x)
            x = self.drop(x)
            x = x + shortcut

            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            self.norm2(x)
            x = x+ shortcut

            return x
        
    class FeedForward(nn.Module):

        def __init__(self, config=ModelTrainerConfig):
            super().__init__()

            self.config = config

            self.ff = nn.Sequential(
                nn.Linear(self.config.emb_dim, self.config.ff_dim),
                nn.GELU(),
                nn.Linear(self.config.ff_dim, self.config.emb_dim),
                nn.Dropout(self.config.drop_rate)
            )

        def forward(self, x):
            return self.ff(x)

    def forward(self, in_idx):

            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            
            x = tok_embeds + pos_embeds
            x = self.drop_emb(x)
            
            x = self.trf_blocks(x)
           
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits
        
    def calc_loss_batch(self, input_batch, target_batch, device):
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
           
            logits = self(input_batch)

            loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            return loss
        
    def calc_loss_loader(self, data_loader, device, num_batches=None):
            total_loss = 0.
            if len(data_loader) == 0:
                return float("nan")
            elif num_batches is None:
                num_batches = len(data_loader)
            else:
                num_batches = min(num_batches, len(data_loader))

            
            for i, (input_batch, target_batch) in enumerate(data_loader):
                if i < num_batches:
                   
                    loss = self.calc_loss_batch(input_batch, target_batch, device)
                    total_loss += loss.item()
                else:
                    break

           
            return total_loss / num_batches
    
    def evaluate_model(self, train_loader, val_loader, device, eval_iter):
        self.eval()  
        with torch.no_grad():
            train_loss = self.calc_loss_loader(train_loader, device, eval_iter)
            val_loss = self.calc_loss_loader(val_loader, device, eval_iter)
        self.train()  
        return train_loss, val_loss

    def text_to_token_ids(self, text):
        
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)  
        return encoded_tensor

    def token_ids_to_text(self, token_ids):
        
        flat = token_ids.squeeze(0)  
        return self.tokenizer.decode(flat.tolist())

    def generate_and_print_sample(self, start_context, device):
        self.eval() 
        context_size = self.pos_emb.weight.shape[0]
        encoded = self.text_to_token_ids(start_context).to(device)
        with torch.no_grad():
            token_ids = self.generate_text_simple(encoded, 50, context_size)
            decoded_text = self.token_ids_to_text(token_ids)
            print(decoded_text.replace("\n", " ")) 
        self.train()  

    def generate_text_simple(self, idx, max_new_tokens, context_size):
       
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]  
            with torch.no_grad():
                logits = self(idx_cond)  
            logits = logits[:, -1, :] 
            probas = torch.softmax(logits, dim=-1)  
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)  
            idx = torch.cat((idx, idx_next), dim=1)  
        return idx
    
    def train_model(self,model, train_loader, val_loader, device , epochs=5, eval_interval=100):
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
        model = model.to(device)

        scaler = GradScaler()
        train_losses, val_losses = [], []
        tokens_seen = []

        accumulation_steps = 4 
        global_step = 0

        for epoch in range(epochs):
            model.train()

            epoch_loss = 0
            optimizer.zero_grad()

            for i, (input_batch, target_batch) in enumerate(train_loader):
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            
                with autocast(device_type='cuda' if device == torch.device('cuda') else 'cpu'):
                    logits = model(input_batch)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))

        
                scaler.scale(loss).backward()

            
                if (i + 1) % accumulation_steps == 0:
            
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                
                    global_step += 1
                    epoch_loss += loss.item()

                
                    scheduler.step()

            
                if global_step % eval_interval == 0:
                    val_loss = model.evaluate_model(train_loader, val_loader, device, eval_iter=10)
                    print(f'Epoch {epoch+1}, Step {global_step}, Train Loss: {epoch_loss / (i+1)}, Val Loss: {val_loss}')
                    train_losses.append(epoch_loss / (i + 1))
                    val_losses.append(val_loss)
                    tokens_seen.append(global_step * self.config.batch_size)

            print(f"End of Epoch {epoch+1}: Average Training Loss: {epoch_loss / len(train_loader)}")

def save(model, config = ModelTrainerConfig):
        config = config
        create_directories([config.save_dir])
        
        model_state_dict = model.state_dict()
        save_file(model_state_dict, os.path.join(config.save_dir, "model.safetensors"))

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        optimizer_state_dict = optimizer.state_dict()
        save_file(optimizer_state_dict, os.path.join(config.save_dir, "optimizer_state.safetensors"))


