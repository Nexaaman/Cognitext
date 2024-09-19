import tiktoken
from datasets import load_from_disk
import torch
from torch.utils.data import Dataset, DataLoader
from Cognitext.entity import DataTransformationConfig
from Cognitext.utils.common import create_directories
import json
import csv


class DataTransformation(Dataset):
    def __init__(self, config = DataTransformationConfig):
        self.config = config
        create_directories([self.config.root_dir])

        Data = load_from_disk(self.config.data_path)
        Data = Data["text"]
        Data = " ".join(Data)

        self.input_ids = []
        self.target_ids = []

        self.tokenizer = tiktoken.get_encoding(self.config.tokenizer)
        token_ids = self.tokenizer.encode(Data , allowed_special={"|endoftext|"})

        for i in range(0, len(token_ids) - self.config.max_length, self.config.stride):
            input_chunk = token_ids[i: i + self.config.max_length]
            target_chunk = token_ids[i+1: i + self.config.max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def _len_(self):
        return len(self.input_ids)
    
    def __getitem__(self , idx):
        return self.input_ids[idx] , self.target_ids[idx]
    
    def save(self, file_type = "json"):
        file_path = self.config.save_dir
        data_to_save = [
            {
                "input_ids": input_tensor.tolist(),
                "target_ids": target_tensor.tolist()
            }
            for input_tensor, target_tensor in zip(self.input_ids, self.target_ids)
        ]

        if file_type == "json":
            with open(file_path, "w") as json_file:
                json.dump(data_to_save, json_file, indent=4)
            print(f"Data successfully saved to {file_path} as JSON.")
        
        elif file_type == "csv":
            with open(file_path, "w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["input_ids", "target_ids"])
                writer.writeheader()
                for row in data_to_save:
                    writer.writerow(row)
            print(f"Data successfully saved to {file_path} as CSV.")
        
        else:
            raise ValueError("Unsupported file type. Please choose either 'json' or 'csv'.")
