from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    data_path: Path
    save_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer: str
    max_length: int
    stride: int 
    save_dir: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    save_dir: Path
    batch_size: 4
    max_length: int
    stride: int 
    shuffle: True 
    drop_last: True 
    num_workers: int
    vocab_size: int       
    emb_dim: int           
    context_length: int     
    n_heads: int             
    n_layers: int            
    drop_rate: int        
    ff_dim: int           
    qkv_bias: bool          
    learning_rate: float
    tokenizer: str


@dataclass(frozen=True)
class PredictionConfig:
    model: Path
    tokenizer: str
    max_token: int

