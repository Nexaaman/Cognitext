import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s: ')

project_name = "Cognitext"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "research/trials.ipynb",
    "setup.py"
]

for file_path in list_of_files:
    file_path =  Path(file_path)
    dir, file = os.path.split(file_path)

    if dir!="":
        os.makedirs(dir, exist_ok=True)
        logging.info(f"Creating Directory: {dir} for file {file}")
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
            with open(file_path, 'w') as f:
                pass
                logging.info(f"Creating empty file: {file_path}")
        
    else:
        logging.info(f"{file} is already exists")