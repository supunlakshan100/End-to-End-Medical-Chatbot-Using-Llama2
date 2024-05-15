import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "research/trails.ipynb",
    "app.py",
    "store_index.py", 
    "static/.gitkeep",
    "setup.py",
    "templates/chat.html",   
     

    ]

# create files and directories
for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)

    # create directory if it does not exist
    if filedir!='':
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file {filename}")

    # create file if it does not exist
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0): # create file if it does not exist
        with open(filepath, 'w') as f:
            logging.info(f"Creating empty file; {filepath}")
    
    # log if file already exists
    else:
        logging.info(f"{filename} is already created")
    