from enum import Enum
from pathlib import Path

# class syntax
class Constants(Enum):
    root_path = root = Path.cwd().parent
    raw_file = 'raw_dataset'
    processed_file = 'processed_dataset'
    path_raw = f'/data/raw/'
    path_processed = f'/data/processed/'