import os
import numpy as np

class DataLoader:
    def __init__(self):
        self.folder_path = None
        self.file_list = []
        self.current_idx = 0
        self.current_file = None
        self.current_data = None
        
    