from typing import List
from collections import Counter
from utils import Dataset, get_topic_index
import numpy as np
import pandas as pd

from utils import GROUPS_NUMBER
import math

class EmModel:
    def __init__(self, dataset_file_path:str):
        self.dataset = Dataset(dataset_file_path)
        self._initialize()