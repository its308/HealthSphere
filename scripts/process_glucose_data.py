import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



base_path='data/diabetes_subset_sensor_data'


def find_bb_files(base_path):
    bb_files = []
    for root,dirs,files in os.walk(base_path):
        for file in files:
            if file.endswith('_BB.csv'):
                bb_files.append(os.path.join(root,file))
    return bb_files

bb_files=find_bb_files(base_path)





