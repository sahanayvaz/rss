import os 

import numpy as np
import pandas as pd

log_dir = './LEO-F-log_dir'
list_dir = os.listdir(log_dir)

base_names = ['MARIO-RSS-seed', 'MARIO-RSS-NOISE-seed']
for l in list_dir:
    for b in base_names:
        if b in l:
            