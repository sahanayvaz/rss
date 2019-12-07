import os
import pandas as pd
from subprocess import call

HOME = '/cluster/home/sayvaz'
dir_path = os.path.join(HOME, 'log_dir')
dirs = os.listdir(dir_path)

for d in dirs:
    try:
        csv = os.path.join(dir_path, d, 'inter', 'progress.csv')
        data = pd.read_csv(csv)
        iters = data['iter'].tolist()
        if iters[-1] != 1464 and not('NSTD-1.0' in d or 'NSTD-0.5' in d):
            print('restarting experiments:', d, iters[-1])
            model_spec = '/cluster/home/sayvaz/model_specs/{}.json'.format(d)
            call(["bsub", "-n", "8", "python3", "run.py", 
                  "--server_type", "LEONHARD", 
                  "--model_spec", model_spec, 
                  "--restore_iter", iters[-1]])
    except:
        pass