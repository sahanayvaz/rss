import os
import pandas as pd
from subprocess import call
import shutil

HOME = '/cluster/home/sayvaz/rss'
dir_path = os.path.join(HOME, 'log_dir')
dirs = os.listdir(dir_path)

for d in dirs:
    try:
        csv = os.path.join(dir_path, d, 'inter', 'progress.csv')
        data = pd.read_csv(csv)
        iters = data['iter'].tolist()
        if 'TRD' in d:
            dir_path = os.path.join(HOME, 'log_dir', d)
            print(dir_path)
            shutil.rmtree(d)

        if iters[-1] != 1464 and not('NSTD-1.0' in d or 'NSTD-0.5' in d):
            print('restarting experiments:', d, iters[-1])
            model_spec = os.path.join(HOME, 'model_specs', '{}.json'.format(d))
            subcommand = "python3 run.py --server_type LEONHARD --model_spec {} --restore_iter {}".format(model_spec, iters[-1])
            command = "bsub -n 8 '{}'".format(subcommand)
            os.system(command)
    except:
        pass