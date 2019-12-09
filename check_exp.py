import os
import pandas as pd
from subprocess import call
import shutil

HOME = '/cluster/home/sayvaz/rss'
dir_path = os.path.join(HOME, 'log_dir')
dirs = os.listdir(dir_path)

'''
for d in dirs:
    if 'TRD' in d:
        remove_path = os.path.join(HOME, 'log_dir', d)
        print(remove_path)
        shutil.rmtree(remove_path)
        print('removed')
'''

dirs = os.listdir(dir_path)
for d in dirs:
    try:
        if 'MARIO' in d:
            check_iter = 1464
            csv = os.path.join(dir_path, d, 'inter', 'progress.csv')
            data = pd.read_csv(csv)
            iters = data['iter'].tolist()

            if iters[-1] < check_iter and not('NSTD-1.0' in d or 'NSTD-0.5' in d):
                print('restarting experiments:', d, iters[-1])
                model_spec = os.path.join(HOME, 'model_specs', '{}.json'.format(d))
                subcommand = "python3 run.py --server_type LEONHARD --model_spec {} --restore_iter {}".format(model_spec, iters[-1])
                command = "bsub -n 8 '{}'".format(subcommand)
                os.system(command)
        elif 'COINRUN' in d:
            check_iter = 2441
            csv = os.path.join(dir_path, d, 'inter', 'progress.csv')
            data = pd.read_csv(csv)
            iters = data['iter'].tolist()
            if iters[-1] != check_iter:
                print('undone experiments: {}, iter: {}'.format(d, iters[-1]))
                model_spec = os.path.join(HOME, 'model_specs', '{}.json'.format(d))
                subcommand = "python3 run.py --server_type LEONHARD --model_spec {} --restore_iter {}".format(model_spec, iters[-1])
                command = "bsub -n 16 '{}'".format(subcommand)
                os.system(command)

    except:
        if 'MARIO' in d:
            print(d)
            model_spec = os.path.join(HOME, 'model_specs', '{}.json'.format(d))
            # subcommand = "python3 run.py --server_type LEONHARD --model_spec {} --restore_iter {}".format(model_spec, 1464)
            subcommand = "python3 run.py --server_type LEONHARD --model_spec {}".format(model_spec)
            command = "bsub -n 8 '{}'".format(subcommand)
            os.system(command)

        elif 'COINRUN' in d:
            print('undone experiments: {}, iter: {}'.format(d, iters[-1]))
            model_spec = os.path.join(HOME, 'model_specs', '{}.json'.format(d))
            subcommand = "python3 run.py --server_type LEONHARD --model_spec {} --restore_iter {}".format(model_spec, iters[-1])
            command = "bsub -n 16 '{}'".format(subcommand)
            os.system(command)