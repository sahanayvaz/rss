import os 

import numpy as np
import pandas as pd
import deepdish as dd

def read_save(progress_csv, base_name, dict_):
    with open(progress_csv, 'r') as file:
        data = pd.read_csv(file)
        rew_mean = data['train-rew_mean'].tolist()
        # check iterations to be safe
        iters = data['iter'].tolist()
        if not len(rew_mean) == 60:
            print(progress_csv)

        if base_name in dict_.keys():
            dict_[base_name].append(rew_mean)
        else:
            dict_[base_name] = [rew_mean]


LEO_log_dir = './LEO-F-log_dir'
EULER_log_dir = './EULER-F-log_dir'

base_names = ['MARIO-RSS', 'MARIO-RSS-NOISE']
NL = [2, 3, 5]
KD = [50, 150, 250]

seeds = [0, 17, 41]

dict_exp = {}

KN = [25, 50, 100]
NSTD = 0.1

for b in base_names:
    for s in seeds:
        for n in NL:
            if b == 'MARIO-RSS':
                for k in KD:
                    base_name = 'MARIO_RSS_NL_{}_KD_{}'.format(n, k)
                    dir_name = '{}-seed-{}-NL-{}-KD-{}-1-1-v0'.format(b, s, n, k)
                    try:
                        progress_csv = os.path.join(LEO_log_dir, dir_name, 'inter', 'progress.csv')
                        read_save(progress_csv, base_name, dict_exp)    
                    except:
                        progress_csv = os.path.join(EULER_log_dir, dir_name, 'inter', 'progress.csv')
                        read_save(progress_csv, base_name, dict_exp)

            # we did not test NL=5 for NOISE
            elif (b == 'MARIO-RSS-NOISE') and (n != 5):
                for v in KN:
                    base_name = 'MARIO_RSS_NOISE_NL_{}_KN_{}_NSTD_{}'.format(n, v, NSTD)
                    dir_name = '{}-seed-{}-NL-{}-KN-{}-NSTD-{}-1-1-v0'.format(b, s, n, v, NSTD)
                    try:
                        progress_csv = os.path.join(LEO_log_dir, dir_name, 'inter', 'progress.csv')
                        read_save(progress_csv, base_name, dict_exp)    
                    except:
                        progress_csv = os.path.join(EULER_log_dir, dir_name, 'inter', 'progress.csv')
                        read_save(progress_csv, base_name, dict_exp)

save_dict = {}
for k in dict_exp.keys():
    rew_mean_list = dict_exp[k]
    
    for r in rew_mean_list:
        print(np.asarray(r).shape)
    rew_mean_arr = np.asarray(rew_mean_list)


    rew_mean = np.mean(rew_mean_arr, axis=0)
    rew_std = np.std(rew_mean_arr, axis=0)
    save_dict[k] = {'rew_mean_arr': rew_mean_arr,
                    'rew_mean': rew_mean,
                    'rew_std': rew_std}
    print(k, rew_mean[-1])

dd.io.save('sparse_results.h5', save_dict)
print('saved')