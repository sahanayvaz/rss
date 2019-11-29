import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def sliding_mean(data_array, window=5):  
    # data_array = np.array(data_array)  
    new_list = []
    for i in range(data_array.shape[0]):  
        indices = range(max(i - window + 1, 0),  
                        min(i + window + 1, data_array.shape[0]))  
        avg = 0
        for j in indices:
            avg += data_array[j]  
        avg /= float(len(indices))  
        new_list.append(avg)
    return np.asarray(new_list)

log_dir = './LEO-log_dir'

level = '3-1'
experiments = ['MARIO-1-1-baseline-TR-{}-v0'.format(level),
               'MARIO-1-1-RSS-TR-{}-v0'.format(level),
               'MARIO-1-1-RSS-NOISE-TR-{}-v0'.format(level),
               'MARIO-1-1-HH-TR-{}-v0'.format(level),
               'MARIO-{}-baseline-v0'.format(level)]
'''
level = '1-1'
experiments = ['MARIO-{}-RSS-v0'.format(level),
               'MARIO-{}-RSS-NOISE-v0'.format(level),
               'MARIO-{}-RSS-SPARSE-v0'.format(level),
               'MARIO-{}-RSS-SPARSE-v0'.format(level),
               'MARIO-{}-RSS-SPARSE-v0'.format(level),]
'''

plt.figure(figsize=(12, 9))
plotHandles = []
labels = []
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

data_csv = []

for exp in experiments:
    csv_path = os.path.join(log_dir, exp, 'inter', 'progress.csv')
    exp_data = np.genfromtxt(csv_path, dtype=float, delimiter=',', names=True)
    iterations = exp_data['iter'][:30] * (128 * 8 * 4) 
    rew_mean = sliding_mean(exp_data['trainrew_mean'])[:30]
    # rew_mean = exp_data['trainrew_mean']
    x, = plt.plot(iterations, rew_mean)
    plotHandles.append(x)
    labels.append('-'.join(exp.split('-')[1:5]))
    
    rew_std = sliding_mean(exp_data['trainrew_std'])[:30]

    data_csv.append([exp, rew_mean[-1], rew_std[-1]])

    plt.fill_between(iterations, rew_mean - rew_std,
                     rew_mean + rew_std, alpha=0.1)
plt.legend(plotHandles, labels, loc='lower right', ncol=1)
plt.savefig('./visuals/TR11to{}.png'.format(level))
plt.show()

csv_file = './visuals/exp_TR11to{}.csv'.format(level)

with open(csv_file, 'w') as _file:
    writer = csv.writer(_file, delimiter=',')
    for d in data_csv:
        writer.writerow(d)