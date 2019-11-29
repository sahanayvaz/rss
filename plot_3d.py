import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

exp_dir = ['./LEO-vis_dir/MARIO-3-1-baseline-v0/']

for e in exp_dir:
    for restore_iter in range(0, 1000, 200):
        npz_save_file = os.path.join(e, 'surface-{}.npz'.format(restore_iter))
        data = np.load(npz_save_file)

        print('surface projections...')

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_surface(data['xs'], data['ys'], data['zs'], cmap=cm.coolwarm, edgecolor='none')
        ax.set_title('loss-surface-{}'.format(restore_iter))
        plt.show()
        