import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

exp_dir = ''

for e in exp_dir:
    for restore_iter in range(0, 1400, 100):
    npz_save_file = os.path.join(surface_dir, 'surface-{}.npz'.format(restore_iter))
    data = np.load(npz_save_file)

    print('surface projections...')

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm, edgecolor='none')
    ax.set_title('loss-surface-{}'.format(self.restore_iter))
    plt.show()
    