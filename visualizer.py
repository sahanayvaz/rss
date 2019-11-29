
import tensorflow as tf
import numpy as np

ckpt_file = "./save_dir/MARIO-1-1-baseline-v0/model-0"

npz_file = "./save_dir/MARIO-1-1-baseline-v0/extra-0.npz"
data = np.load(npz_file)

obs=data['obs']
acs=data['acs']
nlps=data['nlps']
advs=data['advs']
oldvpreds=data['oldvpreds']
rets=data['rets']
cliprange=data['cliprange']

# get random directions

# normalize directions

# create d1 and d2

var_dict = {}
dir_dict = {0: {}, 1: {}}
# we create r_dir only once
for var_ckpt in tf.train.list_variables(ckpt_file):
    # remove learning-related variables
    # we are also ignoring biases
    not_count = 'beta' in var_ckpt[0] or 'Adam' in var_ckpt[0] or 'bias' in var_ckpt[0]
    if not not_count:
        # this gives the shapes of variables
        var_shape = var_ckpt[1]
        var = tf.train.load_variable(ckpt_file, var_ckpt[0])
        print(var.shape)
        print(var_ckpt[0], var_shape)
        
        var_dict[var_ckpt[0]] = var

        for i in range(2):
            r_dir = np.random.normal(size=var_shape)
            # this means convolution
            if len(var_shape) > 3:
                # normalize cnns
                num_filter = var_shape[-1]
                for ind in range(num_filter):
                    fro_weight = np.linalg.norm(var[:, :, :, ind])
                    fro_dir = np.linalg.norm(r_dir[:, :, :, ind])
                    r_dir[:, :, :, ind] = (r_dir[:, :, :, ind] / fro_dir) * fro_weight
            else:
                fro_weight = np.linalg.norm(var)
                fro_dir = np.linalg.norm(r_dir)
                r_dir = (r_dir / fro_dir) * fro_weight

            dir_dict[i][var_ckpt[0]] = r_dir

print(var_dict.keys())
print('done')