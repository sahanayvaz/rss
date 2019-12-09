import numpy as np
import tensorflow as tf
import multiprocessing
import random

def guess_available_cpus():
    return int(multiprocessing.cpu_count())

def setup_tensorflow_session():
    # i do not want too much overhead on my cpus
    # because i will be running multiple experiments at the same time
    num_cpu = guess_available_cpus()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.69,
                                allow_growth=True)

    # check if using num_cpu = 32 (our max) causes performance problems???
    tf_config = tf.ConfigProto(gpu_options=gpu_options,
                               inter_op_parallelism_threads=num_cpu,
                               intra_op_parallelism_threads=num_cpu,
                               allow_soft_placement=True)
    return tf.Session(config=tf_config)


def set_global_seeds(seed):
    import tensorflow as tf
    from gym.utils.seeding import hash_seed
    seed = hash_seed(seed, max_bytes=4)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def activ(activation):
    if activation == 'relu':
        return tf.nn.relu
    elif activation == None:
        return None
    else:
        raise NotImplementedError()

nature_cnn_spec = [{'filters': 32, 'kernel_size': 8, 'strides': (4, 4)},
                   {'filters': 64, 'kernel_size': 4, 'strides': (2, 2)},
                   {'filters': 64, 'kernel_size': 3, 'strides': (1, 1)}]

# initializer are taken from coinrun
def nature_cnn(out, activation, batchnormalize, init_scale=1.0, init_bias=0.0, coinrun=False):
    """
    Model used in the paper "Human-level control through deep reinforcement learning" 
    https://www.nature.com/articles/nature14236
    """

    if coinrun:
        print('''

            coinrun observation normalization / 255.0

            ''')

        out = tf.cast(out, tf.float32) / 255.
    
    activation = activ(activation)

    # bn = tf.layers.batch_normalization if batchnormalize else lambda x: x

    for i, n in enumerate(nature_cnn_spec):
        # default padding is VALID
        out = tf.layers.conv2d(out, filters=n['filters'], kernel_size=n['kernel_size'], 
                                  strides=n['strides'], activation=activation,
                                  kernel_initializer=tf.initializers.orthogonal(init_scale),
                                  bias_initializer=tf.constant_initializer(init_bias),
                                  name='conv2d-{}'.format(i))

    '''
    out = bn(tf.layers.conv2d(out, filters=64, kernel_size=4, strides=(2, 2), activation=activation,
                              kernel_initializer=tf.initializers.orthogonal(init_scale),
                              bias_initializer=tf.constant_initializer(init_bias)))

    out = bn(tf.layers.conv2d(out, filters=64, kernel_size=3, strides=(1, 1), activation=activation,
                              kernel_initializer=tf.initializers.orthogonal(init_scale),
                              bias_initializer=tf.constant_initializer(init_bias)))
    '''
    # we return unflattened output
    return out

def flatten(out):
    return tf.reshape(out, (-1, np.prod(out.get_shape().as_list()[1:])))

def batch_norm(out):
    return tf.layers.batch_normalization(out)

def fc(out, units, activation, batchnormalize, init_scale=1.0, init_bias=0.0):
    bn = tf.layers.batch_normalization if batchnormalize else lambda x: x
    activation = activ(activation)
    return bn(tf.layers.dense(out, units=units, activation=activation,
                              kernel_initializer=tf.initializers.orthogonal(init_scale),
                              bias_initializer=tf.constant_initializer(init_bias)))

def feat_v0(out, feat_dim, activation):
    # perception output (feature embeddings)
    out = fc(out, feat_dim, activation='relu', batchnormalize=False, init_scale=np.sqrt(2))
    return out
        
def single_rss(inpt, feat_dim, activation, keep_dim, add_noise=False, keep_noise=0, 
               num_layers=2, noise_std=1.0, layer_name='', base_name='', transfer_name=''):
    in_feat_dim = inpt.get_shape().as_list()[-1]

    # print('in_feat_dim: {}'.format(in_feat_dim))

    # really unoptimized, will work on it LATER
    random_idx = []

    for i in range(feat_dim):
        r_id = np.random.choice(in_feat_dim, keep_dim, replace=False)
        for r in r_id:
            random_idx.append([r, i])

    weight_initializer = tf.initializers.orthogonal(1.0)(shape=(keep_dim, feat_dim))
    weight_init = tf.reshape(weight_initializer, [(keep_dim) * feat_dim])
    # sparse_weights = tf.Variable(initial_value=weight_initializer, trainable=True, dtype=tf.float32, name='sparse_weights')
    # print('building sparse_tensor...')

    # when we define sparse_weights, this is how we do it
    sparse_weights = tf.SparseTensor(indices=random_idx, values=weight_init, dense_shape=(in_feat_dim, feat_dim))
    # print('success...')

    dense_weights = tf.sparse.to_dense(sparse_weights, default_value=0.0, validate_indices=False)

    weights = tf.Variable(dense_weights, trainable=True, dtype=tf.float32, name='{}_weights{}{}'.format(base_name, layer_name, transfer_name))
    bias_initializer = tf.constant_initializer(0.0)(shape=(feat_dim,))
    biases = tf.Variable(initial_value=bias_initializer, trainable=True, dtype=tf.float32, name='{}_biases{}{}'.format(base_name, layer_name, transfer_name))

    # poor man's added noise
    # this should not be like that, we should be changing the connections
    # this added noise should work like dropout, but i do not want to use 
    # dropout layer here (because of tensorflow's scaling)
    if add_noise:
        # get random weights
        random_w_idx = []
        for i in range(feat_dim):
            r_id = np.random.choice(in_feat_dim, keep_noise, replace=False)
            for r in r_id:
                random_w_idx.append([r, i])

        # do not forget to cancel them out
        random_w = tf.random.normal(shape=[(keep_noise) * feat_dim],
                                    mean=0.0,
                                    stddev=noise_std,
                                    name='random_{}_{}{}'.format(base_name, layer_name, transfer_name))

        sparse_r_weights = tf.SparseTensor(indices=random_w_idx, values=random_w, dense_shape=(in_feat_dim, feat_dim))
        dense_r_weights = tf.sparse.to_dense(sparse_r_weights, default_value=0.0, validate_indices=False)
        weights = tf.add(weights, dense_r_weights)

    out = tf.add(tf.matmul(inpt, weights), biases)

    # random_idx is list (feat_dim-1) is int
    random_idx = [random_idx, feat_dim-1]
    full_dim = [(in_feat_dim, feat_dim), (feat_dim,)]

    return out, random_idx, full_dim

def rss(inpt, feat_dim, activation, keep_dim, num_layers, act_dim,
        add_noise=False, keep_noise=0, 
        noise_std=1.0, base_name='feat_v1', transfer_name=''):
    
    outs = [inpt]
    random_idx = []
    full_dims = []

    activation = activ(activation)

    for n in range(num_layers):
        inter_out = 0.0
        for n_sub in range(n+1):
            layer_name = '_{}_{}'.format((n+1), (n_sub+1))

            divisor = (n + 1)

            out, r_idx, f_dim = single_rss(inpt=outs[n_sub], feat_dim=feat_dim, activation=activation, 
                                           keep_dim=keep_dim // divisor, add_noise=add_noise, 
                                           keep_noise=keep_noise // divisor, noise_std=noise_std,
                                           layer_name=layer_name, base_name=base_name, 
                                           transfer_name=transfer_name)

            inter_out = tf.add(inter_out, out)
            random_idx += r_idx
            full_dims += f_dim

        # i cannot believe that i did this mistake
        # those were all linear maps without any activation
        inter_out = activation(inter_out)
        
        outs.append(inter_out)
        print(outs)

    print('building policy and value functions...')

    for n in range(2):
        inter_out = 0.0
        divisor = (num_layers + 1)

        tmp_dim = 1
        l_name = 'val'
        if n == 0:
            tmp_dim = act_dim
            l_name = 'pol'

        for n_sub in range(num_layers + 1):
            layer_name = '_{}_{}'.format(l_name, (n_sub+1))

            out, r_idx, f_dim = single_rss(inpt=outs[n_sub], feat_dim=tmp_dim, activation=activation, 
                                           keep_dim=keep_dim // divisor, add_noise=add_noise, 
                                           keep_noise=keep_noise // divisor, noise_std=noise_std,
                                           layer_name=layer_name, base_name=base_name, 
                                           transfer_name=transfer_name)
            inter_out = tf.add(inter_out, out)
            random_idx += r_idx
            full_dims += f_dim

        outs.append(inter_out)
        print(outs)

    
    print('''

        outs for {}

        '''.format(base_name))

    for o in outs:
        print(o, o.shape)

    r_outs = outs[-2:]

    return r_outs, random_idx, full_dims

# i want to make this (sparse and skipped)
# we will train this without adding any noise
def feat_rss_v0(out, feat_dim, activation, keep_dim, act_dim, 
                add_noise=False, keep_noise=0, num_layers=2, noise_std=1.0,
                transfer_load=False, transfer_dim=None, base_name='feat_v1'):
    outs, train_random_idx, train_full_dim = rss(inpt=out, feat_dim=feat_dim, activation=activation, keep_dim=keep_dim, act_dim=act_dim,
                                                 add_noise=add_noise, keep_noise=keep_noise, num_layers=num_layers, noise_std=noise_std, 
                                                 base_name=base_name, transfer_name='')

    random_idx = {'train_random_idx': train_random_idx}
    full_dim = {'train_full_dim': train_full_dim}

    if transfer_load:
        outs, trans_random_idx, trans_full_dim = rss(inpt=out, feat_dim=feat_dim, activation=activation, keep_dim=transfer_dim, act_dim=act_dim,
                                                     add_noise=add_noise, keep_noise=keep_noise, num_layers=num_layers, noise_std=noise_std, 
                                                     base_name=base_name, transfer_name='_transfer')
        random_idx['trans_random_idx'] = trans_random_idx
        full_dim['trans_full_dim'] = trans_full_dim

    return outs, random_idx, full_dim

# we are making a lot of changes
def cr_fc_v0(out, ncat):
    # might not be necessary

    # out = fc(out, 512, activation='relu', batchnormalize=False, init_scale=np.sqrt(2))
    # out = fc(out, 512, activation='relu', batchnormalize=False, init_scale=np.sqrt(2))

    pdparam = fc(out, ncat, activation=None, batchnormalize=False, init_scale=np.sqrt(2))
    vpred = fc(out, 1, activation=None, batchnormalize=False, init_scale=1.0)[:, 0]
    return pdparam, vpred

def cr_fc_v1(out, ncat):
    # might not be necessary
    # be careful about this
    
    # to must also get results for this one for the future when we decide to use
    # jacobian penalty
    out = fc(out, 512, activation='relu', batchnormalize=False, init_scale=np.sqrt(2))
    out = fc(out, 512, activation='relu', batchnormalize=False, init_scale=np.sqrt(2))

    pdparam = fc(out, ncat, activation=None, batchnormalize=False, init_scale=np.sqrt(2))
    vpred = fc(out, 1, activation=None, batchnormalize=False, init_scale=1.0)[:, 0]
    return pdparam, vpred


def ls_c_v0(out, ncat, activation):
    hidsize = 512

    # actual policy
    out = fc(out, hidsize, activation=activation, batchnormalize=False, init_scale=np.sqrt(2))
    out = fc(out, hidsize, activation=activation, batchnormalize=False, init_scale=np.sqrt(2))
    
    pdparam = fc(out, ncat, activation=None, batchnormalize=False, init_scale=0.01)
    vpred = fc(out, 1, activation=None, batchnormalize=False, init_scale=1.0)[:, 0]
    return pdparam, vpred


def ls_c_hh(out, ncat, activation):
    hidsize = 512

    # might not be necessary
    if nentities:
        hidsize = nentities
    
    # actual policy
    # removing one of the outs for head-to-head comparison
    # out = fc(out, hidsize, activation=activation, batchnormalize=False, init_scale=np.sqrt(2))
    out = fc(out, hidsize, activation=activation, batchnormalize=False, init_scale=np.sqrt(2))
    
    pdparam = fc(out, ncat, activation=None, batchnormalize=False, init_scale=0.01)
    vpred = fc(out, 1, activation=None, batchnormalize=False, init_scale=1.0)[:, 0]
    return pdparam, vpred


def ls_c_v1(out, ncat, activation):
    feat_dim = 512
    hidsize = 512

    # might not be necessary
    if nentities:
        hidsize = nentities

    # perception output (feature embeddings)
    out = fc(out, feat_dim, activation='leaky_relu', batchnormalize=False, init_scale=np.sqrt(2))
    
    # actual policy
    out = fc(out, hidsize, activation='leaky_relu', batchnormalize=False, init_scale=np.sqrt(2))
    out = fc(out, hidsize, activation='leaky_relu', batchnormalize=False, init_scale=np.sqrt(2))
    
    pdparam = fc(out, ncat, activation=None, batchnormalize=False, init_scale=0.01)
    vpred = fc(out, 1, activation=None, batchnormalize=False, init_scale=1.0)[:, 0]
    return pdparam, vpred

def constfn(val):
    def f(_):
        return val
    return f

def random_agent_mean_std(env, nsteps=10000):
    ob = np.asarray(env.reset())
    num_envs = env.num_envs
    obs = [ob]
    for _ in range(nsteps // num_envs):
        acs = np.random.randint(low=0, high=env.action_space.n, size=num_envs)
        ob, _, done, _ = env.step(acs)
        obs.append(np.asarray(ob))
    
    obs = np.asarray(obs)

    t, N, H, W, f = obs.shape
    obs = np.reshape(obs, [t*N, H, W, f])
    
    print('utils.py, def random_agent_ob_mean_std, obs.shape: ', obs.shape)
    
    mean = np.mean(obs, 0).astype(np.float32)

    # this is how large-scale-curiosity has done it
    std = np.std(obs, 0).mean().astype(np.float32)
    return mean, std

def get_moments(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    count = arr.shape[0]
    return mean, std, count

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def kl(logits0, logits1):
    a0 = logits0 - tf.reduce_max(logits0, axis=-1, keepdims=True)
    a1 = logits1 - tf.reduce_max(logits1, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    ea1 = tf.exp(a1)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

def forward_backward_kl(logits0, logits1):
    return kl(logits0, logits1) + kl(logits1, logits0)


def append_coords(out):
    # append coordinates
    B, m, n, f = out.shape

    # out.shape gives static shapes, B is dynamic
    B = tf.shape(out)[0]

    # N = number of entities
    N = m * n

    # this concatenated coordinates makes it harder to swap
    # we must fix this at some point
    # concatenate coordinates        
    x = tf.linspace(-1.0, 1.0, n)
    y = tf.linspace(-1.0, 1.0, m)
    coord = tf.transpose(tf.meshgrid(x, y), [1, 2, 0])
    ## increase feature size by 2
    f += 2

    def concat(s):
        return tf.concat([s, coord], -1)
    
    out = tf.vectorized_map(concat, out)
    return out

def reshape_E(out):
    # append coordinates
    _, m, n, f = out.shape

    # out.shape gives static shapes, B is dynamic
    B = tf.shape(out)[0]

    # reshape to E: (batch_size, num_entities, f)
    out = tf.reshape(tf.transpose(out, [0, 3, 1, 2]), (B, f, m * n))
    out = tf.transpose(out, [0, 2, 1])
    return out

def get_nentities_per_state(input_shape, attention):
    # not implemented yet
    if attention:
        x, y = input_shape[0], input_shape[1]
        for n in nature_cnn_spec:
            x = np.floor((x - n['kernel_size']) / n['strides'][0]) + 1
            y = np.floor((y - n['kernel_size']) / n['strides'][1]) + 1
        return x * y
    return 0