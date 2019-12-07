import numpy as np
import tensorflow as tf
from baselines.common.distributions import make_pdtype
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

from attention.RelationalBlock import RelationalBlock

import utils

sess = tf.get_default_session

class Policy(object):
    def __init__(self, scope, ob_space, ac_space, ob_mean, ob_std, 
                 perception, feat_spec, policy_spec, 
                 activation, layernormalize, batchnormalize, 
                 add_noise, keep_noise, noise_std, transfer_load,
                 num_layers, keep_dim, transfer_dim,
                 vf_coef, coinrun):
        
        # warnings
        # i do not want to accidentally pass layernormalize and batchnormalize
        # for coinrun
        if layernormalize:
            print("Warning: policy is operating on top of layer-normed features.")
            raise NotImplementedError()

        if batchnormalize:
            print("Warning: policy is operating on top of batch-normed features.")
            raise NotImplementedError()

        self.transfer_load = transfer_load
        self.transfer_dim = transfer_dim

        self.num_layers = num_layers
        self.keep_dim = keep_dim

        self.coinrun = coinrun
        
        self.ob_mean = ob_mean
        self.ob_std = ob_std

        self.add_noise = add_noise
        self.keep_noise = keep_noise
        self.noise_std = noise_std

        self.layernormalize = layernormalize
        self.batchnormalize = batchnormalize
        
        self.vf_coef = vf_coef
        
        input_shape = ob_space.shape
        
        # perception module
        self.perception = perception

        # feature dimensions (HARD-CODED NOT GOOD)
        self.feat_dim = 512

        # policy module
        self.feat_spec = feat_spec
        self.policy_spec = policy_spec

        self.activation = activation

        with tf.variable_scope(scope):
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.ac_pdtype = make_pdtype(ac_space)

            # placeholders
            dtype = ob_space.dtype
            if dtype == np.int8:
                dtype = np.uint8
            print('policy.py, class Policy, def __init__, dtype: {}'.format(dtype))
            
            # taken from baselines.common.input import observation_input
            self.ph_ob = tf.to_float(tf.placeholder(dtype=ob_space.dtype,
                                                    shape=(None,) + ob_space.shape,
                                                    name='ob'))

            self.ph_ac = self.ac_pdtype.sample_placeholder([None], name='ac')
            self.pd = self.vpred = None
            self.scope = scope
            self.pdparamsize = self.ac_pdtype.param_shape()[0]

            with tf.variable_scope(self.scope + '_representation', reuse=False):
                self.unflattened_out = self.get_out(self.ph_ob, reuse=False)    
                out = utils.flatten(self.unflattened_out)
                print('policy.py, class Policy, def __init__, self.out.shape: {}'.format(out.shape))
                # we get features (feat_dim 512)
                self.features = self.get_features(out, reuse=False)

            pdparam, self.vpred = self.get_policy(self.features, reuse=False)
            self.pd = pd =self.ac_pdtype.pdfromflat(pdparam)
            self.a_samp = pd.sample()
            self.entropy = pd.entropy()
            self.nlp_samp = pd.neglogp(self.a_samp)
            self.logits = pdparam

            print('policy.py, class Policy, def __init__, pdparam.shape: {}, pdparam.dtype: {}'.format(pdparam.shape, pdparam.dtype))
            print('policy.py, class Policy, def __init__, self.vpred: {}'.format(self.vpred.shape))
            print('policy.py, class Policy, def __init__, self.a_samp: {}'.format(self.a_samp.shape))
            print('policy.py, class Policy, def __init__, self.entropy.shape: {}'.format(self.entropy.shape))
            print('policy.py, class Policy, def __init__, self.nlp_samp.shape: {}'.format(self.nlp_samp.shape))
            print('policy.py, class Policy, def __init__, self.logits.shape: {}'.format(self.logits.shape))

    def get_out(self, x, reuse):
        with tf.variable_scope(self.scope + '_out', reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            if self.perception == 'nature_cnn':
                x = utils.nature_cnn(x, activation=self.activation, batchnormalize=self.batchnormalize,
                                     init_scale=np.sqrt(2), coinrun=self.coinrun)
            else:
                raise NotImplementedError('only nature_cnn is allowed')

        # nature_cnn returns unflattened outputs
        return x

    def get_features(self, x, reuse):
        with tf.variable_scope(self.scope + '_features', reuse=reuse):
            if self.feat_spec == 'feat_v0':
                x = utils.feat_v0(x, feat_dim=self.feat_dim, activation=self.activation)
                self.random_idx = None

            elif self.feat_spec == 'feat_rss_v0':
                if self.policy_spec == 'full_sparse':
                    return x
                else:
                    raise NotImplementedError('only use full sparse...')

            else:
                raise NotImplementedError('not implemented method in get_features...')

        return x

    def get_policy(self, x, reuse):
        with tf.variable_scope(self.scope + '_policy', reuse=reuse):
            if self.policy_spec == 'cr_fc_v0':
                # we might need to ignore the extra 512 in the future
                # for entity-based policy
                x = utils.cr_fc_v0(x, self.pdparamsize, self.nentities_per_state)
            elif self.policy_spec == 'cr_fc_v1':
                # we might need to ignore the extra 512 in the future
                # for entity-based policy
                x = utils.cr_fc_v1(x, self.pdparamsize, self.nentities_per_state)
            elif self.policy_spec == 'ls_c_v0':
                x = utils.ls_c_v0(x, ncat=self.pdparamsize, activation=self.activation, nentities=self.nentities_per_state)
            elif self.policy_spec == 'ls_c_v1':
                x = utils.ls_c_v1(x, ncat=self.pdparamsize, activation=self.activation, nentities=self.nentities_per_state)
            elif self.policy_spec == 'ls_c_hh':
                x = utils.ls_c_hh(x, ncat=self.pdparamsize, activation=self.activation, nentities=self.nentities_per_state)
            elif self.policy_spec == 'full_sparse':
                print('''

                    adding random sparse noise: {}

                    '''.format(self.add_noise))

                # the problem with the sparsity is that it creates non-trainable paths because of the restrictions of the
                # information flow; our solution for this was to incorporate skipped connections. in the current, full-sparsity
                # tests, i am again restricting the flow. i need to fix this arbitrary restriction of the information flow.
                x, random_idx, full_dim = utils.feat_rss_v0(out=x, feat_dim=self.feat_dim, activation=self.activation, 
                                                            add_noise=self.add_noise, keep_dim=self.keep_dim, 
                                                            act_dim=self.ac_space.n,
                                                            keep_noise=self.keep_noise, noise_std=self.noise_std,
                                                            num_layers=self.num_layers,
                                                            transfer_load=self.transfer_load, transfer_dim=self.transfer_dim)

                # we will use those idx to mask the gradients of not-selected indices as well as 
                # inject some noise

                self.random_idx = random_idx['train_random_idx']
                self.full_dim = full_dim['train_full_dim']
                
                if self.transfer_load:
                    self.random_idx = random_idx['trans_random_idx']
                    self.full_dim = full_dim['trans_full_dim']

                self.random_idx_dict = random_idx

                return x[0], tf.squeeze(x[1])
            else:
                raise NotImplementedError('only some types policies are allowed')

        pdparam, vpred = x[0], x[1]
        return pdparam, vpred

    # return actions, vpreds and negative log likelihoods
    def step(self, ob):
        a, vpred, nlp = sess().run([self.a_samp, self.vpred, self.nlp_samp],
                                  feed_dict={self.ph_ob: ob})
        return a, vpred, nlp