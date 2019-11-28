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
                 attention, reduce_max, dropout_attention, recurrent,
                 jacobian_loss, nparticles,
                 entity_loss, tol_entity_loss, nentities_per_state, nentities_per_batch,
                 entity_randomness,
                 num_repeat, num_replace_ratio,
                 add_noise, keep_noise, noise_std, transfer_load,
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

        if recurrent:
            # recurrent layer will be implemented LATER
            print("Warning: policy is operating with recurrent model.")
            raise NotImplementedError()

        # attention will be used as a baseline for entity-centric random swaps
        if attention:
            print('''

                Warning: policy is operating with entity-centric attention...

                ''')

        # implementing jacobian_loss
        if jacobian_loss:
            print('''

                WARNING: policy is operating with jacobian penalty {}...
                
                '''.format(jacobian_loss))
            
        if entity_loss:
            print('''

                WARNING: policy is operating with entity-centric penalty...
                
                ''')
        self.transfer_load = transfer_load

        self.coinrun = coinrun
        self.num_repeat = int(num_repeat)
        self.num_replace_ratio = int(num_replace_ratio)
        self.entity_randomness = entity_randomness
        
        self.ob_mean = ob_mean
        self.ob_std = ob_std

        self.add_noise = add_noise
        self.keep_noise = keep_noise
        self.noise_std = noise_std

        self.layernormalize = layernormalize
        self.batchnormalize = batchnormalize
        self.attention = attention
        self.reduce_max = reduce_max
        print('''

            we are using reduce_max {} for features
            
            '''.format(reduce_max))

        self.dropout_attention = dropout_attention
        self.recurrent = recurrent

        self.jacobian_loss = jacobian_loss
        self.vf_coef = vf_coef
        self.entity_loss = entity_loss
        self.nentities_per_batch = nentities_per_batch

        ## CHANGE THIS TO PLACEHOLDER
        self.tol_entity_loss = tol_entity_loss

        input_shape = ob_space.shape
        self.nentities_per_state = int(nentities_per_state)

        print('policy.py, class Policy, def __init__, self.nentities_per_state: {}'.format(self.nentities_per_state))
        
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
                
                # attention is part of representation
                if self.attention:
                    self.E = self.get_coords_reshape(self.unflattened_out, reuse=False)
                    E_features, self.att_weights = self.get_relations(self.E, reuse=False)
                    if self.reduce_max:
                        self.features = tf.reduce_max(E_features, axis=-2, keepdims=False)
                    else:
                        out = utils.flatten(E_features)
                        self.features = self.get_features(out, reuse=False)
                    print('policy.py, class Policy, def __init__, self.features.shape after reduce_max: {}'.format(self.features.shape))
                    print('policy.py, class Policy, def __init__, self.E.shape: {}'.format(self.E.shape))
                    print('policy.py, class Policy, def __init__, self.att_weights.shape: {}'.format(self.att_weights.shape))
                
                else:
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

            '''
            # major change to jacobian_loss
            if jacobian_loss:
                self.ph_feats = tf.placeholder(dtype=self.features.dtype, shape=(None, self.feat_dim), name='ph_feats')
                self.ph_logits = tf.placeholder(dtype=pdparam.dtype, shape=(None, ac_space.n), name='ph_logits')
                self.ph_vpreds = tf.placeholder(dtype=self.vpred.dtype, shape=(None,), name='ph_vpreds')

                # HARD-CODED feat_dim: not a good practice
                self.ph_jacobians = tf.placeholder(dtype=pdparam.dtype, shape=(None, ac_space.n, self.feat_dim), name='ph_jacobians')
                self.ph_tol_jacobian_loss = tf.placeholder(tf.float32, [])

                self.jacobians = batch_jacobian(tf.nn.softmax(self.logits), self.features)

                print('policy.py, class Policy, def __init__, jacobian.shape: {}'.format(self.jacobians.shape))

                self.pol_jacobian_loss, self.rep_jacobian_loss = self.get_jacobian_loss()
            '''

            # jacobian_loss 1 is the full input
            if jacobian_loss == 1:
                raise NotImplementedError()

            # jacobian_loss 2 works only on the features
            elif jacobian_loss == 2:
                # remember that we are using unnormalized probabilities
                self.jacobians = batch_jacobian(tf.log(tf.nn.softmax(self.logits) + 1e-8), self.features)
                # do NOT forget to change optimizers.py
                # self.ph_eps = tf.placeholder(dtype=self.features.dtype, shape=(None, self.feat_dim), name='ph_eps')
                self.eps = tf.random.truncated_normal(stddev=1.0, shape=(nparticles, self.feat_dim))
                self.pol_jacobian_loss = self.get_jacobian_loss()

            if entity_loss:
                # call entity related things here
                f = self.E.shape[-1]

                # we only feed N = nentities_per_state
                # f is the E
                self.ph_E = tf.placeholder(dtype=self.E.dtype, shape=(None, self.nentities_per_state, f), name='ph_E')

                self.rep_entity_loss = self.get_entity_loss()

    def get_coords_reshape(self, x, reuse):
        with tf.variable_scope(self.scope + '_coords_reshape', reuse=reuse):
            # append coordinates
            x = utils.append_coords(x)

            # reshape to (batch_size, nentities_per_state, f)
            x = utils.reshape_E(x)
            if self.dropout_attention:
                print('''

                    dropout on attention

                    ''')
                
                num_replace = self.nentities_per_state // 2
                mask_zeros = tf.zeros((self.nentities_per_state - num_replace, 1), dtype=tf.float32)
                mask_ones = tf.ones((num_replace, 1), dtype=tf.float32)
                mask = tf.concat([mask_zeros, mask_ones], axis=0)
                mask = tf.concat([tf.random.shuffle(mask) for _ in range(B)], axis=0)

                self.nentities_per_state = self.nentities_per_state - num_replace
                x = tf.boolean_mask(x, mask)
        return x

    def get_relations(self, x, reuse):
        with tf.variable_scope(self.scope + '_relations', reuse=reuse):
            _, N, f = x.shape
            
            # we reduced the number of relational blocks to 1
            # we reduced the number of heads to 2 
            # (super simplified, but time constraints)
            nrel_block = 1
            for _ in range(nrel_block):
                # we might play with RelationalBlock's parameters
                x, att_weights = RelationalBlock(d_features=f, d_embed=2*f, num_heads=2, num_entities=self.nentities_per_state)(x)
        return x, att_weights

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
                # this is for RSS part
                self.keep_dim = 30
                print('''

                    adding random sparse noise: {}

                    '''.format(self.add_noise))

                x, random_idx, full_dim = utils.feat_rss_v0(out=x, feat_dim=self.feat_dim, activation=self.activation, add_noise=self.add_noise, keep_dim=self.keep_dim,
                                                            keep_noise=self.keep_noise, noise_std=self.noise_std,
                                                            transfer_load=self.transfer_load)
                
                # we will use those idx to mask the gradients of not-selected indices as well as 
                # inject some noise
                self.random_idx = random_idx
                self.full_dim = full_dim
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

            else:
                raise NotImplementedError('only cr_fc_v0 or ls_c_v0 policies are allowed')
        pdparam, vpred = x[0], x[1]
        return pdparam, vpred

    # return actions, vpreds and negative log likelihoods
    def step(self, ob):
        a, vpred, nlp = sess().run([self.a_samp, self.vpred, self.nlp_samp],
                                  feed_dict={self.ph_ob: ob})
        return a, vpred, nlp

    # get logits and jacobians
    def get_feats_vpreds_logits_jacobians(self, ob):
        # in the first experiments, i used nn.softmax to compute jacobians instead of logits
        # i will use logits for now, possible change in the future
        feats, vpreds, logits, jacobians = sess().run([self.features, self.vpred, 
                                                       self.logits, self.jacobians],
                                                       feed_dict={self.ph_ob: ob})
        return feats, vpreds, logits, jacobians

    # get logits and jacobians
    def get_entities_weights(self, ob):
        # in the first experiments, i used nn.softmax to compute jacobians instead of logits
        # i will use logits for now, possible change in the future
        entities, weights = sess().run([self.E, self.att_weights],
                                       feed_dict={self.ph_ob: ob})
        return entities, weights
    def get_entity_loss(self):
        # split features and coordinates
        # hardcoded for now
        E_feat, E_coord = tf.split(self.E, [64, 2], axis=-1)

        ph_E_feat, ph_E_coord = tf.split(self.ph_E, [64, 2], axis=-1)

        # number of replace repeat (how many times we will replace the same batch)
        
        '''
        # LOOK AT THOSE
        # num_repeat is a hyperparameter, 4 for coinrun, 8 for mario
        if self.coinrun:
            num_repeat = 4
        else:
            num_repeat = 8
        print('num_repeat for get_entity_loss: {}'.format(num_repeat))

        # this is 8 (for coinrun with nature cnn), 7 (for mario with nature cnn)
        if self.coinrun:
            num_replace = self.nentities_per_state // 2
        else:
            num_replace = self.nentities_per_state // 7
        print('num_replace for get_entity_loss: {}'.format(num_replace))
        '''
        
        num_replace = self.nentities_per_state // self.num_replace_ratio

        print('num_repeat for get_entity_loss: {}'.format(self.num_repeat))
        print('num_replace for get_entity_loss: {}'.format(num_replace))

        mask_zeros = tf.zeros((self.nentities_per_state - num_replace, 1), dtype=tf.float32)
        mask_ones = tf.ones((num_replace, 1), dtype=tf.float32)
        mask = tf.concat([mask_zeros, mask_ones], axis=0)
        batch_loss = 0.0

        if self.entity_randomness == 'v1':
            ph_E_feat = tf.random.shuffle(ph_E_feat, name='shuffler')

        # i should vectorize this operation
        for i in range(self.num_repeat):
            # shuffle the mask
            mask = tf.random.shuffle(mask)

            # get the reverse mask for E_feat
            r_mask = 1.0 - mask

            # calculate the modified E_feat (we swap entities randomly from an entity list)
            modifE_feat = tf.multiply(E_feat, r_mask) + tf.multiply(ph_E_feat, mask)
            self.modifE_feat = modifE_feat

            # modifE_feat = tf.boolean_mask(E_feat, r_mask, axis=1) + tf.expand_dims(tf.boolean_mask(self.ph_E, mask, axis=1), axis=0)

            # concat coordinates
            modifE = tf.concat([modifE_feat, E_coord], axis=-1)
            self.modifE = modifE

            with tf.variable_scope(self.scope + '_representation', reuse=True):
                # get modified relations
                # we migth need to change this for swapping relations too
                modif_Efeat, _ = self.get_relations(modifE, reuse=True)
                
                # get modified features
                if self.reduce_max:
                    modif_feat = tf.reduce_max(modif_Efeat, axis=-2, keepdims=False)
                else:
                    out = utils.flatten(modif_Efeat)
                    modif_feat = self.get_features(out, reuse=True)

                self.modif_feat = modif_feat

            # get modified logits
            modif_logits, _ = self.get_policy(modif_feat, reuse=True)

            fb_kl = utils.forward_backward_kl(self.logits, modif_logits)
            penalty_ratio = tf.math.maximum(1.0 - fb_kl / self.tol_entity_loss, 0.0)
            self.penalty_ratio = penalty_ratio

            '''
            ### this part should be changed too, try to bring the final features closer together
            norm_E = tf.nn.l2_normalize(self.E, axis=-1)
            norm_modifE = tf.nn.l2_normalize(modifE, axis=-1)

            batch_loss += penalty_ratio * tf.reduce_sum(tf.multiply(norm_E, tf.stop_gradient(norm_modifE)), axis=-1)
            '''

            # either l2-loss between final features
            if self.entity_loss == 1:
                batch_loss += tf.stop_gradient(penalty_ratio) * tf.reduce_sum(tf.square(tf.stop_gradient(modif_feat) - self.features), axis=-1)

            # or cosine distance
            elif self.entity_loss == 2:
                norm_feat = tf.nn.l2_normalize(self.features, axis=-1)
                norm_modif_feat = tf.nn.l2_normalize(modif_feat, axis=-1)
                batch_loss += tf.stop_gradient(penalty_ratio) * (-1.0) * tf.reduce_sum(tf.multiply(tf.stop_gradient(norm_modif_feat), norm_feat), axis=-1)
            # others are not implemented
            else:
                raise NotImplementedError()

        self.entity_stats = {'entity_pr_mean': tf.reduce_mean(penalty_ratio),
                             'entity_kl_mean': tf.reduce_mean(fb_kl)}

        return batch_loss / self.num_repeat

    def get_jacobian_loss(self):
        neta_op = tf.matmul(self.jacobians, self.eps, transpose_b=True) 
        neta_01 = tf.nn.softmax(neta_op, axis=-1)

        # neta_op = tf.divide(matmul_op, tf.expand_dims(self.logits, -1)) + 1.0
        exp_neta_op = tf.reduce_mean((neta_01) * tf.log(neta_01 + 1.0), axis=-1)

        self.jac_stats = {'jacobian': tf.reduce_mean(self.jacobians),
                          'neta': tf.reduce_mean(neta_op),
                          'min_neta': tf.reduce_min(neta_op),
                          'neta_01': tf.reduce_mean(neta_01),
                          'min_neta_01': tf.reduce_min(neta_01),
                          'exp_neta_op': tf.reduce_mean(exp_neta_op)}
        return tf.reduce_sum(tf.multiply(tf.nn.softmax(self.logits), exp_neta_op), axis=-1)