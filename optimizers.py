import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from baselines.common import explained_variance
from baselines.common.running_mean_std import RunningMeanStd

import utils
import csv
from runner import Runner

sess = tf.get_default_session

class PPO(object):
    def __init__(self, scope, env, test_env, nenvs, save_dir, log_dir, 
                 policy, use_news, gamma, lam,
                 nepochs, nminibatches, nsteps, vf_coef,
                 ent_coef, max_grad_norm, 
                 normrew, cliprew, normadv,
                 for_visuals, transfer_load=False, load_path=None, freeze_weights=False):
        
        self.save_dir = save_dir
        self.log_dir = log_dir

        self.transfer_load = transfer_load
        self.freeze_weights = freeze_weights

        # save the random_idx of the random connections for the future
        # this is only to check that the same connections are established
        if policy.random_idx is not None:
            random_idx = np.asarray(policy.random_idx_dict['train_random_idx'])
            npz_path = os.path.join(self.save_dir, 'train_random_idx.npz')
            np.savez_compressed(npz_path,
                                random_idx=random_idx)

        with tf.variable_scope(scope):
            # ob_space, ac_space is from policy
            self.ob_space = policy.ob_space
            self.ac_space = policy.ac_space
            self.env = env
            self.test_env = test_env
            self.nenvs = nenvs

            self.policy = policy
            self.for_visuals = for_visuals

            # use_news
            self.use_news = use_news
            self.normrew = normrew
            self.cliprew = cliprew
            self.normadv = normadv

            # gamma and lambda
            self.gamma = gamma
            self.lam = lam
            self.max_grad_norm = max_grad_norm

            # update epochs and minibatches
            self.nepochs = nepochs
            self.nminibatches = nminibatches
            # nsteps = number of timesteps per rollout per environment
            self.nsteps = nsteps

            # placeholders
            self.ph_adv = tf.placeholder(tf.float32, [None])
            # ret = advs + vpreds, R = ph_ret
            self.ph_ret = tf.placeholder(tf.float32, [None])
            self.ph_oldnlp = tf.placeholder(tf.float32, [None])
            self.ph_oldvpred = tf.placeholder(tf.float32, [None])

            self.ph_lr = tf.placeholder(tf.float32, [])

            self.ph_cliprange = tf.placeholder(tf.float32, [])

            neglogpac = self.policy.pd.neglogp(self.policy.ph_ac)
            
            ## add to summary
            entropy = tf.reduce_mean(self.policy.pd.entropy())

            # clipped vpred, same as coinrun
            vpred = self.policy.vpred
            vpredclipped = self.ph_oldvpred + tf.clip_by_value(self.policy.vpred - self.ph_oldvpred, -self.ph_cliprange, self.ph_cliprange)
            vf_losses1 = tf.square(vpred - self.ph_ret)
            vf_losses2 = tf.square(vpredclipped - self.ph_ret)

            ## add to summary
            vf_loss = vf_coef * (0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)))

            ratio = tf.exp(self.ph_oldnlp - neglogpac)
            negadv = -self.ph_adv
            pg_losses1 = negadv * ratio
            pg_losses2 = negadv * tf.clip_by_value(ratio, 1.0 - self.ph_cliprange, 1.0 + self.ph_cliprange)
            pg_loss_surr = tf.maximum(pg_losses1, pg_losses2)
            
            ## add to summary
            pg_loss = tf.reduce_mean(pg_loss_surr)
            
            ent_loss = (-ent_coef) * entropy
            
            ## add to summary
            approxkl = 0.5 * tf.reduce_mean(tf.square(neglogpac - self.ph_oldnlp))
            
            ## add to summary
            clipfrac = tf.reduce_mean(tf.to_float(tf.abs(pg_losses2 - pg_loss_surr) > 1e-6))
            
            ## add to summary
            self.policy_loss = pg_loss + ent_loss + vf_loss

            # set summaries
            self.to_report = {'policy_loss': self.policy_loss,
                              'pg_loss': pg_loss,
                              'vf_loss': vf_loss,
                              'ent': entropy,
                              'approxkl': approxkl,
                              'clipfrac': clipfrac}

            if self.transfer_load:
                self._pre_load(load_path)

            # initialize various parameters
            self._init()

    def _gradient_summaries(self, gradsandvar):
        for gradient, variable in gradsandvar:
            if isinstance(gradient, ops.IndexedSlices):
                grad_values = gradient.values
            else:
                grad_values = gradient
            tf.summary.histogram(variable.name, variable)
            tf.summary.histogram(variable.name + './gradients', grad_values)
            tf.summary.histogram(variable.name + '/gradient_norms', 
                                 clip_ops.global_norm([grad_values]))

    def _init(self):
        self.loss_names, self._losses = zip(*list(self.to_report.items()))

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        print('__init trainable variables in the collection...')
        for p in params:
            print(p)

        # changed epsilon value
        trainer = tf.train.AdamOptimizer(learning_rate=self.ph_lr, epsilon=1e-5)
        
        gradsandvar = trainer.compute_gradients(self.policy_loss, params)

        grads, var = zip(*gradsandvar)

        # we only do this operation if our features network is not feat_v0
        if self.policy.feat_spec == 'feat_rss_v0':
            if self.policy.policy_spec == 'ls_c_v0':
                # this is a gradient hack to make rss work
                end_idx = -8
            elif self.policy.policy_spec == 'ls_c_hh':
                end_idx = -6
            elif self.policy.policy_spec == 'cr_fc_v0':
                end_idx = -4

            # i do not want to worry about 'full_sparsity' at the moment
            # full sparsity is a must though (if i want to use this for gradient predictions)
            elif self.policy.policy_spec == 'full_sparse':
                end_idx = 0
            else:
                raise NotImplementedError()
            
            # we changed this because of completely sparse training from + 0 to + 2
            sum_get = [(i+1) for i in range(self.policy.num_layers)]
            mult = np.sum(sum_get) + 2 * (sum_get[-1] + 1)
            start_idx = end_idx - (mult * 2)

        
            print('start_idx: {} and end_idx: {}'.format(start_idx, end_idx))
            for g in grads:
                print(g)

            print('''


                ''')

            for i, g in enumerate(grads[start_idx:]):
                print('g: {}'.format(g))
                sparse_idx = self.policy.random_idx[i]
                full_dim = self.policy.full_dim[i]
                mult_conts = np.zeros(full_dim, dtype=np.float32)

                # this is the case for weights
                if isinstance(sparse_idx, list):
                    # we must separate (row, col) coords
                    sparse_idx = np.asarray(sparse_idx)
                    row_idx = sparse_idx[:, 0]
                    col_idx = sparse_idx[:, 1] 
                    mult_conts[row_idx, col_idx] = 1.0
                    if i == 6:
                        print('MANUAL TEST SAVING...')
                        np.savetxt('./mult_conts.csv', mult_conts)
                        np.savetxt('./r_idx.csv', sparse_idx)

                elif isinstance(sparse_idx, int):
                    mult_conts[:] = 1.0

                else:
                    raise TypeError('sparse_idx have not specified type')

                g = tf.multiply(g, tf.convert_to_tensor(mult_conts))
            
        if self.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

        gradsandvar = list(zip(grads, var))

        # add gradient summaries
        # self._gradient_summaries(gradsandvar)

        self._train = trainer.apply_gradients(gradsandvar)

        ## initialize variables
        sess().run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

        ## runner
        self.runner = Runner(env=self.env, test_env=self.test_env, nenvs=self.nenvs, policy=self.policy, 
                             nsteps=self.nsteps, cliprew=self.cliprew)

        self.buf_advs = np.zeros((self.nenvs, self.nsteps), np.float32)
        self.buf_rets = np.zeros((self.nenvs, self.nsteps), np.float32)

        # set saver
        self.saver = tf.train.Saver(max_to_keep=None)
        
        if self.transfer_load:
            self.saver = tf.train.Saver(var_list=self.vars_dict, max_to_keep=None)
        # self.summary_op = tf.summary.merge_all()
        # self.summary_writer = tf.summary.FileWriter(self.log_dir, sess().graph)

        # reward normalization
        if self.normrew:
            self.rff = utils.RewardForwardFilter(self.gamma)
            self.rff_rms = RunningMeanStd()

    def calculate_advantages(self, rews, use_news, gamma, lam):
        nsteps = self.nsteps
        lastgaelam = 0
        for t in range(nsteps - 1, -1, -1):  # nsteps-2 ... 0
            nextnew = self.runner.buf_news[:, t + 1] if t + 1 < nsteps else self.runner.buf_new_last
            if not use_news:
                nextnew = 0
            nextvals = self.runner.buf_vpreds[:, t + 1] if t + 1 < nsteps else self.runner.buf_vpred_last
            nextnotnew = 1 - nextnew
            delta = rews[:, t] + gamma * nextvals * nextnotnew - self.runner.buf_vpreds[:, t]
            self.buf_advs[:, t] = lastgaelam = delta + gamma * lam * nextnotnew * lastgaelam
        self.buf_rets[:] = self.buf_advs + self.runner.buf_vpreds

    def update(self, lr, cliprange):
        # fill rollout buffers
        self.runner.rollout()

        ## TODO: normalized rewards
        # coinrun does NOT normalize its rewards
        if self.normrew:
            rffs = np.array([self.rff.update(rew) for rew in self.runner.buf_rews.T])
            # print('optimizers.py, class PPO, def update, rffs.shape: {}'.format(rffs.shape))
            rffs_mean, rffs_std, rffs_count = utils.get_moments(rffs.ravel())
            self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
            rews = self.runner.buf_rews / np.sqrt(self.rff_rms.var)
        else:
            rews = np.copy(self.runner.buf_rews)

        self.calculate_advantages(rews=rews, use_news=self.use_news, 
                                  gamma=self.gamma, lam=self.lam)

        # this is a little bit different than the original coinrun implementation
        # they only normalize advantages using batch mean and std instead of
        # entire data & we add 1e-7 instead of 1e-8
        if self.normadv:
            mean, std = np.mean(self.buf_advs), np.std(self.buf_advs)
            self.buf_advs = (self.buf_advs - mean) / (std + 1e-7)

        ## this only works for non-recurrent version
        nbatch = self.nenvs * self.nsteps
        nbatch_train = nbatch // self.nminibatches
        
        # BUG FIXED: np.arange(nbatch_train) to np.arange(nbatch)
        # might be the cause of unstable training performance
        train_idx = np.arange(nbatch)

        # another thing is that they completely shuffle the experiences
        # flatten axes 0 and 1 (we do not swap)
        def f01(x):
            sh = x.shape
            return x.reshape(sh[0] * sh[1], *sh[2:])

        flattened_obs = f01(self.runner.buf_obs)

        ph_buf = [(self.policy.ph_ob, flattened_obs),
                  (self.policy.ph_ac, f01(self.runner.buf_acs)),
                  (self.ph_oldvpred, f01(self.runner.buf_vpreds)),
                  (self.ph_oldnlp, f01(self.runner.buf_nlps)),
                  (self.ph_ret, f01(self.buf_rets)),
                  (self.ph_adv, f01(self.buf_advs))]

        # when we begin to work with curiosity, we might need make a couple of
        # changes to this training strategy

        mblossvals = []

        for e in range(self.nepochs):
            np.random.shuffle(train_idx)

            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbidx = train_idx[start:end]
                fd = {ph: buf[mbidx] for (ph, buf) in ph_buf}
                fd.update({self.ph_lr: lr,
                           self.ph_cliprange: cliprange})
                
                mblossvals.append(sess().run(self._losses + (self._train,), feed_dict=fd)[:-1])
                
        mblossvals = [mblossvals[0]]

        info = dict(
            advmean=self.buf_advs.mean(),
            advstd=self.buf_advs.std(),
            retmean=self.buf_rets.mean(),
            retstd=self.buf_rets.std(),
            vpredmean=self.runner.buf_vpreds.mean(),
            vpredstd=self.runner.buf_vpreds.std(),
            ev=explained_variance(self.runner.buf_vpreds.ravel(), self.buf_rets.ravel()),
            rew_mean=np.mean(self.runner.buf_rews),
        )

        info.update(zip(['opt_' + ln for ln in self.loss_names], np.mean([mblossvals[0]], axis=0)))
        
        return info

    def evaluate(self, nlevels, save_video):
        return self.runner.evaluate(nlevels, save_video)
        
    def save(self, curr_iter, cliprange):
        save_path = os.path.join(self.save_dir, 'model')
        self.saver.save(sess(), save_path, global_step=curr_iter)

        def f01(x):
            sh = x.shape
            return x.reshape(sh[0] * sh[1], *sh[2:])

        if self.for_visuals:
            obs = f01(self.runner.buf_obs)            
            acs = f01(self.runner.buf_acs)
            nlps = f01(self.runner.buf_nlps)
            advs = f01(self.buf_advs)
            oldvpreds = f01(self.runner.buf_vpreds)
            rets = f01(self.buf_rets)

            npz_path = os.path.join(self.save_dir, 'extra-{}.npz'.format(curr_iter))
            np.savez_compressed(npz_path,
                                obs=obs,
                                acs=acs,
                                nlps=nlps,
                                advs=advs,
                                oldvpreds=oldvpreds,
                                rets=rets,
                                cliprange=cliprange)

    def _pre_load(self, load_path):
        print('''

            PRE LOADING...

            ''')
        
        trainable_variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)

        self.vars_dict = {}
        for var_ckpt in tf.train.list_variables(load_path):
            # remove the variables in the ckpt from trainable variables
            for t in trainable_variables:
                if var_ckpt[0] == t.op.name:
                    self.vars_dict[var_ckpt[0]] = t
                    if self.freeze_weights:
                       trainable_variables.remove(t)

    # load buffers
    def load_ph_bufs(self, bufs):
        self.load_fd = {self.policy.ph_ob: bufs['obs'],
                        self.policy.ph_ac: bufs['acs'],
                        self.ph_oldvpred: bufs['oldvpreds'],
                        self.ph_oldnlp: bufs['nlps'],
                        self.ph_ret: bufs['rets'],
                        self.ph_adv: bufs['advs'],
                        self.ph_lr: 0.0,
                        self.ph_cliprange: bufs['cliprange']}

    # write everything to here, because ckpt contains adam variables too
    def variable_assignment(self, load_ckpt):
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # instead of restore, we use this to quickly update our variables
    def _assign_op(self, v_dict, dir_dict, alpha, beta):
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        assign_op = []
        for k in v_dict.keys():
            for t in trainable_variables:
                if k == t.op.name:
                    assign_op.append(tf.assign(t, v_dict[t.op.name] + alpha * dir_dict[0][t.op.name] + beta * dir_dict[1][t.op.name]))
        return sess().run(assign_op)

    def get_loss(self, v_dict, dir_dict, alpha, beta):
        self._assign_op(v_dict, dir_dict, alpha, beta)
        return sess().run(self.policy_loss, feed_dict=self.load_fd)

    def re_run_loss(self, v_dict, dir_dict, alpha, beta, bufs):
        self._assign_op(v_dict, dir_dict, alpha, beta)
        self.runner.rollout()

        ## TODO: normalized rewards
        # coinrun does NOT normalize its rewards
        if self.normrew:
            rffs = np.array([self.rff.update(rew) for rew in self.runner.buf_rews.T])
            # print('optimizers.py, class PPO, def update, rffs.shape: {}'.format(rffs.shape))
            rffs_mean, rffs_std, rffs_count = utils.get_moments(rffs.ravel())
            self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
            rews = self.runner.buf_rews / np.sqrt(self.rff_rms.var)
        else:
            rews = np.copy(self.runner.buf_rews)

        self.calculate_advantages(rews=rews, use_news=self.use_news, 
                                  gamma=self.gamma, lam=self.lam)

        # this is a little bit different than the original coinrun implementation
        # they only normalize advantages using batch mean and std instead of
        # entire data & we add 1e-7 instead of 1e-8
        if self.normadv:
            mean, std = np.mean(self.buf_advs), np.std(self.buf_advs)
            self.buf_advs = (self.buf_advs - mean) / (std + 1e-7)

        def f01(x):
            sh = x.shape
            return x.reshape(sh[0] * sh[1], *sh[2:])
        flattened_obs = f01(self.runner.buf_obs)

        ph_buf = {self.policy.ph_ob: flattened_obs,
                  self.policy.ph_ac: bufs['acs'],
                  self.ph_oldvpred: f01(self.runner.buf_vpreds),
                  self.ph_oldnlp: f01(self.runner.buf_nlps),
                  self.ph_ret: f01(self.buf_rets),
                  self.ph_adv: f01(self.buf_advs),
                  self.ph_lr: 0.0,
                  self.ph_cliprange: bufs['cliprange']}

        return sess().run(self.policy_loss, feed_dict=ph_buf)
    
    def load(self, load_path):

        print('''

            load variables

            ''')
        for variable in tf.train.list_variables(load_path):
            print(variable[0])

        print('''

            global variables

            ''')
        for variable in tf.global_variables():
            print(variable.name)

        self.saver.restore(sess(), load_path)
        print('loaded already trained model from {}'.format(load_path))