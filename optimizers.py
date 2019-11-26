import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from baselines.common import explained_variance
from baselines.common.running_mean_std import RunningMeanStd
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

import utils
import csv
from runner import Runner

import time

sess = tf.get_default_session

class PPO(object):
    def __init__(self, scope, env, nenvs, save_dir, log_dir, policy,
                 use_news, recurrent, gamma, lam,
                 nepochs, nminibatches, nsteps, vf_coef,
                 ent_coef, max_grad_norm,
                 normrew, cliprew, normadv,
                 jacobian_loss, nparticles,
                 entity_loss, nentities_per_batch, entity_randomness,
                 num_traj_rep, list_grads, cap_buf, context_dim,
                 max_keep_prob, max_history, max_get, update_freq,
                 predictor):
        
        # for now we do not have recurrent implementation
        if recurrent:
            raise NotImplementedError()
        
        self.save_dir = save_dir
        self.log_dir = log_dir

        # do not forget to add this as a hyperparameter
        self.num_traj_rep = num_traj_rep
        
        # we create a replay buffer for gradients and contexts
        # num_grads is hardcoded for coinruns
        self.list_grads = list_grads
        self.num_grads = np.sum([np.prod(l) for l in list_grads])
        # print('num_grads: {}'.format(self.num_grads))

        self.cap_buf = cap_buf
        self.update_freq = update_freq

        # part1, part2, full
        self.buf_grads = [[], [], []]
        self.buf_contexts = [[], [], []]

        # self.buf_grads = np.zeros([cap_buf, num_grads], dtype=np.float32)
        # self.buf_contexts = np.zeros([cap_buf, context_dim], dtype=np.float32)
        self.context_dim = context_dim
        self.max_keep_prob = max_keep_prob # 0.6
        self.max_history = max_history # 3
        self.max_get = max_get 
        with tf.variable_scope(scope):
            ###
            # \SENTETIK BEGIN
            ###
            # add contextor, a random LSTM to project the inputs

            # add predictor, a trainable MLP to predict the gradients
            self.predictor = predictor

            self.ph_temp = tf.placeholder(tf.float32, [])
            self.ph_grad_list = []
            for lg in list_grads:
                try:
                    # weights
                    self.ph_grad_list.append(tf.placeholder(tf.float32, [lg[0], lg[1]]))
                except:
                    # biases
                    self.ph_grad_list.append(tf.placeholder(tf.float32, [lg[0]]))

            ###
            # \SENTETIK END
            ###

            # ob_space, ac_space is from policy
            self.ob_space = policy.ob_space
            self.ac_space = policy.ac_space
            self.env = env
            self.nenvs = nenvs

            self.policy = policy
            
            # recurrent
            self.recurrent = recurrent

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

            ## EXTRA LOSSES
            self.jacobian_loss = jacobian_loss
            self.ph_beta_jacobian_loss = tf.placeholder(tf.float32, [])

            # number of random variables to approximate expectation
            self.nparticles = nparticles 
            
            self.entity_loss = entity_loss
            self.ph_beta_entity_loss = tf.placeholder(tf.float32, [])
            self.nentities_per_batch = nentities_per_batch
            self.entity_randomness = entity_randomness
            print('''

                entity_randomness: {}

                '''.format(entity_randomness))

            self.batch_size = 1024
            # placeholders
            # ret = advs + vpreds, R = ph_ret
            self.ph_ret = tf.placeholder(tf.float32, [self.batch_size])
            # no need for ph_rews
            # self.ph_rews = tf.placeholder(tf.float32, [None])
            self.ph_oldnlp = tf.placeholder(tf.float32, [self.batch_size])
            self.ph_oldvpred = tf.placeholder(tf.float32, [self.batch_size])

            self.ph_lr = tf.placeholder(tf.float32, [])
            # tf.summary.scalar('lr', self.ph_lr)

            self.ph_cliprange = tf.placeholder(tf.float32, [])
            # tf.summary.scalar('cliprange', self.ph_cliprange)

            neglogpac = self.policy.pd.neglogp(self.policy.ph_ac)
            
            # clipped vpred, same as coinrun
            vpred = self.policy.vpred
            vpredclipped = self.ph_oldvpred + tf.clip_by_value(self.policy.vpred - self.ph_oldvpred, -self.ph_cliprange, self.ph_cliprange)
            vf_losses1 = tf.square(vpred - self.ph_ret)
            vf_losses2 = tf.square(vpredclipped - self.ph_ret)

            ## add to summary
            # i want to try the gradient shapes without reduce_mean
            
            # vf_loss = vf_coef * (0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)))
            vf_loss = vf_coef * (0.5 * tf.maximum(vf_losses1, vf_losses2))
            
            # tf.summary.scalar('vf_loss', vf_loss)

            ratio = tf.exp(self.ph_oldnlp - neglogpac)
            negadv = -self.policy.ph_adv
            pg_losses1 = negadv * ratio
            pg_losses2 = negadv * tf.clip_by_value(ratio, 1.0 - self.ph_cliprange, 1.0 + self.ph_cliprange)
            pg_loss_surr = tf.maximum(pg_losses1, pg_losses2)
            
            ## add to summary
            # pg_loss = tf.reduce_mean(pg_loss_surr)
            # pg_loss should be [None,]
            pg_loss = pg_loss_surr
            self.pg_loss = pg_loss
            # tf.summary.scalar('pg_loss', pg_loss)
            
            ## add to summary
            # entropy = tf.reduce_mean(self.policy.pd.entropy())
            entropy = self.policy.pd.entropy()
            ent_loss = (-ent_coef) * entropy
            self.ent_loss = ent_loss

            # tf.summary.scalar('ent_loss', ent_loss)
            
            ## add to summary
            approxkl = 0.5 * tf.reduce_mean(tf.square(neglogpac - self.ph_oldnlp))
            # tf.summary.scalar('approxkl', approxkl)
            
            ## add to summary
            clipfrac = tf.reduce_mean(tf.to_float(tf.abs(pg_losses2 - pg_loss_surr) > 1e-6))
            # tf.summary.scalar('clipfrac', clipfrac)
            
            ## add to summary
            self.policy_loss = pg_loss + ent_loss + vf_loss
            self.rep_loss = tf.reduce_mean(self.policy_loss)

            # self.total_loss = self.policy_loss
            # tf.summary.scalar('total_loss', self.policy_loss)

            # set summaries
            self.to_report = {'policy_loss': tf.reduce_mean(self.policy_loss),
                              'pg_loss': tf.reduce_mean(pg_loss),
                              'vf_loss': tf.reduce_mean(vf_loss),
                              'ent': tf.reduce_mean(entropy),
                              'approxkl': approxkl,
                              'clipfrac': clipfrac}

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

        policy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.policy.scope + '/' + self.policy.scope + '_policy')
        
        # changed epsilon value
        policy_trainer = tf.train.AdamOptimizer(learning_rate=self.ph_lr, epsilon=1e-5)
        
        # gradsandvar = policy_trainer.compute_gradients(self.policy_loss, all_params)
        # we should flatten what's inside the list, concat and then use this
        # grads, var = zip(*gradsandvar)

        print('setting up policy_gradients...')
        
        
        self.policy_grads = tf.vectorized_map(lambda s: tf.gradients(s, policy_params), self.policy_loss)

        print('''

            we are done

            ''')
        
        '''
        self.policy_grads = grads[:-2]

        # check the returned shape of this operation
        # self.policy_grads = tf.concat([tf.reshape(g, tf.prod(g.shape)) for g in policy_grads])

        # another change will be about the policy smoothing
        for i, g in enumerate(grads[:-2]):
            g = self.ph_temp * g + (1 - self.ph_temp) * self.ph_grad_list[i]
        '''

        print('extra calcs...')
        # if we are going to predict the gradients, we should also add value
        # the real question in here is that what about CNN's weights?
        # this separation of representation and policy learning might not be necessary (or even a good idea here)
        
        grads = []
        for i, p_grad in enumerate(self.policy_grads):
            grads.append(self.ph_temp * tf.reduce_mean(p_grad, 0) + (1.0 - self.ph_temp) * self.predictor.re_grads[i])

        '''
        self.policy_grads_0 = policy_grads[0]
        self.policy_grads_1 = policy_grads[1]
        grads_0 = self.ph_temp * tf.reduce_mean(self.policy_grads_0, 0) + (1.0 - self.ph_temp) * self.predictor.grad_0
        grads_1 = self.ph_temp * tf.reduce_mean(self.policy_grads_1, 0) + (1.0 - self.ph_temp) * self.predictor.grad_1
        # grads_0 = tf.reduce_mean(self.policy_grads[0], 0)
        # grads_1 = tf.reduce_mean(self.policy_grads[1], 0)

        grads_2 = tf.reduce_mean(policy_grads[2], 0)
        grads_3 = tf.reduce_mean(policy_grads[3], 0)
        '''

        # self.policy_grads = [grads_0, grads_1, grads_2, grads_3]
        
        if self.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

        gradsandvar = list(zip(grads, policy_params))
        
        # add gradient summaries
        # self._gradient_summaries(gradsandvar)

        self._policy_train = policy_trainer.apply_gradients(gradsandvar)
        
        rep_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.policy.scope + '/' + self.policy.scope + '_representation')
        rep_trainer = tf.train.AdamOptimizer(learning_rate=self.ph_lr, epsilon=1e-5)
        
        rep_gradsandvar = rep_trainer.compute_gradients(self.rep_loss, rep_params)
        rep_grads, rep_var = zip(*rep_gradsandvar)

        if self.max_grad_norm is not None:
            rep_grads, _rep_grad_norm = tf.clip_by_global_norm(rep_grads, self.max_grad_norm)

        rep_gradsandvar = list(zip(rep_grads, rep_var))

        self._rep_train = rep_trainer.apply_gradients(rep_gradsandvar)

        ## initialize variables
        sess().run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

        ## runner
        self.runner = Runner(env=self.env, nenvs=self.nenvs, policy=self.policy, 
                             nsteps=self.nsteps, cliprew=self.cliprew)

        self.buf_advs = np.zeros((self.nenvs, self.nsteps), np.float32)
        self.buf_rets = np.zeros((self.nenvs, self.nsteps), np.float32)

        # set saver
        self.saver = tf.train.Saver(max_to_keep=None)
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

    def update(self, rep_train, curr_iter, lr, cliprange, beta_jacobian_loss, tol_jacobian_loss, beta_entity_loss):
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
        # print('nbatch_train: {}'.format(nbatch_train))

        # BUG FIXED: np.arange(nbatch_train) to np.arange(nbatch)
        # might be the cause of unstable training performance
        train_idx = np.arange(nbatch)

        # another thing is that they completely shuffle the experiences
        # flatten axes 0 and 1 (we do not swap)
        def f01(x):
            sh = x.shape
            return x.reshape(sh[0] * sh[1], *sh[2:])

        # should we use advantages too???
        flattened_obs = f01(self.runner.buf_obs)
        flattened_acs = f01(self.runner.buf_acs)
        flattened_advs = f01(self.buf_advs)
        flattened_vpreds = f01(self.runner.buf_vpreds)
        flattened_nlps = f01(self.runner.buf_nlps)
        flattened_rets = f01(self.buf_rets)

        ph_buf = [(self.policy.ph_ob, flattened_obs),
                  (self.policy.ph_ac, flattened_acs),
                  (self.ph_oldvpred, flattened_vpreds),
                  (self.ph_oldnlp, flattened_nlps),
                  (self.ph_ret, flattened_rets),
                  (self.policy.ph_adv, flattened_advs)]

        mblossvals = []

        for e in range(self.nepochs):
            # shuffle train_idx
            np.random.shuffle(train_idx)

            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbidx = train_idx[start:end]
                fd = {ph: buf[mbidx] for (ph, buf) in ph_buf}

                # we only use predicted gradients for each self.update_freq
                temp = 1.0
                if (curr_iter % self.update_freq == 0):
                    temp = self.predictor.temp

                fd.update({self.ph_lr: lr,
                           self.ph_cliprange: cliprange,
                           self.ph_temp: self.predictor.temp})

                # i need to make this faster, it is super slow
                keep_prob = np.random.uniform()

                sampled_traj_lengths = []
                if keep_prob > self.max_keep_prob:
                    s_begin = time.time()
                    # r = np.random.randint(low=1, high=(self.num_traj_rep + 1))
                    # get two samples
                    r = self.num_traj_rep
                    for _ in range(r):
                        traj_length = np.random.randint(low=nbatch_train // 4, high=nbatch_train)
                        sampled_traj_lengths.append(traj_length - 1)
                    sampled_traj_lengths.append(nbatch_train - 1)
                # self.contextor.predict(trajectory=trajectory_comes_from_policy)

                # i do NOT need this block at all
                if self.predictor.trained:
                    # there is a problem about the full_context
                    # i should remove full_context, so there is no self.predictor.ph_c2
                    # it directly comes from the policy

                    # we only look at the full past contexts to predict the gradients of the
                    # the current context
                    full_past_contexts = np.asarray(self.buf_contexts[-1][-self.max_get:])
                    # print('full_past_contexts: {}'.format(full_past_contexts.shape))

                    full_past_gradients = np.asarray(self.buf_grads[-1][-self.max_get:])
                    # print('full_past_gradients: {}'.format(full_past_gradients.shape))

                    # do not forget to update self.predictor. *stats
                    fd.update({self.predictor.ph_grads: full_past_gradients,
                               self.predictor.ph_c1: full_past_contexts,
                               self.predictor.ph_c_mean: self.predictor.c_mean,
                               self.predictor.ph_c_std: self.predictor.c_std,
                               self.predictor.ph_g_mean: self.predictor.g_mean,
                               self.predictor.ph_g_std: self.predictor.g_std})
                else:
                    fd.update({self.predictor.ph_grads: np.zeros((self.max_get, self.num_grads), dtype=np.float32),
                               self.predictor.ph_c1: np.zeros((self.max_get, self.context_dim), dtype=np.float32),
                               self.predictor.ph_c2: np.zeros((self.max_get, self.context_dim), dtype=np.float32),
                               self.predictor.ph_c_mean: self.predictor.c_mean,
                               self.predictor.ph_c_std: self.predictor.c_std,
                               self.predictor.ph_g_mean: self.predictor.g_mean,
                               self.predictor.ph_g_std: self.predictor.g_std})

                # training loop
                # we run this loop only %40 percent of the time, should increase speed
                if len(sampled_traj_lengths):                        
                    mbvals = sess().run(self._losses + (self.policy.contexts,
                                                        self.policy_grads, 
                                                        self._policy_train, self._rep_train), feed_dict=fd)[:-2]

                    # get_grads
                    all_grads = mbvals[-1]
                    contexts = mbvals[-2]
                    # first, second: partial, last: full_context
                    contexts = np.squeeze(contexts)[sampled_traj_lengths]

                    # self.buf_grads = [[part1_grads], [part2_grads], [full_grads]]
                    # self.buf_contexts = [[part1_con], [part2_con], [full_con]]
                    for i, t in enumerate(sampled_traj_lengths):
                        re_grads = []
                        for grad in all_grads:
                            # print('grad.shape: {}, t: {}'.format(grad.shape, t))
                            mean_grad = np.mean(grad[:(t+1)], 0)
                            # print('mean_grad.shape: {}, t: {}'.format(mean_grad.shape, t))
                            re_mean_grad = np.reshape(mean_grad, int(np.prod(mean_grad.shape)))
                            # print('re_mean_grad.shape: {}'.format(re_mean_grad.shape))
                            re_grads.append(re_mean_grad)

                        grads = np.concatenate([r for r in re_grads], -1)

                        # print('grads.shape: {}'.format(grads.shape))
                        self.buf_grads[i].append(grads)
                        self.buf_contexts[i].append(contexts[i])
                    
                    # append the loss values
                    mblossvals.append(mbvals[:-2])

                else:
                    mbvals = sess().run(self._losses + (self._policy_train, self._rep_train), feed_dict=fd)[:-2]
                    # append the loss values
                    mblossvals.append(mbvals)
                
                # if we are over cap_buf, delete half of the buffer
                # we should fill this cap_buf in 5000 iterations, 
                # then in each 2500 iterations, we delete 3/4 of the old ones
                # forget about this idea, just refill and train only for one epoch
                '''
                if len(self.buf_contexts[0]) > self.cap_buf:
                    for b in self.buf_contexts:
                        b = b[-(self.cap_buf // 4):]
                    for g in self.buf_grads:
                        g = b[-(self.cap_buf // 4):]
                '''

        # forget about self.jacobian_loss
        # rep_train == gradient_predictor update, we should define a loss term in terms of this gradient
        # predictor, and specifically monitor how does this loss behave, train at T/10 for the first time (for N epochs),
        # and then at each T/20 with the new gradients (for 1-2 epochs)
        if rep_train:
            print('''

                here we go with sentetik, curr iter: {}

                '''.format(curr_iter))

            # train predictor
            self.predictor.train(buf_contexts=self.buf_contexts, 
                                 buf_grads=self.buf_grads)
            
            # empty context and grad buffers, except the last self.max_get's
            for b in self.buf_contexts:
                b = b[-self.max_get:]
            for g in self.buf_grads:
                g = g[-self.max_get:]

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
        
    def save(self, curr_iter):
        save_path = os.path.join(self.save_dir, 'model')
        self.saver.save(sess(), save_path, global_step=curr_iter)

        # i need to save the following for restore
        # i jsut need the max_get full contexts and grads
        contexts = np.asarray(self.buf_contexts[-1][-self.max_get:])
        grads = np.asarray(self.buf_grads[-1][-self.max_get:])
        c_mean = self.predictor.c_mean
        c_std = self.predictor.c_std
        g_mean = self.predictor.g_mean
        g_std = self.predictor.g_std

        extra_save_path = os.path.join(self.save_dir, 'extras-{}.npz'.format(curr_iter))
        np.savez_compressed(extra_save_path,
                            contexts=contexts,
                            grads=grads,
                            c_mean=c_mean,
                            c_std=c_std,
                            g_mean=g_mean,
                            g_std=g_std)

    def load(self, load_path):
        self.saver.restore(sess(), load_path)
        print('loaded already trained model from {}'.format(load_path))