import tensorflow as tf
import numpy as np

sess = tf.get_default_session

class GradPredictor(object):
    # initialize the gradient predictor
    def __init__(self, list_grads, context_dim, cap_buf, 
                 min_temp, max_iter, policy_context, max_pos,
                 max_history, csv_path):

        # this is going to be an MLP
        # look at context vectors and gradient lengths
        self.trained = False
        self.pointer = 0
        self.cap_buf = cap_buf

        # as we train more and more gradients, this temp should decrease
        # self.temp always starts from 1.0
        max_temp = 1.0
        self.temp = max_temp
        self.decay_ratio = max_temp - min_temp
        self.train_iter = 0.0
        self.max_iter = max_iter

        self.max_pos = int(max_pos)
        self.max_history = max_history

        # initialize mlp
        split_grads = [np.prod(l) for l in list_grads]
        num_grads = np.sum(split_grads)

        self.ph_grads = tf.placeholder(tf.float32, [None, num_grads], name='ph_grads')
        self.ph_true_grads = tf.placeholder(tf.float32, [None, num_grads], name='ph_true_grads')
        self.ph_c1 = tf.placeholder(tf.float32, [None, context_dim], name='ph_c1')
        self.ph_c2 = tf.placeholder(tf.float32, [None, context_dim], name='ph_c2')
        
        # do NOT forget to get those
        self.ph_c_mean = tf.placeholder(tf.float32, [1, context_dim], name='ph_c_mean')
        self.ph_c_std = tf.placeholder(tf.float32, [1, context_dim], name='ph_s_mean')
        self.ph_g_mean = tf.placeholder(tf.float32, [1, num_grads], name='ph_c_mean')
        self.ph_g_std = tf.placeholder(tf.float32, [1, num_grads], name='ph_s_mean')

        self.c_mean = np.zeros((1, context_dim), dtype=np.float32)
        self.c_std = np.ones((1, context_dim), dtype=np.float32)
        self.g_mean = np.zeros((1, num_grads), dtype=np.float32)
        self.g_std = np.ones((1, num_grads), dtype=np.float32)
        
        in_grads = (self.ph_grads - self.ph_g_mean) / self.ph_g_std
        in_c1 = (self.ph_c1 - self.ph_c_mean) / self.ph_c_std
        in_c2 = (policy_context - self.ph_c_mean) / self.ph_c_std

        units = num_grads // 2
        inf_out = tf.concat([in_grads, in_c1, in_c2], axis=-1)

        predictor_scope = 'gradpredictor'
        with tf.variable_scope(predictor_scope, reuse=False):
            out = tf.layers.dense(inf_out, units=units, activation=tf.nn.relu,
                                  kernel_initializer=tf.initializers.orthogonal(np.sqrt(2)),
                                  bias_initializer=tf.constant_initializer(0.0))
            out = tf.layers.dense(out, units=num_grads, activation=None,
                                  kernel_initializer=tf.initializers.orthogonal(np.sqrt(2)),
                                  bias_initializer=tf.constant_initializer(0.0))

        # mean_out = tf.reduce_mean(out, 0)
        grads = tf.split(out, split_grads, axis=1)
        self.re_grads = []
        for i, sh_grads in enumerate(list_grads):
            g = tf.reshape(grads[i], [-1,] + sh_grads)
            self.re_grads.append(tf.reduce_mean(g, 0))
        
        # grad_0 = tf.reshape(grads[0], [-1, 512, 7])
        # grad_1 = tf.reshape(grads[1], [-1, 7])
        
        # grad_2 = tf.reshape(grads[2], [-1, 512, 1])
        # grad_3 = tf.reshape(grads[3], [-1, 1])

        # self.grad_0 = tf.reduce_mean(grad_0, 0)
        # self.grad_1 = tf.reduce_mean(grad_1, 0)

        # this part is for training
        train_out = tf.concat([self.ph_grads, self.ph_c1, self.ph_c2], axis=-1)
        with tf.variable_scope(predictor_scope, reuse=True):
            out = tf.layers.dense(train_out, units=units, activation=tf.nn.relu,
                                  kernel_initializer=tf.initializers.orthogonal(np.sqrt(2)),
                                  bias_initializer=tf.constant_initializer(0.0))
            out = tf.layers.dense(out, units=num_grads, activation=None,
                                  kernel_initializer=tf.initializers.orthogonal(np.sqrt(2)),
                                  bias_initializer=tf.constant_initializer(0.0))

        ## i will change this LATER
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(out - self.ph_true_grads), axis=-1))
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.train_op = trainer.minimize(self.loss)
        self.csv_path = csv_path

    # this is the training step
    # i want to feed those values explicitly
    def train(self, buf_contexts, buf_grads):
        print('''

            training gradients

            ''')


        # increase training_iter
        self.train_iter += 1.0

        # update temperature
        self.temp = self.temp - self.decay_ratio * (self.train_iter / self.max_iter)
        # print('current_temp: {}'.format(self.temp))

        # subtract some number from contexts to easily align the training set
        max_subt = self.max_pos * (self.max_history + 1)

        ## normalization
        def mean_std(arr):
            return np.expand_dims(np.mean(arr, axis=0), 0), np.expand_dims((np.std(arr, axis=0) + 0.1), 0)

        contexts = np.concatenate([np.asarray(buf_contexts[i]) for i in range(3)], 0)
        grads = np.concatenate([np.asarray(buf_grads[i]) for i in range(3)], 0)

        self.c_mean, self.c_std = mean_std(contexts)
        self.g_mean, self.g_std = mean_std(grads) 

        # print(contexts.shape, grads.shape)
        # print(self.c_mean.shape, self.c_std.shape, self.g_mean.shape, self.g_std.shape)

        contexts = (contexts - self.c_mean) / self.c_std
        grads = (grads - self.g_mean) / self.g_std

        # print(contexts.shape, grads.shape)

        train_len = (contexts.shape[0] // 3) - max_subt
        # print('train_len: {}, max_subt: {}'.format(train_len, max_subt))

        c1, c2, grads_c1, grads_c2 = [], [], [], []

        # print(contexts[-(train_len + max_subt):].shape)

        for i in range(self.max_history + 1):
            for j in range(2):
                train_size = np.random.randint(low=train_len // 2, high=train_len)
                rand_idx = np.random.randint(low=0, high=train_len, size=train_size)
                # new_idx = train_idx[rand_idx]
                rand_idx_c = rand_idx + j * (train_len + max_subt)

                c1.append(contexts[rand_idx_c])

                # i made a mistake there, but found it thanks to GOD!
                # could have been drive me crazy!!!
                c2.append(contexts[-(train_len + max_subt):][rand_idx + i * self.max_pos])

                grads_c1.append(grads[rand_idx])
                grads_c2.append(grads[-(train_len + max_subt):][rand_idx + i * self.max_pos])

        c1 = np.concatenate([c for c in c1], 0)
        c2 = np.concatenate([c for c in c2], 0)
        grads_c1 = np.concatenate([c for c in grads_c1], 0)
        grads_c2 = np.concatenate([c for c in grads_c2], 0)
        
        # print('c1.shape: {}, c2.shape: {}, grads_c1.shape: {}, grads_c2.shape: {}'.format(
        #        c1.shape, c2.shape, grads_c1.shape, grads_c2.shape))

        if self.trained:
            nepochs = 1
            training_data = list(np.genfromtxt(self.csv_path, delimiter=','))

        else:
            # first time training
            self.trained = True
            nepochs = 5
            training_data = []
            
        nbatch = c1.shape[0]
        idx = np.arange(nbatch)
        nbatch_train = 512
        # nbatch = train_size // n_batch_train
        for e in range(nepochs):
            np.random.shuffle(idx)
            batch_loss = []
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbidx = idx[start:end]
                b_l , _ = sess().run([self.loss, self.train_op],
                                      feed_dict={self.ph_c1: c1[mbidx],
                                                 self.ph_c2: c2[mbidx],
                                                 self.ph_grads: grads_c1[mbidx],
                                                 self.ph_true_grads: grads_c2[mbidx]})
                batch_loss.append(b_l)
            training_data.append([np.mean(b_l), self.train_iter - 1, self.temp])
            # print('predictor training epoch: {}, training loss: {}'.format(e, np.mean(batch_loss)))
        
        training_data = np.asarray(training_data)
        np.savetxt(self.csv_path, training_data, delimiter=',')
    
    '''
    # not using predict
    def predict(self, c1, c2, grad_c1):
        # normalize
        c1 = (c1 - self.m_cont) / (self.s_cont)
        # print(c1.shape)
        c2 = (c2 - self.m_cont) / (self.s_cont)
        c2 = np.asarray([c2 for _ in range(c1.shape[0])])
        # print(c2.shape)

        grad_c1 = (grad_c1 - self.m_grad) / (self.s_grad)

        inpt = np.concatenate([grad_c1, c1, c2], axis=-1)
        pred = sess().run(self.reduce_out, feed_dict={self.ph_inputs: inpt})
        return pred
    '''