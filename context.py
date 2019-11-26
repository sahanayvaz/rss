import tensorflow as tf
import numpy as np

sess = tf.get_default_session

class ContextGenerator(object):
    # this the random context generator
    def __init__(self, num_input_feat, context_dim, trajectory):
        # initialize a very basic RNN network here
        # time_major is False, [batch-size, max_time, num_feat]
        # self.ph_feats = tf.placeholder(tf.float32, [None, None, num_input_feat])

        '''
        rnn_cell = tf.contrib.cudnn_rnn.CudnnRNNTanh(num_layers=1,
                                                     num_units=context_dim,
                                                     kernel_initializer=tf.initializers.orthogonal(np.sqrt(2)),
                                                     bias_initializer=tf.constant_initializer(0.0))
        '''
        rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(context_dim)
        # we might try to give everything in batch
        # i should look at the speed improvement

        # do not forget to add this to your 
        # self.ph_batch_size = tf.placeholder(tf.int32, [])

        # initial_state = rnn_cell.zero_state(1, dtype=tf.float32)
        initial_state = np.zeros((1, context_dim), dtype=np.float32)

        # we connect self.policy.trajectory to dynamic_rnn
        self.outputs, state = tf.compat.v1.nn.dynamic_rnn(rnn_cell, trajectory,
                                                          initial_state=initial_state,
                                                          dtype=tf.float32,
                                                          parallel_iterations=64)
    '''
    # this module will not be trained
    def train(self):
        raise NotImplementedError()

    # get the context from randomly initialized network
    def predict(self, trajectory):
        # return self.outputs
        return sess().run(self.outputs, feed_dict={self.ph_feats: trajectory})
    '''