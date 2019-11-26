import numpy as np
import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian
import time

nbatch_train = 1
feat_dim = 5
h, w, c = 64, 64, 3
act_dim = 2
neps = 6

features = np.random.normal(size=(nbatch_train, feat_dim))

ph_features = tf.placeholder(dtype=tf.float32, shape=(None, feat_dim), name='features')
ph_jacobians = tf.placeholder(dtype=tf.float32, shape=(None, act_dim, feat_dim), name='jacobians')

out = tf.layers.dense(ph_features, units=feat_dim, activation=tf.nn.relu, name='layer1')
out = tf.layers.dense(out, units=feat_dim, activation=tf.nn.relu, name='layer2')
out = tf.layers.dense(out, units=act_dim, activation=None, name='layer3')
out = tf.nn.softmax(out)

'''
jacobian_op = batch_jacobian(out, ph_features)
eps = tf.random.normal(shape=(feat_dim, neps))

# returns batch_size, act_dim, neps
matmul_op = tf.matmul(jacobian_op, eps)

neta_op = tf.divide(matmul_op, tf.expand_dims(out, -1)) + 1.0

# should return batch_size, act_dim
exp_neta_op = tf.reduce_mean((neta_op - 1) * tf.log(neta_op), axis=-1)
loss = tf.reduce_sum(tf.multiply(out, exp_neta_op), axis=-1)
red_loss = tf.reduce_mean(loss)
'''

params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
trainer = tf.train.AdamOptimizer()
gradsandvar = trainer.compute_gradients(out, params)

# grads are the gradients and var is the variable values in the computational graph
grads, var = zip(*gradsandvar)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    grads_out, v_out = sess.run([grads, var], feed_dict={ph_features: features})
    
    print(len(grads_out))
    print(grads_out)
    for g in grads_out:
        print(g.shape)
    g_array = np.concatenate([np.reshape(g, [np.prod(g.shape),]) for g in grads_out])
    print(g_array.shape)