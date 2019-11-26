import tensorflow as tf
import numpy as np

# check for multiplying a sparse tensor with another sparse random matrix

in_feat_dim = 3
feat_dim = 5
keep_dim = 1

random_idx = []
for i in range(feat_dim):
    r_id = np.random.choice(in_feat_dim, keep_dim, replace=False)
    for r in r_id:
        random_idx.append([r, i])
print(random_idx)

w_init = tf.initializers.orthogonal(1.0)(shape=(keep_dim, feat_dim))
w_init = tf.reshape(w_init, [keep_dim * feat_dim])
sparse_weights = tf.SparseTensor(indices=random_idx, values=w_init, dense_shape=(in_feat_dim, feat_dim))
dense_weights = tf.sparse.to_dense(sparse_weights, default_value=0.0, validate_indices=False)

weights = tf.Variable(dense_weights, trainable=True, dtype=tf.float32, name='weights')

# get random weights
random_w_idx = []
for i in range(feat_dim):
    r_id = np.random.choice(in_feat_dim, keep_dim, replace=False)
    for r in r_id:
        random_w_idx.append([r, i])
print(random_w_idx)


# do not forget to cancel them out
random_w = tf.random.normal(shape=[keep_dim * feat_dim],
                            mean=0.0,
                            stddev=0.5,
                            name='random')

sparse_r_weights = tf.SparseTensor(indices=random_w_idx, values=random_w, dense_shape=(in_feat_dim, feat_dim))
dense_r_weights = tf.sparse.to_dense(sparse_r_weights, default_value=0.0, validate_indices=False)

result = weights + dense_r_weights

# r_weights = tf.Variable(dense_r_weights, trainable=False, dtype=tf.float32, name='r_weights')

params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(params)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(2):
        d_w, d_r_w, r = sess.run([weights, dense_r_weights, result])
        print('''

            here's d_w

            ''')
        print(d_w)
        print('''

            here's d_r_w

            ''')
        print(d_r_w)

        print('''

            here's res

            ''')
        print(r)