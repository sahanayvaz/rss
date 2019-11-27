from wrappers import make_mario_env
import numpy as np
import warnings
import tensorflow as tf
from stable_baselines.common.vec_env import SubprocVecEnv

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    env_id = 'SuperMarioBros-1-1-v0'
    nenvs = 8 
    frameskip = 4

    env = SubprocVecEnv(env_fns=[make_mario_env(env_id, frameskip, i)  for i in range(nenvs)],
                        start_method='spawn')
    
    action_n = env.action_space.n

    print('observation_shape: {}'.format(env.observation_space.shape))

    ph_obs = tf.to_float(tf.placeholder(dtype=env.observation_space.dtype, shape=(None,) + env.observation_space.shape, name='ph_obs'))
    out = tf.layers.conv2d(ph_obs, filters=5, kernel_size=10,
                           strides=10, activation=tf.nn.relu,
                           kernel_initializer=tf.initializers.orthogonal(1.0),
                           bias_initializer=tf.constant_initializer(0.0),
                           name='conv2d')
    out = tf.layers.flatten(out)
    out = tf.layers.dense(out, units=action_n, activation=None,
                          kernel_initializer=tf.initializers.orthogonal(1.0),
                          bias_initializer=tf.constant_initializer(0.0))
    out = tf.random.categorical(logits=out, num_samples=1)
    obs = env.reset()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10):
            print('current iter: {}'.format(i))
            # acs = np.random.randint(low=0, high=action_n, size=(8,))
            obs = obs.astype(env.observation_space.dtype)
            acs = sess.run(out, feed_dict={ph_obs: obs})
            obs, rews, dones, infos = env.step(np.squeeze(acs))
    
    for info in infos:
        print(info)