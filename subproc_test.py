from wrappers import make_mario_vec_env
import numpy as np
import warnings
import tensorflow as tf

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    env = make_mario_vec_env('SuperMarioBros-1-1-v0', 8, 4)

    action_n = env.action_space.n

    obs = env.reset()
    
    for i in range(10):
        print('current iter: {}'.format(i))
        acs = np.random.randint(low=0, high=action_n, size=(8,))
        obs, rews, dones, infos = env.step(acs)
        '''
        for i, d in enumerate(dones):
            if d:
                env.reset()
        '''
    
    for info in infos:
        print(info)