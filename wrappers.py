import gym
import numpy as np
from PIL import Image

def make_coinrun_env(config):
    from coinrun import coinrunenv
    from coinrun.coinrunenv import init_args_and_threads
    from coinrun import wrappers

    cpu_count = 4
    init_args_and_threads(cpu_count=cpu_count,
                          rand_seed=49,
                          config=config)

    env = coinrunenv.make(config)
    env = wrappers.EpisodeRewardWrapper(env)
    
    return env

class FrameSkip(gym.Wrapper):
    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n = n

    def step(self, action):
        done = False
        totrew = 0
        for _ in range(self.n):
            ob, rew, done, info = self.env.step(action)
            totrew += rew
            if done: break
        return ob, totrew, done, info


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env, crop=True, keep_dims=False):
        self.crop = crop
        self.keep_dims = keep_dims
        super(ProcessFrame84, self).__init__(env)
        if keep_dims:
            # as you know, i hate this hard-coded stuff; however, i need to hurry up
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(190, 240, 1), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs, crop=self.crop, keep_dims=self.keep_dims)

    @staticmethod
    def process(frame, crop, keep_dims):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        elif frame.size == 224 * 240 * 3:  # mario resolution
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
        elif frame.size == 240 * 256 * 3: # gym_super_mario resolution
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution." + str(frame.size)
        
        # grayscale
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        
        if keep_dims:
            x_t = img[24:214, :]
            # x_t = img
            # print(x_t.shape)
            x_t = np.reshape(x_t, [190, 240, 1])

        else:
            size = (84, 110 if crop else 84)
            resized_screen = np.array(Image.fromarray(img).resize(size,
                                                                  resample=Image.BILINEAR), dtype=np.uint8)
            x_t = resized_screen[18:102, :] if crop else resized_screen
            x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

'''
1. gym_super_mario has Discrete(12) actions, large-scale-curiosity mario Discrete(14) 
2. gym_super_mario has slightly different reward function than that of retro(SuperMarioBros) (punishing standstill and death)
'''
def make_gym_mario_env(env_id, frameskip, crop=True, frame_stack=True):
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    from baselines.common.atari_wrappers import FrameStack

    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    env = FrameSkip(env, frameskip)
    env = ProcessFrame84(env, crop=crop)

    if frame_stack:
        env = FrameStack(env, 4)
    return env

def make_mario_env(env_id, frameskip, rank, seed=0):
    '''
    env_id: environment id 
    rank: rank of the environment
    seed: to control randomness 
    '''
    def __init():
        env = make_gym_mario_env(env_id, frameskip)
        env.seed(seed + rank)
        return env
    return __init

def make_mario_vec_env(env_id, nenvs, frameskip):
    from stable_baselines.common.vec_env import SubprocVecEnv
    env = SubprocVecEnv([make_mario_env(env_id, frameskip, i)  for i in range(nenvs)])
    return env
