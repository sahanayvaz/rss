import numpy as np
import copy

'''
TODO: implement recurrent version
'''

class Runner(object):
    def __init__(self, env, nenvs, policy, nsteps, cliprew):
        # environment related
        self.env = env
        self.nenvs = nenvs
        self.ob_space = env.observation_space
        self.ac_space = env.action_space

        # policy
        self.policy = policy
        
        # nsteps
        self.nsteps = nsteps

        # whether to cliprewards or not
        self.cliprew = cliprew

        self.buf_obs = np.empty((nenvs, self.nsteps, *self.ob_space.shape), self.ob_space.dtype)
        self.buf_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_acs = np.empty((nenvs, self.nsteps, *self.ac_space.shape), self.ac_space.dtype)
        
        self.buf_vpreds = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_nlps = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_obs_last = np.empty((nenvs, *self.ob_space.shape), np.float32)

        self.buf_news = np.zeros((nenvs, self.nsteps), np.float32)
        
        self.buf_new_last = self.buf_news[:, 0, ...].copy()
        self.buf_vpred_last = self.buf_vpreds[:, 0, ...].copy()

        self.obs = self.env.reset()
        self.dones = [False for _ in range(self.nenvs)]

    def rollout(self):
        for t in range(self.nsteps):
            acs, vals, nlps = self.policy.step(self.obs)

            # fill the buffers
            self.buf_obs[:, t] = self.obs
            self.buf_acs[:, t] = acs
            self.buf_vpreds[:, t] = vals
            self.buf_nlps[:, t] = nlps
            self.buf_news[:, t] = self.dones
            
            self.obs, rews, self.dones, infos = self.env.step(acs)
            self.buf_rews[:, t] = rews

        acs, vals, _ = self.policy.step(self.obs)
        self.buf_new_last = self.dones
        self.buf_vpred_last = vals

        # clip rewards between -1.0 and 1.0
        # we might change the clipping factors
        if self.cliprew:
            self.buf_rews[:] = np.clip(self.buf_rews, -1.0, 1.0)

    def evaluate(self, nlevels, save_video):
        # this is NOT a good practice
        eval_news = 0.0
        eval_rews = []
        eval_lengths = []
        eval_xpos = []
        eval_flag = []
        eval_imgs = []

        nenv_rews = np.zeros(self.env.num_envs)
        nenv_lengths = np.zeros(self.env.num_envs)

        buf_imgs = [[] for _ in range(self.env.num_envs)]

        while eval_news < nlevels:
            acs, vals, nlps = self.policy.step(self.obs)
            self.obs, rews, news, infos = self.env.step(acs)

            if save_video:
                ## if we are going to render images this way, i NEED to change 
                ## stable_baselines.common.vec_env.subproc_vec_env.py
                imgs = self.env.get_images()
                for i, img in enumerate(imgs):
                    buf_imgs[i].append(copy.deepcopy(imgs[i]))

            nenv_rews += rews
            nenv_lengths += 1.0
            for i, z in enumerate(zip(news, infos)):
                d = z[0]
                try:
                    f = z[1]['flag_get']
                except:
                    f = False
                if d or f:
                    eval_rews.append(nenv_rews[i])
                    eval_lengths.append(nenv_lengths[i])
                    eval_imgs.append(buf_imgs[i])
                    buf_imgs[i] = []

                    # for MARIO, we have x_pos and flag_get
                    try:
                        eval_xpos.append(infos[i]['x_pos'])
                        flag_get = 1 if infos[i]['flag_get'] > 0 else 0
                        eval_flag.append(flag_get)

                    # for COINRUN, we only have reward
                    except:
                        # check out coinrun's infos
                        # info['episode']['r'] and ['l'] and ['t']
                        eval_xpos.append(0)
                        eval_flag.append(0)

                    nenv_rews[i] = 0
                    nenv_lengths[i] = 0
                    eval_news += 1

        results = {'rew_mean': np.mean(eval_rews),
                   'rew_std': np.std(eval_rews),
                   'rew_max': np.max(eval_rews),
                   'rew_min': np.min(eval_rews),
                   'rew_maxID': np.argmax(eval_rews),
                   'rew_minID': np.argmin(eval_rews),
                   'len_mean': np.mean(eval_lengths),
                   'len_std': np.std(eval_lengths),
                   'x_mean': np.mean(eval_xpos),
                   'x_std': np.std(eval_xpos),
                   'x_max': np.max(eval_xpos),
                   'x_min': np.min(eval_xpos),
                   'x_maxID': np.argmax(eval_xpos),
                   'x_minID': np.argmin(eval_xpos),
                   'flag_sum': np.sum(eval_flag)}

        return results, eval_imgs