# base modules
import os

# necessary modules
import tensorflow as tf
import numpy as np

# policy and optimizer-related modules
from policy import Policy
from optimizers import PPO
import utils

# wrapper modules
from wrappers import make_coinrun_env, make_mario_vec_env, make_gym_mario_env
from baselines import logger

# extras
import csv
import json
import cloudpickle
import cv2
import time
import matplotlib.pyplot as plt
import sys
import shutil
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import warnings

warnings.filterwarnings('ignore')

def start_experiment(**args):
    # create environment
    # coinrun environment is already vectorized
    env, test_env = make_env_all_params(args=args)

    # set random seeds for reproducibility
    utils.set_global_seeds(seed=args['seed'])

    # create tf.session
    tf_sess = utils.setup_tensorflow_session()

    if args['server_type'] == 'local':
        logger_context = logger.scoped_configure(dir=args['log_dir'],
                                                 format_strs=['stdout', 'csv'])
    else:
        logger_context = logger.scoped_configure(dir=args['log_dir'],
                                                 format_strs=['csv'])


    with logger_context, tf_sess:
        print("logging directory: {}".format(args['log_dir']))

        # create trainer
        trainer = Trainer(env=env, test_env=test_env, args=args)

        if args['evaluation'] == 1:
            # load_path is changed to model_path
            print('run.py, def start_experiment, evaluating model: {}'.format(args['load_path']))
            trainer.eval()

        # this is for visualizing the loss landscape
        elif args['visualize'] == 1:
            print('running visualization...')
            trainer.visualize()
        else:
            print('run.py, def start_experiment, training begins...')
            trainer.train()

def make_env_all_params(args):
    if args['env_kind'] == 'coinrun':
        # DO NOT FORGET coinrun_wrapper module
        if args['input_shape'] == '64x64':
            env = make_coinrun_env(args)
        else:
            raise NotImplementedError()

    elif args['env_kind'] == 'mario':
        if args['input_shape'] == '84x84':
            start_method = 'forkserver'

            # the problem was about the initial memory cost
            # increasing the memory requirement fixed the problem
            if args['server_type'] == 'LEONHARD':
                start_method = 'forkserver'
            env = make_mario_vec_env(nenvs=args['NUM_ENVS'],
                                     env_id=args['env_id'],
                                     frameskip=args['nframeskip'],
                                     start_method=start_method)
            '''
            test_env = make_mario_vec_env(nenvs=args['NUM_ENVS'],
                                     env_id=args['test_id'],
                                     frameskip=args['nframeskip'],
                                     start_method=start_method)
            '''

        else:
            raise NotImplementedError()

    else:
        # accept only coinrun and mario
        raise NotImplementedError()
    test_env = None
    return env, test_env

class Trainer(object):
    def __init__(self, env, test_env, args):
        # coinrun is already vectorized
        # when we switch to mario, make it already vectorized
        self.env = env
        self.test_env = test_env
        self.args = args

        # ntimesteps : total number of timesteps (== number of frames for all experiments)
        # make necessary changes when you move to mario
        # nsteps : number of timesteps per rollout
        # nenvs : number of parallel rollouts
        # nframeskip : number of frames to skip (we count the skipped frames as experiences)
        self.max_iter = int(args['num_timesteps']) // (args['nsteps'] * args['NUM_ENVS'] * args['nframeskip'])
        
        # i do not want to mingle with my other frac dependent quantities
        self.early_max_iter = int(args['early_final']) // (args['nsteps'] * args['NUM_ENVS'] * args['nframeskip'])
        
        print('''

            self.max_iter: {},

            self.early_max_iter: {}

            '''.format(self.max_iter, self.early_max_iter))

        # set environment variables
        self._set_env_vars()

        # we merged cnn and lstm policies in Policy
        # calculate num_entities automatically from ob_space.shape and perception
        # kernels and strides

        coinrun = 1 if self.args['env_kind'] == 'coinrun' else 0

        transfer_dim = None
        if args['transfer_load']:
            transfer_dim = args['transfer_dim']
            
        self.policy = Policy(scope='policy',
                             ob_space=self.ob_space,
                             ac_space=self.ac_space,
                             ob_mean=self.ob_mean,
                             ob_std=self.ob_std,
                             perception=args['perception'],
                             feat_spec=args['feat_spec'],
                             policy_spec=args['policy_spec'],
                             activation=args['activation'],
                             layernormalize=args['layernormalize'],
                             batchnormalize=args['batchnormalize'],
                             vf_coef=args['vf_coef'],
                             coinrun=coinrun,
                             add_noise=args['add_noise'],
                             keep_noise=args['keep_noise'],
                             noise_std=args['noise_std'],
                             transfer_load=args['transfer_load'],
                             transfer_dim=transfer_dim,
                             num_layers=args['num_layers'],
                             keep_dim=args['keep_dim'])

        # cliprange will be annealed (was 0.1 for mario experiments)
        # linear annealing for learning rate
        lr = args['lr']
        if args['lr_lambda']:
            print('''

                linearly annealing lambda

                ''')
            lr = lambda f: f * args['lr']
        
        # linear annealing for cliprange
        cliprange = args['cliprange']
        if args['cliprange_lambda']:
            print('''

                linearly annealing cliprange
                
                ''')
            
            cliprange = lambda f: f * args['cliprange']

        ## lr and cliprange are lambda functions
        if isinstance(lr, float): lr = utils.constfn(lr)
        else: assert callable(lr)
        self.lr = lr

        if isinstance(cliprange, float): cliprange = utils.constfn(cliprange)
        else: assert callable(cliprange)
        self.cliprange = cliprange

        # max_grad_norm
        max_grad_norm = args['max_grad_norm']

        # in case we are restoring the training
        self.restore_iter = self.args['restore_iter']
        self.load_path = None
        if self.restore_iter > -1:
            self.load_dir = self.args['save_dir']
            if args['transfer_load']:
                print('''


                    TRANSFER LOAD...


                    ''')
                self.load_dir = self.args['load_dir']
            self.load_path = os.path.join(self.load_dir, 'model-{}'.format(self.restore_iter))

        print('''

            loading model from {}

            '''.format(self.load_path))

        # get ob_space from self.policy
        self.agent = PPO(scope='ppo',
                         env=self.env,
                         test_env=self.test_env,
                         nenvs=args['NUM_ENVS'],
                         save_dir=args['save_dir'],
                         log_dir=args['log_dir'],
                         policy=self.policy,
                         use_news=args['use_news'],
                         gamma=args['gamma'],
                         lam=args["lambda"],
                         nepochs=args['nepochs'],
                         nminibatches=args['nminibatches'],
                         max_grad_norm=args['max_grad_norm'],
                         nsteps=args['nsteps'],
                         vf_coef=args['vf_coef'],
                         ent_coef=args['ent_coeff'],
                         normrew=args['norm_rew'],
                         cliprew=args['clip_rew'],
                         normadv=args['norm_adv'],
                         for_visuals=args['for_visuals'],
                         transfer_load=args['transfer_load'],
                         load_path=self.load_path,
                         freeze_weights=args['freeze_weights'])

    def _load_mean_std(self, load_path_pkl):
        with open(load_path_pkl, 'rb') as file_:
            data = cloudpickle.load(file_)
        self.ob_mean, self.ob_std = data['ob_mean'], data['ob_std']

    def _set_env_vars(self):
        # set observation_space, action_space
        self.ob_space, self.ac_space = self.env.observation_space, self.env.action_space

        self.ob_mean, self.ob_std = 0.0, 1.0
        
        if self.args['norm_obs']:        
            if self.args['evaluation']:
                # load_path = self.args['load_path'][0].split('/')[:-1]
                # load_path = '/'.join(load_path)
                # load_path_pkl = os.path.join(load_path, 'mean_std.pkl')
                
                load_path_pkl = os.path.join(self.args['load_dir'], 'mean_std.pkl') 
                self._load_mean_std(load_path_pkl)
            else:
                save_path_pkl = os.path.join(self.args['load_dir'], 'mean_std.pkl')
                if self.args['restore_iter'] > -1:
                    self._load_mean_std(save_path_pkl)

                    # copy the transfered mean_std.pkl to the new folder
                    # this will be used to test for forgetting
                    if self.args['transfer_load']:
                        cp_path_pkl = os.path.join(self.args['save_dir'], 'mean_std.pkl')
                        shutil.copyfile(save_path_pkl, cp_path_pkl)

                else:
                    self.ob_mean, self.ob_std = utils.random_agent_mean_std(env=self.env)
                    with open(save_path_pkl, 'wb') as file_:
                        data = {'ob_mean': self.ob_mean,
                                'ob_std': self.ob_std}
                        cloudpickle.dump(data, file_)

        # october 30, 2019
        # we moved mean and std for observation to agent
        # not sure if coinrun normalizes observations, will be checked
        # october 31, 2019
        # coinrun DOES NOT normalize observations with running mean and std
        # only scales dividing by 255. (already done)

    def train(self):
        curr_iter = 0

        # train progress results logger
        format_strs = ['csv']
        format_strs = filter(None, format_strs)
        dirc = os.path.join(self.args['log_dir'], 'inter')
        output_formats = [logger.make_output_format(f, dirc) for f in format_strs]
        self.result_logger = logger.Logger(dir=dirc, output_formats=output_formats)

        # in case we are restoring the training
        if self.restore_iter > -1:
            self.agent.load(self.load_path)
            if not self.args['transfer_load']:
                curr_iter = self.restore_iter
            
        print('max_iter: {}'.format(self.max_iter))

        
        # interim saves to compare in the future
        # for 128M frames, 
        
        inter_save = []
        for i in range(3):
            divisor = (2**(i+1))
            inter_save.append(int(self.args['num_timesteps'] // divisor) // (self.args['nsteps'] * self.args['NUM_ENVS'] * self.args['nframeskip']))
        print('inter_save: {}'.format(inter_save))

        total_time = 0.0
        # results_list = []

        while curr_iter < self.early_max_iter:
            frac = 1.0 - (float(curr_iter) / self.max_iter)

            # self.agent.update calls rollout
            start_time = time.time()

            ## linearly annealing
            curr_lr = self.lr(frac)
            curr_cr = self.cliprange(frac)
                        
            ## removed within training evaluation
            ## i could not make flag_sum to work properly
            ## evaluate each 100 run for 20 training levels
            # only for mario (first evaluate, then update)
            # i am doing change to get zero-shot generalization without any effort
            if curr_iter % (self.args['save_interval']) == 0:
                save_video = False
                nlevels = 20 if self.args['env_kind'] == 'mario' else self.args['NUM_LEVELS']
                results, _ = self.agent.evaluate(nlevels, save_video)
                results['iter'] = curr_iter
                for (k, v) in results.items():
                    self.result_logger.logkv(k, v)
                self.result_logger.dumpkvs()

            # representation learning in each 25 steps
            info = self.agent.update(lr=curr_lr, cliprange=curr_cr)
            end_time = time.time()

            # additional info
            info['frac'] = frac
            info['curr_lr'] = curr_lr
            info['curr_cr'] = curr_cr
            info['curr_iter'] = curr_iter
            # info['max_iter'] = self.max_iter
            info['elapsed_time'] = end_time - start_time
            # info['total_time'] = total_time = (total_time + info['elapsed_time']) / 3600.0
            info['expected_time'] = self.max_iter * info['elapsed_time'] / 3600.0

            ## logging results using baselines's logger
            logger.logkvs(info)
            logger.dumpkvs()

            if curr_iter % self.args['save_interval'] == 0:
                self.agent.save(curr_iter, cliprange=curr_cr)

            if curr_iter in inter_save:
                self.agent.save(curr_iter, cliprange=curr_cr)
            
            curr_iter += 1

        self.agent.save(curr_iter, cliprange=curr_cr)

        # final evaluation for mario
        save_video = False
        nlevels = 20 if self.args['env_kind'] == 'mario' else self.args['NUM_LEVELS']
        results, _ = self.agent.evaluate(nlevels, save_video)
        results['iter'] = curr_iter
        for (k, v) in results.items():
            self.result_logger.logkv(k, v)
        self.result_logger.dumpkvs()

    def eval(self):
        # create base_dir to save results
        env_id = self.args['env_id'] if self.args['env_kind'] == 'mario' else self.args['eval_type']
        # base_dir =  os.path.join(self.args['log_dir'], self.args['exp_name'], env_id)
        # os.makedirs(base_dir, exist_ok=True)

        # i forget to restore, i cannot believe myself
        # load_path = self.args['load_path']

        # args['IS_HIGH_RES'] is used to signal whether save videos
        nlevels =  self.args['NUM_LEVELS']

        save_video = False
        
        # train progress results logger
        format_strs = ['csv']
        format_strs = filter(None, format_strs)
        dirc = os.path.join(self.args['log_dir'], 'inter')
        output_formats = [logger.make_output_format(f, dirc) for f in format_strs]
        self.result_logger = logger.Logger(dir=dirc, output_formats=output_formats)

        if self.args['env_kind'] == 'mario':
            # do NOT FORGET to change this
            nlevels = 20

        # curr_iter = 0
        # results_list = []
        restore_iter = [25 * i for i in range(59)] + [1464]

        for r in restore_iter:
            load_path = os.path.join(self.args['load_dir'], 'model-{}'.format(r))
            print(load_path)
            self.agent.load(load_path)
            
            save_video = False
            nlevels = 20 if self.args['env_kind'] == 'mario' else self.args['NUM_LEVELS']
            results, _ = self.agent.evaluate(nlevels, save_video)
            results['iter'] = r
            for (k, v) in results.items():
                self.result_logger.logkv(k, v)
            self.result_logger.dumpkvs()
        
        '''    
        results['iter'] = curr_iter = int(l.split('/')[-1].split('-')[-1])
        print(results)
        results_list.append(results)

        csv_columns = results_list[0].keys()
        print(csv_columns)

        curr_dir = os.path.join(base_dir, str(curr_iter))
        os.makedirs(curr_dir, exist_ok=True)
        
        csv_save_path = os.path.join(curr_dir, 'results.csv'.format())
        with open(csv_save_path, 'w') as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)
            writer.writeheader()
            for data in results_list:
                writer.writerow(data)
        print('results are dumped to {}'.format(csv_save_path))
        '''

        '''
        # saving video
        print('beginning saving video...')
        # maximum number of saved videos
        # max_eval_imgs = 5
        # print(len(eval_imgs))
        # eval_imgs = eval_imgs[:max_eval_imgs]
        # print(len(eval_imgs), len(eval_imgs[0]))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        h, w = eval_imgs[0][0].shape[1], eval_imgs[0][0].shape[0]
        outs = [cv2.VideoWriter(os.path.join(curr_dir, 'rlvideo-{}.avi'.format(i)), 
                                fourcc, 20.0, (h, w)) for i in range(len(eval_imgs))]
        for ix, episode in enumerate(eval_imgs):
            for frame in episode:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                outs[ix].write(frame)

        for out in outs:
            out.release()
        '''

    def visualize(self):

        r_dir_taken = False

        surface_dir = os.path.join(self.args['save_dir'], 'surface_plots')
        os.makedirs(surface_dir, exist_ok=True)

        var_dict = {}
        dir_dict = {0: {}, 1: {}}
        # temp_r_dict = {0: {}, 1: {}}
        # for restore_iter in range(0, 1500, 300):
        for restore_iter in [1450]:
            temp_r_dict = {0: {}, 1: {}}
        
            print('restore_iter: {}'.format(restore_iter))

            npz_file = '{}/extra-{}.npz'.format(self.args['save_dir'], restore_iter)
            bufs = np.load(npz_file)

            print(bufs['obs'].shape)

            # load data
            self.agent.load_ph_bufs(bufs)

            # we create r_dir only once
            load_path = os.path.join(self.args['save_dir'], 'model-{}'.format(restore_iter))

            for var_ckpt in tf.train.list_variables(load_path):
                # remove learning-related variables
                # we are also ignoring biases
                not_count = 'beta' in var_ckpt[0] or 'Adam' in var_ckpt[0]
                if not not_count:
                    # this gives the shapes of variables
                    var_shape = var_ckpt[1]
                    var = tf.train.load_variable(load_path, var_ckpt[0])                    
                    var_dict[var_ckpt[0]] = var

                    for i in range(2):
                        if not r_dir_taken:
                            r_dir = np.random.normal(size=var_shape)
                            dir_dict[i][var_ckpt[0]] = r_dir
                            # temp_r_dict[i][var_ckpt[0]] = r_dir
                        else:
                            r_dir = np.copy(dir_dict[i][var_ckpt[0]])

                        # this means convolution
                        if len(var_shape) > 3:
                            # normalize cnns
                            num_filter = var_shape[-1]
                            for ind in range(num_filter):
                                fro_weight = np.linalg.norm(var[:, :, :, ind])
                                fro_dir = np.linalg.norm(r_dir[:, :, :, ind])
                                r_dir[:, :, :, ind] = (r_dir[:, :, :, ind] / fro_dir) * fro_weight
                        else:
                            fro_weight = np.linalg.norm(var)
                            fro_dir = np.linalg.norm(r_dir)
                            r_dir = (r_dir / fro_dir) * fro_weight
                        temp_r_dict[i][var_ckpt[0]] = r_dir
            r_dir_taken = True

            print('done creating directions')

            print('getting losses')
            xs = np.arange(-1.0, 1.0, 0.1)
            ys = np.arange(-1.0, 1.0, 0.1)
            zs = np.zeros((xs.shape[0], ys.shape[0]))
            
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    # start_time = time.time()
                    '''
                    z = self.agent.get_loss(v_dict=var_dict, dir_dict=temp_r_dict,
                                            alpha=x, beta=y)
                    '''
                    z = self.agent.re_run_loss(v_dict=var_dict, dir_dict=temp_r_dict,
                                               alpha=x, beta=y, bufs=bufs)

                    # end_time = time.time()
                    # print('one iteration takes: {}'.format(end_time-start_time))
                    zs[i, j] = np.clip(z, -1.0, 1.0)
            xs, ys = np.meshgrid(xs, ys)
            
            npz_save_file = os.path.join(surface_dir, 'surface-{}.npz'.format(restore_iter))
            np.savez_compressed(npz_save_file,
                                xs=xs,
                                ys=ys,
                                zs=zs)

            '''
            print('surface projections...')

            fig = plt.figure()
            ax = plt.axes(projection='3d')

            ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm, edgecolor='none')
            ax.set_title('loss-surface-{}'.format(restore_iter))
            plt.show()
            '''
###
# MAIN
###
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_spec', type=str, default='')
    parser.add_argument('--restore_iter', type=int, default=-1)
    parser.add_argument('--server_type', type=str, default='local')
    parser.add_argument('--visualize', type=int, default=0)
    parser.add_argument('--evaluate', type=int, default=0)
    args = parser.parse_args()

    with open(args.model_spec, 'r') as file:
        print('loading configuration variables from: {}'.format(args.model_spec))
        train_args = json.load(file)

    # only create log_dir and save_dir for the experiment
    # IF YOU ARE RUNNING THE F*CKING EXPERIMENT!!!
    os.makedirs(train_args['log_dir'], exist_ok=True)
    os.makedirs(train_args['save_dir'], exist_ok=True)

    train_args['restore_iter'] = args.restore_iter
    train_args['server_type'] = args.server_type
    train_args['visualize'] = args.visualize
    train_args['evaluation'] = args.evaluate
    start_experiment(**train_args)