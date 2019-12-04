import json
import argparse
import os

def add_environment_params(parser):
    # environment configs
    parser.add_argument('--env_kind', type=str, default='mario')
    parser.add_argument('--env_id', type=str, default='SuperMarioBros-1-1-v0')
    parser.add_argument('--test_id', type=str, default=None)
    parser.add_argument('--NUM_ENVS', type=int, default=8)
    parser.add_argument('--NUM_LEVELS', type=int, default=500)

    # we might play with this seed
    parser.add_argument('--SET_SEED', type=int, default=13)
    parser.add_argument('--PAINT_VEL_INFO', type=int, default=1)
    parser.add_argument('--USE_DATA_AUGMENTATION', type=int, default=0)
    parser.add_argument('--GAME_TYPE', type=str, default='standard')
    parser.add_argument('--USE_BLACK_WHITE', type=int, default=0)
    parser.add_argument('--IS_HIGH_RES', type=int, default=0)
    parser.add_argument('--HIGH_DIFFICULTY', type=int, default=0)
    parser.add_argument('--nframeskip', type=int, default=4)

def add_optimization_params(parser):
    # optimization related params

    # use_news default 1 as used in coinrun experiments
    parser.add_argument('--use_news', type=int, default=1)
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--nminibatches', type=int, default=8)

    # lr_lambda is to linearly anneal
    parser.add_argument('--lr_lambda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)

    # look at coinrun cliprange
    # cliprange_lambda is to linearly anneal cliprange
    parser.add_argument('--cliprange_lambda', type=int, default=0)
    parser.add_argument('--cliprange', type=float, default=0.1)

    parser.add_argument('--norm_obs', type=int, default=1)
    parser.add_argument('--norm_adv', type=int, default=1)
    parser.add_argument('--norm_rew', type=int, default=1)
    parser.add_argument('--clip_rew', type=int, default=1)
    parser.add_argument('--ent_coeff', type=float, default=0.001)

    # look at coinrun max_grad_norm
    parser.add_argument('--vf_coef', type=float, default=1.0)
    parser.add_argument('--max_grad_norm', type=float, default=40.0)
    
    # we are running those experiments for 64M timesteps (== frames)
    parser.add_argument('--num_timesteps', type=int, default=int(128e6))
    parser.add_argument('--early_final', type=int, default=int(128e6))

def add_rollout_params(parser):
    # rollout related params
    # the original coinrun uses 32 environments per works with 256 timesteps per env
    # this way, we make sure that we are making the same updates as coinrun
    parser.add_argument('--nsteps', type=int, default=int(128))

def add_network_params(parser):
    parser.add_argument('--input_shape', type=str, default='84x84')

    # network_related params
    parser.add_argument('--perception', type=str, default='nature_cnn')

    ## if you make a change to cr-fc-v0, make sure to update
    parser.add_argument('--feat_spec', type=str, default='feat_v0')
    parser.add_argument('--policy_spec', type=str, default='ls_c_v0')

    # check coinrun activation
    parser.add_argument('--activation', type=str, default='leaky_relu')

    parser.add_argument('--layernormalize', type=int, default=0)
    parser.add_argument('--batchnormalize', type=int, default=0)

    # sparsity-related parameters
    parser.add_argument('--add_noise', type=int, default=0)
    parser.add_argument('--keep_noise', type=int, default=0)
    parser.add_argument('--noise_std', type=float, default=0.0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--keep_dim', type=int, default=30)
    
def add_saver_loger_params(parser):
    # saver_loger params
    parser.add_argument('--save_interval', type=int, default=1000)

    parser.add_argument('--save_dir', type=str, default='./save_dir')
    parser.add_argument('--log_dir', type=str, default='./log_dir')
    parser.add_argument('--load_dir', type=str, default=None)

    ## added those new
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--transfer_load', type=int, default=0)
    parser.add_argument('--freeze_weights', type=int, default=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environment_params(parser)
    add_optimization_params(parser)
    add_rollout_params(parser)
    add_network_params(parser)
    add_saver_loger_params(parser)

    ## evaluation
    parser.add_argument('--exp_name', type=str, default='m000')
    parser.add_argument('--specs_dir', type=str, default='./model_specs')
    parser.add_argument('--evaluation', type=int, default=0)
    parser.add_argument('--for_visuals', type=int, default=0)

    # in this way i can re-run experiments with different seeds
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    exp_name = args.exp_name

    args.save_dir = os.path.join(args.save_dir, exp_name)
    args.log_dir = os.path.join(args.log_dir, exp_name)

    if args.load_dir is None:
        args.load_dir = args.save_dir

    model_spec_dir = args.specs_dir
    os.makedirs(model_spec_dir, exist_ok=True)
    
    model_spec = os.path.join(model_spec_dir, '{}.json'.format(exp_name))

    with open(model_spec, 'w') as file:
        json.dump(vars(args), file)

    print(model_spec)