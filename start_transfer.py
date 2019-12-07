import os 
import argparse
import json

def transfer(args):
    HOME = args['HOME']
    SCRATCH = args['SCRATCH']

    log_dir = os.path.join(HOME, 'rss', 'log_dir')
    save_dir = os.path.join(SCRATCH, 'F-RSS-TR-save_dir')
    os.makedirs(save_dir, exist_ok=True)

    model_specs = os.path.join(HOME, 'rss', 'model_specs')
    # 41 in EULER, 0 and 17 in LEONHARD
    seeds = [0, 17, 41]
    transfer_dim = [25, 50, 100]
    levels = ['2-1', '3-1']
    base_name = ['MARIO-RSS-seed', 'MARIO-RSS-NOISE-seed']
    NL = 3
    KD = 50

    # we chose KN, because of the training performance
    KN = 25

    NSTD = 0.1
    load_iter = 1464

    total = 0
    for l in levels:
        level_name = 'SuperMarioBros-{}-v0'.format(l)
        for s in seeds:
            for t in transfer_dim:
                for b in base_name:
                    exp_name = '{}-{}-NL-{}-KD-{}-TR-{}-TRD-{}-v0'.format(b, s, NL, KD, l, t)
                    load_exp = '{}-{}-NL-{}-KD-{}-1-1-v0'.format(b, s, NL, KD)
                    
                    if b == 'MARIO-RSS-NOISE-seed':
                        load_exp = '{}-{}-NL-{}-KN-{}-NSTD-{}-1-1-v0'.format(b, s, NL, KN, NSTD)

                    try:
                        load_exp_json = os.path.join(model_specs, '{}.json'.format(load_exp))
                        with open(load_exp_json, 'r') as file:
                            train_args = json.load(file)
                        train_args['env_id'] = level_name
                        train_args['exp_name'] = exp_name
                        exp_save_dir = os.path.join(save_dir, exp_name)
                        exp_log_dir = os.path.join(log_dir, exp_name)
                        os.makedirs(exp_save_dir, exist_ok=True)
                        os.makedirs(exp_log_dir, exist_ok=True)
                        train_args['save_dir'] = exp_save_dir
                        train_args['log_dir'] = exp_log_dir
                        # train_args['load_dir'] = load_dir
                        train_args['transfer_load'] = 1
                        train_args['freeze_weights'] = 1

                        if b == 'MARIO-RSS-NOISE-seed':
                            train_args['add_noise'] = 0
                            train_args['keep_noise'] = 0
                            train_args['noise_std'] = 0.0

                        transfer_model_spec = os.path.join(model_specs, '{}.json'.format(exp_name))
                        with open(transfer_model_spec, 'w') as file:
                            json.dump(train_args, file)

                        if s == 0 or s == 17 or b == 'MARIO-RSS-NOISE-seed':
                            server_type = 'LEONHARD'
                            print('running exp: {}'.format(transfer_model_spec))
                            subcommand = "python3 run.py --server_type {} --model_spec {} --restore_iter {}".format(server_type, transfer_model_spec, load_iter)
                            command = "bsub -n 8 '{}'".format(subcommand)
                            os.system(command)

                        elif s == 41:
                            server_type = 'EULER'
                            print('running exp: {}'.format(transfer_model_spec))
                            subcommand = "python3 run.py --server_type {} --model_spec {} --restore_iter {}".format(server_type, transfer_model_spec, load_iter)
                            command = "bsub -n 8 '{}'".format(subcommand)
                            os.system(command)
                    
                    except:
                        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--HOME', default='./')
    parser.add_argument('--SCRATCH', default='./')
    args = parser.parse_args()
    transfer(vars(args))