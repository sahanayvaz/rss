import os
import json
import argparse

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_config', type=str, default=None)
    parser.add_argument('--divisor', type=int, default=2500)
    args = parser.parse_args()

    with open(args.train_config, 'r') as file:
        train_args = json.load(file)

    # make this more automatic
    iter_all = [int(model.split('.')[0].split('-')[-1]) for model in os.listdir(train_args['save_dir']) if 'model' in model]
    iter_all = list(set(iter_all))

    load_iter = []
    for l in iter_all:
        # i do not want the 0th model
        if (l % args.divisor == 0) and (l != 0):
            load_iter.append(l)

    load_iter.sort()

    load_path = [os.path.join(train_args['save_dir'], 'model-{}'.format(l)) for l in load_iter]

    train_args['log_dir'] = './log_dir/evals'

    train_dir = os.path.join(train_args['save_dir'], 'eval')
    os.makedirs(train_dir, exist_ok=True)
    train_path = os.path.join(train_dir, 'train-eval.json')

    # do NOT forget to change this
    # train_args['env_id'] = 'coinrun'

    train_args['load_path'] = load_path
    train_args['evaluation'] = 1
    train_args['eval_type'] = 'train-eval'
    train_args['IS_HIGH_RES'] = 1
    with open(train_path, 'w') as file:
        json.dump(train_args, file)

    env_kind = train_args['env_kind']
    if env_kind == 'coinrun':
        eval_path = os.path.join(train_dir, 'test-eval-0.json')
        test_levels = 1000
        train_args['NUM_LEVELS'] = test_levels
        train_args['HIGH_DIFFICULTY'] = 0
        train_args['SET_SEED'] = 17
        train_args['eval_type'] = 'test-eval'

        with open(eval_path, 'w') as file:
            json.dump(train_args, file)

        eval_path = os.path.join(train_dir, 'test-eval-1.json')
        train_args['HIGH_DIFFICULTY'] = 1

        with open(eval_path, 'w') as file:
            json.dump(train_args, file)

    elif env_kind == 'mario':
        eval_path = os.path.join(train_dir, 'test-eval-0.json')
        train_args['eval_type'] = 'test-eval'
        train_args['env_id'] = "SuperMarioBros-2-1-v0"
        with open(eval_path, 'w') as file:
            json.dump(train_args, file)
