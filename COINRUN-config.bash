#!/bin/bash

save_dir=$1
save_interval=25
server_type=$2

for seed in 0 17 41
do
     exp_name='COINRUN-baseline-seed-$seed-100'
     exp_path="$HOME/rss/model_specs/$exp_name.json"
     python3 create_config.py --env_kind="coinrun" --env_id="coinrun" --NUM_ENVS=32 --NUM_LEVELS=100 \
                              --SET_SEED=13 --PAINT_VEL_INFO=1 --USE_DATA_AUGMENTATION=0 --GAME_TYPE="standard" \
                              --USE_BLACK_WHITE=0 --IS_HIGH_RES=0 --HIGH_DIFFICULTY=0 --nframeskip=1 \
                              --use_news=1 --lambda=0.95 --gamma=0.999 --nepochs=3 --nminibatches=8 \
                              --lr_lambda=1 --lr=5e-4 --cliprange_lambda=1 --cliprange=0.2 \
                              --norm_obs=0 --norm_adv=1 --norm_rew=0 --clip_rew=0 --ent_coeff=0.01 \
                              --vf_coef=0.5 --max_grad_norm=0.5 --num_timesteps=$num_timesteps \
                              --early_final=$early_final \
                              --nsteps=256 \
                              --input_shape="64x64" --perception="nature_cnn" --feat_spec="feat_v0" \
                              --keep_dim=0 --num_layers=0 \
                              --add_noise=0 --keep_noise=0 --noise_std=0.0 \
                              --policy_spec="cr_fc_v0" --activation="relu" \
                              --layernormalize=0 --batchnormalize=0 \
                              --for_visuals=0 \
                              --seed=$seed\
                              --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                              --specs_dir='./model_specs'\
                              --exp_name=$exp_name --evaluation=0

     bsub -n 20 -W 8:00 -R "rusage[mem=512]" "python3 run.py --server_type $server_type --visualize 0 --model_spec $exp_path"
done

for seed in 0 17 41
do
     num_timesteps=256000000
     early_final=20000000
     for num_layers in 1 2
     do
          keep_dim=100
          for keep_noise in 25 50
          do
               for noise_std in 0.01 0.1
               do
                    exp_name="COINRUN-RSS-NOISE-seed-$seed-NL-$num_layers-KN-$keep_noise-NSTD-$noise_std-1-1-v0"
                    exp_path="$HOME/rss/model_specs/$exp_name.json"
                    python3 create_config.py --env_kind="coinrun" --env_id="coinrun" --NUM_ENVS=32 --NUM_LEVELS=100 \
                                             --SET_SEED=13 --PAINT_VEL_INFO=1 --USE_DATA_AUGMENTATION=0 --GAME_TYPE="standard" \
                                             --USE_BLACK_WHITE=0 --IS_HIGH_RES=0 --HIGH_DIFFICULTY=0 --nframeskip=1 \
                                             --use_news=1 --lambda=0.95 --gamma=0.999 --nepochs=3 --nminibatches=8 \
                                             --lr_lambda=1 --lr=5e-4 --cliprange_lambda=1 --cliprange=0.2 \
                                             --norm_obs=0 --norm_adv=1 --norm_rew=0 --clip_rew=0 --ent_coeff=0.01 \
                                             --vf_coef=0.5 --max_grad_norm=0.5 --num_timesteps=$num_timesteps \
                                             --early_final=$early_final \
                                             --nsteps=256 \
                                             --input_shape="64x64" --perception="nature_cnn" --feat_spec="feat_rss_v0" \
                                             --keep_dim=$keep_dim --num_layers=$num_layers \
                                             --add_noise=1 --keep_noise=$keep_noise --noise_std=$noise_std \
                                             --policy_spec="full_sparse" --activation="relu" \
                                             --layernormalize=0 --batchnormalize=0 \
                                             --for_visuals=0 \
                                             --seed=$seed\
                                             --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                                             --specs_dir='./model_specs'\
                                             --exp_name=$exp_name --evaluation=0
                    bsub -n 20 -W 8:00 -R "rusage[mem=512]" "python3 run.py --server_type $server_type --visualize 0 --model_spec $exp_path"
               done
          done
     done
done
