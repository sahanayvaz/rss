#!/bin/bash

save_dir=$1
save_interval=500
server_type=$2

for seed in 0 17 41
do
     num_timesteps=256000000
     early_final=128000000
     for num_layers in 1 2
     do
          for keep_dim in 50
          do
               exp_name="COINRUN-RSS-seed-$seed-NL-$num_layers-KD-$keep_dim-1-1-v0"
               exp_path="$HOME/rss/model_specs/$exp_name.json"
               if [ $seed -eq 17 ] && [ $server_type == "LEONHARD" ]; then
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
                                             --add_noise=0 --keep_noise=0 --noise_std=0.0 \
                                             --policy_spec="full_sparse" --activation="relu" \
                                             --layernormalize=0 --batchnormalize=0 \
                                             --for_visuals=0 \
                                             --seed=$seed\
                                             --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                                             --specs_dir='./model_specs'\
                                             --exp_name=$exp_name --evaluation=0
                    bsub -n 20 "python3 run.py --server_type $server_type --visualize 0 --model_spec $exp_path"

               elif [ $seed -eq 41 ] && [ $server_type == "EULER" ]; then
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
                                             --add_noise=0 --keep_noise=0 --noise_std=0.0 \
                                             --policy_spec="full_sparse" --activation="relu" \
                                             --layernormalize=0 --batchnormalize=0 \
                                             --for_visuals=0 \
                                             --seed=$seed\
                                             --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                                             --specs_dir='./model_specs'\
                                             --exp_name=$exp_name --evaluation=0
                    bsub -n 20 "python3 run.py --server_type $server_type --visualize 0 --model_spec $exp_path"

               elif [ $seed -eq 0 ] && [ $server_type == "EULER" ]; then
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
                                             --add_noise=0 --keep_noise=0 --noise_std=0.0 \
                                             --policy_spec="full_sparse" --activation="relu" \
                                             --layernormalize=0 --batchnormalize=0 \
                                             --for_visuals=0 \
                                             --seed=$seed\
                                             --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                                             --specs_dir='./model_specs'\
                                             --exp_name=$exp_name --evaluation=0
                    bsub -n 20 "python3 run.py --server_type $server_type --visualize 0 --model_spec $exp_path"
               fi

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
                                        --add_noise=0 --keep_noise=0 --noise_std=0.0 \
                                        --policy_spec="full_sparse" --activation="relu" \
                                        --layernormalize=0 --batchnormalize=0 \
                                        --for_visuals=0 \
                                        --seed=$seed\
                                        --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                                        --specs_dir='./model_specs'\
                                        --exp_name=$exp_name --evaluation=0
          done
     done
done