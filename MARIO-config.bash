#!/bin/bash

save_dir=$1
save_interval=25

# baselines
for level in 1 2 3
do
     for seed in 0 17 41
     do
          early_final=$(($level * 6000000))
          exp_name="MARIO-baseline-seed-$seed-$level-1-v0"
          env_id="SuperMarioBros-$level-1-v0"
          
          python3 create_config.py --env_kind="mario" --env_id=$env_id \
                                   --NUM_ENVS=8 --NUM_LEVELS=0 \
                                   --SET_SEED=0 --PAINT_VEL_INFO=0 --USE_DATA_AUGMENTATION=0 --GAME_TYPE="standard" \
                                   --USE_BLACK_WHITE=0 --IS_HIGH_RES=0 --HIGH_DIFFICULTY=0 --nframeskip=4 \
                                   --use_news=1 --lambda=0.95 --gamma=0.99 --nepochs=3 --nminibatches=8 \
                                   --lr_lambda=0 --lr=1e-4 --cliprange_lambda=0 --cliprange=0.1 \
                                   --norm_obs=1 --norm_adv=1 --norm_rew=1 --clip_rew=1 --ent_coeff=0.001 \
                                   --vf_coef=1.0 --max_grad_norm=40.0 --num_timesteps=64000000 \
                                   --early_final=$early_final \
                                   --nsteps=128 \
                                   --input_shape="84x84" --perception="nature_cnn" --feat_spec="feat_v0" \
                                   --policy_spec="ls_c_v0" --activation="relu" \
                                   --layernormalize=0 --batchnormalize=0 \
                                   --add_noise=0 \
                                   --for_visuals=0 \
                                   --seed=$seed\
                                   --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                                   --specs_dir='./model_spec'\
                                   --exp_name=$exp_name --evaluation=0
                                   
          exp_path="$HOME/rss/model_spec/$exp_name.json"
          bsub -n 16 "python3 run.py --server_type LEONHARD --visualize 0 --model_spec $exp_path"
     done
done

for seed in 0 17 41
do
     early_final=6000000
     env_id="SuperMarioBros-1-1-v0"
     for num_layers in 2 3 5
     do
          for keep_dim in 50 150 250
          do
               exp_name="MARIO-RSS-seed-$seed-NL-$num_layers-KD-$keep_dim-1-1-v0"
               python3 create_config.py --env_kind="mario" --env_id=$env_id \
                                        --NUM_ENVS=8 --NUM_LEVELS=0 \
                                        --SET_SEED=0 --PAINT_VEL_INFO=0 --USE_DATA_AUGMENTATION=0 --GAME_TYPE="standard" \
                                        --USE_BLACK_WHITE=0 --IS_HIGH_RES=0 --HIGH_DIFFICULTY=0 --nframeskip=4 \
                                        --use_news=1 --lambda=0.95 --gamma=0.99 --nepochs=3 --nminibatches=8 \
                                        --lr_lambda=0 --lr=1e-4 --cliprange_lambda=0 --cliprange=0.1 \
                                        --norm_obs=1 --norm_adv=1 --norm_rew=1 --clip_rew=1 --ent_coeff=0.001 \
                                        --vf_coef=1.0 --max_grad_norm=40.0 --num_timesteps=64000000 \
                                        --early_final=$early_final \
                                        --nsteps=128 \
                                        --input_shape="84x84" --perception="nature_cnn" --feat_spec="feat_rss_v0" \
                                        --keep_dim=$keep_dim --num_layers=$num_layers \
                                        --add_noise=0 --keep_noise=0 --noise_std=0.0 \
                                        --policy_spec="full_sparse" --activation="relu" \
                                        --layernormalize=0 --batchnormalize=0 \
                                        --for_visuals=0 \
                                        --seed=$seed\
                                        --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                                        --specs_dir='./model_spec'\
                                        --exp_name=$exp_name --evaluation=0

               exp_path="$HOME/rss/model_spec/$exp_name.json"
               bsub -n 16 "python3 run.py --server_type LEONHARD --visualize 0 --model_spec $exp_path"
          done
     done
done