#!/bin/bash

# we will not employ any jacobian_loss not any entity_based loss
exp_name="rss-v0"
python3 create_config.py --env_kind="mario" --env_id="SuperMarioBros-1-1-v0" --test_id="SuperMarioBros-2-1-v0" \
                         --NUM_ENVS=8 --NUM_LEVELS=0 \
                         --SET_SEED=0 --PAINT_VEL_INFO=0 --USE_DATA_AUGMENTATION=0 --GAME_TYPE="standard" \
                         --USE_BLACK_WHITE=0 --IS_HIGH_RES=0 --HIGH_DIFFICULTY=0 --nframeskip=4 \
                         --use_news=1 --lambda=0.95 --gamma=0.99 --nepochs=3 --nminibatches=8 \
                         --lr_lambda=0 --lr=1e-4 --cliprange_lambda=0 --cliprange=0.1 \
                         --norm_obs=1 --norm_adv=1 --norm_rew=1 --clip_rew=1 --ent_coeff=0.001 \
                         --vf_coef=1.0 --max_grad_norm=40.0 --num_timesteps=64000000 \
                         --jacobian_loss=0 --beta_jacobian_loss=0 --tol_jacobian_loss=0 \
                         --entity_loss=0 --beta_entity_loss=0 \
                         --nsteps=128 \
                         --input_shape="84x84" --perception="nature_cnn" --policy_spec="ls_c_v0" --activation="relu" \
                         --layernormalize=0 --batchnormalize=0 \
                         --attention=0 --recurrent=0 \
                         --add_noise=0 \
                         --save_interval=1000 --save_dir="./save_dir" --log_dir="./log_dir" \
                         --exp_name=$exp_name --evaluation=0
