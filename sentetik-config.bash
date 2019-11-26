#!/bin/bash

# we will not employ any jacobian_loss not any entity_based loss
exp_name="sentetik-v0"
python3 create_config.py --env_kind="coinrun" --env_id="coinrun" --NUM_ENVS=32 --NUM_LEVELS=500 \
                         --SET_SEED=13 --PAINT_VEL_INFO=1 --USE_DATA_AUGMENTATION=0 --GAME_TYPE="standard" \
                         --USE_BLACK_WHITE=0 --IS_HIGH_RES=0 --HIGH_DIFFICULTY=0 --nframeskip=1 \
                         --use_news=1 --lambda=0.95 --gamma=0.999 --nepochs=3 --nminibatches=8 \
                         --lr_lambda=1 --lr=5e-4 --cliprange_lambda=1 --cliprange=0.2 \
                         --norm_obs=0 --norm_adv=1 --norm_rew=0 --clip_rew=0 --ent_coeff=0.01 \
                         --vf_coef=0.5 --max_grad_norm=0.5 --num_timesteps=256000000 \
                         --jacobian_loss=0 --beta_jacobian_loss=0.0 --tol_jacobian_loss=0.0 \
                         --entity_loss=0 --beta_entity_loss=0.0 --tol_entity_loss=0.0 \
                         --nsteps=256 \
                         --input_shape="64x64" --perception="nature_cnn" \
                         --policy_spec="cr_fc_v0" --activation="relu" \
                         --layernormalize=0 --batchnormalize=0 \
                         --attention=0 --recurrent=0 \
                         --num_traj_rep=2 --cap_buf=50000 --context_dim=256 \
                         --save_interval=2500 --save_dir="./save_dir" --log_dir="./log_dir" \
                         --exp_name=$exp_name --evaluation=0

exp_name="sentetik-v1"
python3 create_config.py --env_kind="coinrun" --env_id="coinrun" --NUM_ENVS=32 --NUM_LEVELS=500 \
                         --SET_SEED=13 --PAINT_VEL_INFO=1 --USE_DATA_AUGMENTATION=0 --GAME_TYPE="standard" \
                         --USE_BLACK_WHITE=0 --IS_HIGH_RES=0 --HIGH_DIFFICULTY=0 --nframeskip=1 \
                         --use_news=1 --lambda=0.95 --gamma=0.999 --nepochs=3 --nminibatches=8 \
                         --lr_lambda=1 --lr=5e-4 --cliprange_lambda=1 --cliprange=0.2 \
                         --norm_obs=0 --norm_adv=1 --norm_rew=0 --clip_rew=0 --ent_coeff=0.01 \
                         --vf_coef=0.5 --max_grad_norm=0.5 --num_timesteps=256000000 \
                         --jacobian_loss=0 --beta_jacobian_loss=0.0 --tol_jacobian_loss=0.0 \
                         --entity_loss=0 --beta_entity_loss=0.0 --tol_entity_loss=0.0 \
                         --nsteps=256 \
                         --input_shape="64x64" --perception="nature_cnn" \
                         --policy_spec="cr_fc_v0" --activation="relu" \
                         --layernormalize=0 --batchnormalize=0 \
                         --attention=0 --recurrent=0 \
                         --num_traj_rep=2 --cap_buf=50000 --context_dim=256 \
                         --update_freq=250 \
                         --save_interval=2500 --save_dir="./save_dir" --log_dir="./log_dir" \
                         --exp_name=$exp_name --evaluation=0