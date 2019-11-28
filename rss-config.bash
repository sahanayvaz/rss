#!/bin/bash

save_dir=$1
save_interval=50
# we will not employ any jacobian_loss not any entity_based loss
for level in 1 2 3 4
do
     early_final=$(($level * 6000000))
     exp_name="MARIO-$level-1-baseline-v0"
     env_id="SuperMarioBros-$level-1-v0"
     if [ $level == 4 ]; then
          early_final=$((2 * 6000000))
          exp_name="MARIO-M1-baseline-v0"
          env_id="SuperMarioBros-M$level-1-v0"
     fi

     python3 create_config.py --env_kind="mario" --env_id="SuperMarioBros-$level-1-v0" \
                              --NUM_ENVS=8 --NUM_LEVELS=0 \
                              --SET_SEED=0 --PAINT_VEL_INFO=0 --USE_DATA_AUGMENTATION=0 --GAME_TYPE="standard" \
                              --USE_BLACK_WHITE=0 --IS_HIGH_RES=0 --HIGH_DIFFICULTY=0 --nframeskip=4 \
                              --use_news=1 --lambda=0.95 --gamma=0.99 --nepochs=3 --nminibatches=8 \
                              --lr_lambda=0 --lr=1e-4 --cliprange_lambda=0 --cliprange=0.1 \
                              --norm_obs=1 --norm_adv=1 --norm_rew=1 --clip_rew=1 --ent_coeff=0.001 \
                              --vf_coef=1.0 --max_grad_norm=40.0 --num_timesteps=64000000 \
                              --early_final=$early_final \
                              --jacobian_loss=0 --beta_jacobian_loss=0 --tol_jacobian_loss=0 \
                              --entity_loss=0 --beta_entity_loss=0 \
                              --nsteps=128 \
                              --input_shape="84x84" --perception="nature_cnn" --feat_spec="feat_v0" \
                              --policy_spec="ls_c_v0" --activation="relu" \
                              --layernormalize=0 --batchnormalize=0 \
                              --attention=0 --recurrent=0 \
                              --add_noise=0 \
                              --for_visuals=1 \
                              --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                              --exp_name=$exp_name --evaluation=0
done

for policy_spec in "ls_c_v0" "ls_c_hh"
do
     exp="RSS"
     if [ $policy_spec == "ls_c_hh" ]; then
          exp="HH"
     fi
     exp_name="MARIO-1-1-$exp-v0"

     python3 create_config.py --env_kind="mario" --env_id="SuperMarioBros-1-1-v0" \
                              --NUM_ENVS=8 --NUM_LEVELS=0 \
                              --SET_SEED=0 --PAINT_VEL_INFO=0 --USE_DATA_AUGMENTATION=0 --GAME_TYPE="standard" \
                              --USE_BLACK_WHITE=0 --IS_HIGH_RES=0 --HIGH_DIFFICULTY=0 --nframeskip=4 \
                              --use_news=1 --lambda=0.95 --gamma=0.99 --nepochs=3 --nminibatches=8 \
                              --lr_lambda=0 --lr=1e-4 --cliprange_lambda=0 --cliprange=0.1 \
                              --norm_obs=1 --norm_adv=1 --norm_rew=1 --clip_rew=1 --ent_coeff=0.001 \
                              --vf_coef=1.0 --max_grad_norm=40.0 --num_timesteps=64000000 \
                              --early_final=6000000 \
                              --jacobian_loss=0 --beta_jacobian_loss=0 --tol_jacobian_loss=0 \
                              --entity_loss=0 --beta_entity_loss=0 \
                              --nsteps=128 \
                              --input_shape="84x84" --perception="nature_cnn" --feat_spec="feat_rss_v0" \
                              --policy_spec=$policy_spec --activation="relu" \
                              --layernormalize=0 --batchnormalize=0 \
                              --attention=0 --recurrent=0 \
                              --add_noise=0 \
                              --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                              --exp_name=$exp_name --evaluation=0
done