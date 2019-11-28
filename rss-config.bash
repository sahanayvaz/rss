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


for level in 2 3
do
     early_final=$((($level-1) * 6000000))
     exp_name="MARIO-1-1-baseline-TR-$level-1-v0"
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
                              --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                              --load_dir="$save_dir/MARIO-1-1-baseline-v0/" --transfer_load=1 --freeze_weights=0 \
                              --exp_name=$exp_name --evaluation=0
done

for level in 2 3
do
     early_final=$((($level-1) * 6000000))
     exp_name="MARIO-1-1-RSS-TR-$level-1-v0"
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
                              --input_shape="84x84" --perception="nature_cnn" --feat_spec="feat_rss_v0" \
                              --policy_spec="ls_c_v0" --activation="relu" \
                              --layernormalize=0 --batchnormalize=0 \
                              --attention=0 --recurrent=0 \
                              --add_noise=0 \
                              --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                              --load_dir="$save_dir/MARIO-1-1-RSS-v0/" --transfer_load=1 --freeze_weights=1 \
                              --exp_name=$exp_name --evaluation=0
done


for level in 2 3
do
     early_final=$((($level-1) * 6000000))
     exp_name="MARIO-1-1-HH-TR-$level-1-v0"
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
                              --input_shape="84x84" --perception="nature_cnn" --feat_spec="feat_rss_v0" \
                              --policy_spec="ls_c_hh" --activation="relu" \
                              --layernormalize=0 --batchnormalize=0 \
                              --attention=0 --recurrent=0 \
                              --add_noise=0 \
                              --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                              --load_dir="$save_dir/MARIO-1-1-HH-v0/" --transfer_load=1 --freeze_weights=1 \
                              --exp_name=$exp_name --evaluation=0
done


for exp in baseline RSS HH
do
     freeze_weights=1
     feat_spec="feat_rss_v0"
     if [ $exp == baseline ]; then
          feat_spec="feat_v0"
          freeze_weights=0
     fi

     policy_spec="ls_c_v0"
     if [ $exp == HH ]; then
          policy_spec="ls_c_hh"
     fi

     early_final=6000000
     exp_name="MARIO-1-1-$exp-TR-2-2-v0"
     python3 create_config.py --env_kind="mario" --env_id="SuperMarioBros-2-2-v0" \
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
                              --input_shape="84x84" --perception="nature_cnn" --feat_spec=$feat_spec \
                              --policy_spec=$policy_spec --activation="relu" \
                              --layernormalize=0 --batchnormalize=0 \
                              --attention=0 --recurrent=0 \
                              --add_noise=0 \
                              --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                              --load_dir="$save_dir/MARIO-1-1-$exp-v0/" --transfer_load=1 \
                              --freeze_weights=$freeze_weights \
                              --exp_name=$exp_name --evaluation=0
done

exp_name=MARIO-2-2-baseline-v0
python3 create_config.py --env_kind="mario" --env_id="SuperMarioBros-2-2-v0" \
                         --NUM_ENVS=8 --NUM_LEVELS=0 \
                         --SET_SEED=0 --PAINT_VEL_INFO=0 --USE_DATA_AUGMENTATION=0 --GAME_TYPE="standard" \
                         --USE_BLACK_WHITE=0 --IS_HIGH_RES=0 --HIGH_DIFFICULTY=0 --nframeskip=4 \
                         --use_news=1 --lambda=0.95 --gamma=0.99 --nepochs=3 --nminibatches=8 \
                         --lr_lambda=0 --lr=1e-4 --cliprange_lambda=0 --cliprange=0.1 \
                         --norm_obs=1 --norm_adv=1 --norm_rew=1 --clip_rew=1 --ent_coeff=0.001 \
                         --vf_coef=1.0 --max_grad_norm=40.0 --num_timesteps=64000000 \
                         --early_final=12000000 \
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

for exp in RSS HH
do
     for level in 2-1 3-1 2-2
     do
          freeze_weights=0
          feat_spec="feat_rss_v0"

          policy_spec="ls_c_v0"
          if [ $exp == HH ]; then
               policy_spec="ls_c_hh"
          fi

          early_final=6000000
          exp_name="MARIO-1-1-$exp-NOF-TR-$level-v0"
          python3 create_config.py --env_kind="mario" --env_id="SuperMarioBros-$level-v0" \
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
                                   --input_shape="84x84" --perception="nature_cnn" --feat_spec=$feat_spec \
                                   --policy_spec=$policy_spec --activation="relu" \
                                   --layernormalize=0 --batchnormalize=0 \
                                   --attention=0 --recurrent=0 \
                                   --add_noise=0 \
                                   --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                                   --load_dir="$save_dir/MARIO-1-1-$exp-v0/" --transfer_load=1 \
                                   --freeze_weights=$freeze_weights \
                                   --exp_name=$exp_name --evaluation=0
     done
done

exp_name='coinrun-RSS-v0'
python3 create_config.py --env_kind="coinrun" --env_id="coinrun" --NUM_ENVS=32 --NUM_LEVELS=500 \
                         --SET_SEED=13 --PAINT_VEL_INFO=1 --USE_DATA_AUGMENTATION=0 --GAME_TYPE="standard" \
                         --USE_BLACK_WHITE=0 --IS_HIGH_RES=0 --HIGH_DIFFICULTY=0 --nframeskip=1 \
                         --use_news=1 --lambda=0.95 --gamma=0.999 --nepochs=3 --nminibatches=8 \
                         --lr_lambda=1 --lr=5e-4 --cliprange_lambda=1 --cliprange=0.2 \
                         --norm_obs=0 --norm_adv=1 --norm_rew=0 --clip_rew=0 --ent_coeff=0.01 \
                         --vf_coef=0.5 --max_grad_norm=0.5 --num_timesteps=256000000 \
                         --early_final=256000000 \
                         --jacobian_loss=0 --beta_jacobian_loss=0.0 --tol_jacobian_loss=0.0 \
                         --entity_loss=0 --beta_entity_loss=0.0 --tol_entity_loss=0.0 \
                         --nsteps=256 \
                         --input_shape="64x64" --perception="nature_cnn" --feat_spec="feat_rss_v0" \
                         --policy_spec="cr_fc_v0" --activation="relu" \
                         --layernormalize=0 --batchnormalize=0 \
                         --attention=0 --recurrent=0 \
                         --add_noise=0 \
                         --save_interval=1000 --save_dir="./save_dir" --log_dir="./log_dir" \
                         --exp_name=$exp_name --evaluation=0



for exp in baseline RSS HH
do
     freeze_weights=0
     feat_spec="feat_rss_v0"
     if [ $exp == baseline ]; then
          feat_spec="feat_v0"
          freeze_weights=0
     fi

     policy_spec="ls_c_v0"
     if [ $exp == HH ]; then
          policy_spec="ls_c_hh"
     fi

     early_final=0
     exp_name="MARIO-2-2-$exp-FORGET-1-1-v0"
     python3 create_config.py --env_kind="mario" --env_id="SuperMarioBros-1-1-v0" \
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
                              --input_shape="84x84" --perception="nature_cnn" --feat_spec=$feat_spec \
                              --policy_spec=$policy_spec --activation="relu" \
                              --layernormalize=0 --batchnormalize=0 \
                              --attention=0 --recurrent=0 \
                              --add_noise=0 \
                              --save_interval=$save_interval --save_dir=$save_dir --log_dir="./log_dir" \
                              --load_dir="$save_dir/MARIO-1-1-$exp-TR-2-2-v0/" --transfer_load=1 \
                              --freeze_weights=$freeze_weights \
                              --exp_name=$exp_name --evaluation=0
done