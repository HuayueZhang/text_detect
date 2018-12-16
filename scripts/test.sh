#!/usr/bin/env bash
export PYTHONPATH=~/pixel_link/code/pylib/src:$PYTHONPATH
set -x
set -e
export CUDA_VISIBLE_DEVICES=0
python test_pixel_link.py \
     --checkpoint_path=/home/zhy/pixel_link/models_dssd/model.ckpt-106772 \
     --gpu_idx=0 \
     --dataset_dir=/home/zhy/pixel_link/dataset/mtwi_2018/data_train_eval_split/split_eval/image_eval \
     --gpu_memory_fraction=-1

#--dataset_dir=/home/zhy/pixel_link/dataset/mtwi_2018/mtwi_2018_task3_test/icpr_mtwi_task3/image_test \
# --checkpoint_path=/home/zhy/pixel_link/pretrained_models/model.ckpt-73018 \